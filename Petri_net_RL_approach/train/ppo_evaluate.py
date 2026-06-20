"""
ppo_evaluate.py
---------------
Evaluate the PPO model on the complementary split of K_TRAIN cases.
"""

import ast
import os
import sys
import time
import signal
import yaml

import pandas as pd
import pm4py
import torch
from pm4py.objects.log.importer.xes import importer as xes_importer

# ── Load config ───────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CFG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

_p2    = _cfg["phase2"]
_sch   = _p2["schedule"]
_paths = _cfg["paths"]

PROJECT_ROOT = _paths["project_root"]
DS_CSV       = _paths["ds_csv"]
PNML_PATH    = _paths["pnml_path"]
PPO_PT       = _cfg["evaluate"]["ppo_pt"]
EVAL_OUT_CSV = os.path.join(os.path.dirname(PPO_PT), "/ppo_eval_results.csv")
K_TRAIN      = _sch["k_train"]
MAX_STEPS    = _sch["max_steps"]   # hard cap on decoding steps
XES_PATH         = _paths["xes_path"]
sys.path.append(PROJECT_ROOT)

from ppo_env import AlignmentEnv
from model.ppo_model import ActorCritic
from model.model import Vocab

sys.modules["__main__"].Vocab    = Vocab
sys.modules["model"].Vocab       = Vocab
sys.modules["model.model"].Vocab = Vocab


# ── Timeout helper (works on Windows too via threading) ───────────────────────
import threading

class TimeoutError(Exception):
    pass

def run_with_timeout(fn, args=(), kwargs={}, seconds=10):
    """Run fn(*args, **kwargs) in a thread; raise TimeoutError if it takes too long."""
    result    = [None]
    exception = [None]

    def target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=seconds)

    if t.is_alive():
        raise TimeoutError(f"generate() exceeded {seconds}s")
    if exception[0] is not None:
        raise exception[0]
    return result[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load dataset ──────────────────────────────────────────────────────────
    df = pd.read_csv(DS_CSV, dtype={"case_id": str})
    df["prefix_activities"] = df["prefix_activities"].apply(ast.literal_eval)
    df = df[df["aligned_prefix"].notna()]
    df["aligned_prefix"] = df["aligned_prefix"].apply(ast.literal_eval)
    df["step_types"]     = df["step_types"].apply(ast.literal_eval)

    log_all = xes_importer.apply(XES_PATH)
    traces_fitnes_list = [
        float(trace.attributes.get("trace_fitness"))
        for trace in log_all
    ]

    threshold = 0.9
    eval_cases = [
        trace for i, trace in enumerate(log_all)
        if 0.85 < traces_fitnes_list[i] < threshold
    ]
    print(f"{len(eval_cases)} traces in (0.85, {threshold}) out of {len(log_all)} total")

    eval_cases_ids = {
        str(trace.attributes["concept:name"]).strip()
        for trace in eval_cases
    }
    df["case_id"] = df["case_id"].str.strip()

    df_eval = df[df["case_id"].isin(eval_cases_ids)].reset_index(drop=True)
    cases   = df_eval["case_id"].unique()
    print(f"Evaluating (PPO) on {len(cases)} cases, {len(df_eval)} rows")
    # ── Petri net ─────────────────────────────────────────────────────────────
    net, im, fm = pm4py.read_pnml(PNML_PATH)
    sink_place  = next(p for p in net.places if p.name == "sink")
    fm          = pm4py.generate_marking(net, sink_place)
    labels      = [t.label for t in net.transitions if t.label is not None]
    env         = AlignmentEnv(net, im, labels)

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab = Vocab()
    for label in env.LABEL_SPACE:
        vocab.add(label)

    # ── Load PPO model ────────────────────────────────────────────────────────
    ckpt  = torch.load(PPO_PT, map_location="cpu", weights_only=False)
    model = ActorCritic(len(vocab), env.n_places, len(env.LABEL_SPACE))
    model.load_state_dict(ckpt["state"], strict=True)
    model.eval()
    print(f"Loaded PPO weights from: {PPO_PT}")
    @torch.no_grad()
    def generate_eval(*args, **kwargs):
        return model.generate(*args, **kwargs)
# ── Evaluate ──────────────────────────────────────────────────────────────
    write_header = not os.path.exists(EVAL_OUT_CSV)  # append if file already exists
    skipped      = 0
    timed_out    = 0
    processed    = 0
    total_rows   = len(df_eval)
    t_eval_start = time.perf_counter()

    os.makedirs(os.path.dirname(EVAL_OUT_CSV), exist_ok=True)
    print(f"Writing results to: {EVAL_OUT_CSV}")
    write_header = not os.path.exists(EVAL_OUT_CSV)
    skipped      = 0
    timed_out    = 0
    processed    = 0
    total_rows   = len(df_eval)
    t_eval_start = time.perf_counter()

    for case_id, case_df in df_eval.groupby("case_id", sort=False):
        case_df    = case_df.sort_values("prefix_length")
        case_start = time.perf_counter()
        case_skip  = False

        for _, row in case_df.iterrows():
            prefix = row["prefix_activities"]
            GT_activity_labels = row['aligned_prefix'] 
            GT_move_types     = row['step_types']
            processed += 1

            if not prefix:
                skipped += 1
                continue

            if case_skip:
                skipped += 1
                continue

            src = torch.tensor([vocab.encode(prefix)])

            t0 = time.perf_counter()
            try:
                result  = run_with_timeout(
                    fn      = generate_eval,                 # was: model.generate
                    args    = (src, prefix, env, vocab),
                    kwargs  = {"train": False, "compute_reward": False, "max_len": MAX_STEPS},
                    seconds = 10
                )
                traj, _ = result
                elapsed  = time.perf_counter() - t0

            except TimeoutError:
                elapsed = time.perf_counter() - t0
                print(f"  [TIMEOUT] case={case_id}  prefix_len={len(prefix)}"
                      f"  after {elapsed:.1f}s  → skipping rest of case")
                timed_out += 1
                skipped   += 1
                case_skip  = True
                continue

            except Exception as e:
                print(f"  [ERROR] case={case_id}  prefix_len={len(prefix)}  {e}")
                skipped += 1
                continue

            if not traj or not traj["rewards"]:
                skipped += 1
                continue

            moves_str  = traj["moves_str"]
            labels_str = traj["labels_str"]
            n_sync     = moves_str.count("S")
            n_log      = moves_str.count("L")
            n_model    = moves_str.count("M")
            cost       = n_log + n_model

            # ── Write immediately, one row at a time ──────────────────────────
            pd.DataFrame([{
                "case_id":            case_id,
                "prefix_length":      row["prefix_length"],
                "prefix_activities":  str(prefix),
                "alignment_time_sec": elapsed,
                "steps_taken":        len(moves_str),
                "cost":               cost,
                "sync_moves":         n_sync,
                "log_moves":          n_log,
                "model_moves":        n_model,
                "is_conforming":      (cost == 0),
                "labels_str":         str(labels_str),
                "moves_str":          str(moves_str),
            }]).to_csv(EVAL_OUT_CSV, mode="a", header=write_header, index=False)
            write_header = False   # only True for the very first write

            if processed % 100 == 0:
                elapsed_total = time.perf_counter() - t_eval_start
                rate          = processed / elapsed_total
                remaining     = (total_rows - processed) / rate if rate > 0 else float("inf")
                print(
                    f"  [{processed}/{total_rows}]  "
                    f"skipped={skipped}  timed_out={timed_out}  "
                    f"elapsed={elapsed_total:.1f}s  "
                    f"rate={rate:.1f} rows/s  "
                    f"ETA={remaining:.1f}s"
                )
                print()
                print("ground truth moves types : ", GT_move_types)
                print("ground truth activity labels : ", prefix)
                
                print(" moves types : ", moves_str)
                print("generated activity labels  : ",labels_str)
                # print()

        case_elapsed = time.perf_counter() - case_start
        # status = "SKIPPED (timeout)" if case_skip else "done"
        # print(f"Case {case_id} {status} — {len(case_df)} prefixes in {case_elapsed:.2f}s")

    total_elapsed = time.perf_counter() - t_eval_start
    print(f"\nDone.  processed={processed}  skipped={skipped}  timed_out={timed_out}")
    print(f"Total time : {total_elapsed:.1f}s")
    print(f"Results    -> {EVAL_OUT_CSV}")


if __name__ == "__main__":
    main()
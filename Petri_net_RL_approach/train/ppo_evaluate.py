import ast
import sys
import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pm4py
import yaml

# ── Load config ───────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(_CFG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

_p2    = _cfg["phase2"]
_hp    = _p2["hyperparameters"]
_sch   = _p2["schedule"]
_paths = _cfg["paths"]

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = _paths["project_root"]
DS_CSV           = _paths["ds_csv"]
PNML_PATH        = _paths["pnml_path"]
MODEL_PHASE1_OUT = _p2["model_phase1_out"]   # checkpoint to load 
PPO_OUT          = _p2["ppo_out"]            

# ── PPO hyper-parameters ──────────────────────────────────────────────────────
GAMMA    = _hp["gamma"]
LAM      = _hp["lam"]
CLIP     = _hp["clip"]
ENT_COEF = _hp["ent_coef"]
VF_COEF  = _hp["vf_coef"]
LR       = _hp["lr"]
MAX_GRAD = _hp["max_grad"]

# ── Training schedule ─────────────────────────────────────────────────────────
PPO_EPOCHS = _sch["ppo_epochs"]
EPISODES   = _sch["episodes"]
BATCH_SIZE = _sch["batch_size"]
MAX_STEPS  = _sch["max_steps"]
K_TRAIN    = _sch["k_train"]

# ── Project path ──────────────────────────────────────────────────────────────
sys.path.append(PROJECT_ROOT)

from ppo_env import AlignmentEnv
from model.ppo_model import ActorCritic
from model.model import Vocab

sys.modules['__main__'].Vocab    = Vocab
sys.modules['model'].Vocab       = Vocab
sys.modules['model.model'].Vocab = Vocab

def validate_alignment(generated, prefix, env):

    env.reset(prefix)
    
    n_invalid  = 0
    log_moves  = 0
    model_moves = 0
    sync_moves = 0

    for move, label in zip(generated['moves_str'], generated['labels_str']):
        enabled_labels = {t.label for t in env.real_enabled_visible()}

        if move == "S":
            act = env.current_activity()
            if label != act or label not in enabled_labels:
                n_invalid += 1
                sync_moves += 1
            else:
                sync_moves += 1

        elif move == "M":
            if label not in enabled_labels:
                n_invalid += 1
            model_moves += 1

        elif move == "L":
            log_moves += 1

        move_id  = env.MOVE_SPACE.index(move)
        label_id = env.LABEL_SPACE.index(label) if label in env.LABEL_SPACE else 0
        env.step(move_id, label_id)

    is_valid      = (n_invalid == 0)
    is_conforming = (log_moves == 0 and model_moves == 0)  
    return {
        "is_valid":      is_valid,
        "is_conforming": is_conforming,
        "n_invalid":     n_invalid,
        "sync_moves":    sync_moves,
        "model_moves":   model_moves,
        "log_moves":     log_moves,
        "cost":          log_moves + model_moves, 
    }

def evaluate_case(case_id, case_df, model, vocab, env):
    case_df = case_df.sort_values("prefix_length")
    rows    = list(case_df.iterrows())

    print("\n" + "=" * 70)
    print(f"CASE {case_id}")
    print("=" * 70)

    total_cost_delta = 0
    exact_match_count = 0

    for _, row in rows:
        prefix   = row["prefix_activities"]
        gt_cost  = int(row["cost"])
        gt_steps = ast.literal_eval(row["step_types"]) if isinstance(row["step_types"], str) else row["step_types"]
        gt_aligned = ast.literal_eval(row["aligned_prefix"]) if isinstance(row["aligned_prefix"], str) else row["aligned_prefix"]
        k        = row["prefix_length"]

        src_ids   = torch.tensor([vocab.encode(prefix)])
        generated, _ = model.generate(src_ids, prefix, env, vocab)
        val = validate_alignment(generated, prefix, env)

        gen_moves  = [mv for mv in generated['moves_str']]
        gen_labels = [label for label in generated['labels_str']]
        gen_cost   = sum(1 for mv in gen_moves if mv != "S")

        cost_delta        = gen_cost - gt_cost
        total_cost_delta += cost_delta

        moves_match  = (gen_moves == gt_steps)
        labels_match = (gen_labels == gt_aligned)
        exact_match  = moves_match and labels_match
        exact_match_count += int(exact_match)

        print(f"\nPrefix {k}: {prefix}")
        print(f"  GT  steps  : {gt_steps}")
        print(f"  GEN steps  : {gen_moves}")
        print(f"  GT  labels : {gt_aligned}")
        print(f"  GEN labels : {gen_labels}")
        print(f"  Exact match    : {'YES' if exact_match  else 'NO'}")
        print(f"  GT cost: {gt_cost}  |  GEN cost: {gen_cost}  |  Δ: {cost_delta}")

    n = len(rows)
    print("\n--- Summary ---")
    print(f"Avg cost delta : {total_cost_delta / n:.4f}")
    print(f"Exact match    : {exact_match_count}/{n}")

K_TRAIN = 100
def main():
    ckpt     = torch.load(PPO_OUT, map_location="cpu", weights_only=False)
    ckpt_phase1 = torch.load(MODEL_PHASE1_OUT, map_location="cpu", weights_only=False)
    vocab    = ckpt["vocab"]

    net, im, fm = pm4py.read_pnml(PNML_PATH)
    labels = [t.label for t in net.transitions if t.label is not None]
    env    = AlignmentEnv(net, im, labels)
    model  = ActorCritic(len(vocab), env.n_places, len(env.LABEL_SPACE))
    model.load_state_dict(ckpt["state"])
    model.eval()

    df = pd.read_csv(DS_CSV)
    df["prefix_activities"] = df["prefix_activities"].apply(ast.literal_eval)
    df = df[df["aligned_prefix"].notna()]

    test_cases = df["case_id"].unique()[K_TRAIN:]
    df = df[df["case_id"].isin(test_cases)].reset_index(drop=True)
    print(f"testing on {len(test_cases)} cases, {len(df)} rows")

    for case_id in test_cases:
        evaluate_case(case_id, df[df["case_id"] == case_id],
                      model, vocab, env)

if __name__ == "__main__":
    main()
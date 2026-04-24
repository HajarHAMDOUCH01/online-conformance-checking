import ast
import torch
import pandas as pd
import pm4py

import sys 
sys.path.append("/content/Online-Conformance-Checking/")

from model.train.ppo_env import AlignmentEnv
from model.model.ppo_model import ActorCritic

BASE      = r"/content/drive/MyDrive"
DS_CSV    = BASE + r"/pdc2025/prefix_alignment_dataset_v_2.csv"
PPO_PT    = BASE + r"/pdc2025/ppo_model_epoch_1.pt"
PNML_PATH = BASE + r"/pdc2025/pdc2025_000000.pnml"

N_CASES      = 50
MAX_PREFIXES = 50


def validate_alignment(generated, prefix, env):

    env.reset(prefix)
    
    n_invalid  = 0
    log_moves  = 0
    model_moves = 0
    sync_moves = 0

    for move, label in zip(generated['moves_str'], generated['labels_str']):
        enabled_labels = {t.label for t in env._enabled_visible()}

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
    rows    = list(case_df.iterrows())[:MAX_PREFIXES]

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

        print(f"  Net-valid          : {'YES' if val['is_valid'] else 'NO'} ({val['n_invalid']} illegal fires)")
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

K_TRAIN = 400
def main():
    ckpt     = torch.load(PPO_PT, map_location="cpu", weights_only=False)
    vocab    = ckpt["vocab"]
    n_places = ckpt["n_places"]

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

    cases = df["case_id"].unique()[:N_CASES]
    for case_id in cases:
        evaluate_case(case_id, df[df["case_id"] == case_id],
                      model, vocab, env)

if __name__ == "__main__":
    main()
import ast
import sys
import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pm4py
import yaml

# this code is not currently used

# ── Load config ───────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
 
with open(_CFG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)
 
_p1  = _cfg["phase1"]
_hp  = _p1["hyperparameters"]
_sch = _p1["schedule"]
_paths = _cfg["paths"]

# paths
DS_CSV           = _paths["ds_csv"]
PNML_PATH        = _paths["pnml_path"]
MODEL_PHASE1_OUT = _p1["model_phase1_out"]
PROJECT_ROOT     = _paths["project_root"]

# hyper-parameters
LR       = _hp["lr"]
MAX_GRAD = _hp["max_grad"]
W_LABEL  = _hp["w_label"]
 
# schedule
PHASE1_EPOCHS = _sch["phase1_epochs"]
BATCH_SIZE    = _sch["batch_size"]
MAX_STEPS     = _sch["max_steps"]
K_TRAIN       = _sch["k_train"]
 
# ── Project path ──────────────────────────────────────────────────────────────
# sys.path.append(PROJECT_ROOT)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.ppo_env import AlignmentEnv
from model.ppo_model import ActorCritic
from model.model import Vocab
 
sys.modules['__main__'].Vocab    = Vocab
sys.modules['model'].Vocab       = Vocab
sys.modules['model.model'].Vocab = Vocab

def compute_phase1_loss(traj: dict, GT_labels: list, GT_moves: list,
                        env, vocab) -> torch.Tensor:
    """
    Supervised loss on label and move predictions.
    """
    T = min(len(traj['label_logits']), len(traj['move_logits']), len(GT_labels), len(GT_moves))
    if T == 0:
        return torch.tensor(0.0, requires_grad=True)

    label_logits_stack = torch.stack(traj['label_logits'][:T], dim=0)  # (T, |LABEL_SPACE|)
    move_logits_stack = torch.stack(traj['move_logits'][:T], dim=0)    # (T, |MOVE_SPACE|)

    label2idx = {l: i for i, l in enumerate(env.LABEL_SPACE)}
    label_targets = torch.tensor(
        [label2idx[l] for l in GT_labels[:T]], dtype=torch.long
    )
    move_targets = torch.tensor(
        [env.MOVE_ID_SPACE[m] for m in GT_moves[:T]], dtype=torch.long
    )
    label_loss = nn.CrossEntropyLoss()(label_logits_stack, label_targets)
    move_loss = nn.CrossEntropyLoss()(move_logits_stack, move_targets)
    return label_loss + move_loss


def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(DS_CSV)
    df["prefix_activities"] = df["prefix_activities"].apply(ast.literal_eval)
    df["step_types"]        = df["step_types"].apply(ast.literal_eval)        
    df["aligned_prefix"]    = df["aligned_prefix"].apply(ast.literal_eval)    
    df = df[df["aligned_prefix"].notna()]

    train_cases = df["case_id"].unique()[:K_TRAIN]
    df = df[df["case_id"].isin(train_cases)].reset_index(drop=True)
    print(f"Training phase 1 on {len(train_cases)} cases, {len(df)} rows")

    # ── Environment & vocabulary ───────────────────────────────────────────────
    net, im, fm = pm4py.read_pnml(PNML_PATH)
    labels      = [t.label for t in net.transitions if t.label is not None]
    env         = AlignmentEnv(net, im, labels)

    vocab = Vocab()
    for label in env.LABEL_SPACE:
        vocab.add(label)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ActorCritic(len(vocab), env.n_places, len(env.LABEL_SPACE))

    # if MODEL_PHASE1_OUT is not None and os.path.exists(MODEL_PHASE1_OUT):
    #     ckpt = torch.load(MODEL_PHASE1_OUT, map_location='cpu', weights_only=False)
    #     model.load_state_dict(ckpt['state'], strict=True)
    #     print("Loaded Phase 1 checkpoint.")

    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    cases = df["case_id"].unique()

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(PHASE1_EPOCHS):
            np.random.shuffle(cases)

            batch_trajs  = []
            batch_gt_lbl = []
            batch_gt_mv  = []
            total_loss   = 0.0
            n_updates    = 0
            skipped      = 0

            for idx, cid in enumerate(cases):
                case_df = df[df['case_id'] == cid].sort_values('prefix_length')

                for _, row in case_df.iterrows():
                    prefix = row['prefix_activities']
                    GT_move_types     = row['step_types']
                    if not prefix:
                        continue

                    src                = torch.tensor([vocab.encode(prefix)])
                    GT_activity_labels = row['aligned_prefix']

                    traj, n_invalid = model.generate(
                        src, prefix, env, vocab, max_len=MAX_STEPS,
                        teacher_labels=GT_activity_labels,
                        teacher_moves=GT_move_types,
                    )
                    # if idx % 25 == 0 and len(prefix) > 7:
                        # print()
                        # print("prefix    :", prefix)
                        # print("gen labels:", traj['labels_str'])
                        # print("gen moves :", traj['moves_str'])
                        # print("gt labels :", GT_activity_labels)
                        # print("gt move   :", GT_move_types)

                    if not traj or not traj['label_logits']:
                        skipped += 1
                        continue

                    batch_trajs.append(traj)
                    batch_gt_lbl.append(GT_activity_labels)
                    batch_gt_mv.append(GT_move_types)

                    if len(batch_trajs) >= BATCH_SIZE:
                        opt.zero_grad()
                        loss = torch.stack([
                            compute_phase1_loss(t, gl, gm, env, vocab)
                            for t, gl, gm in zip(batch_trajs, batch_gt_lbl, batch_gt_mv)
                        ]).mean()
                        # print("\nbatch loss:", loss)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD)
                        opt.step()

                        total_loss   += loss.item()
                        n_updates    += 1
                        batch_trajs   = []
                        batch_gt_lbl  = []
                        batch_gt_mv   = []

            if batch_trajs:
                opt.zero_grad()
                loss = torch.stack([
                    compute_phase1_loss(t, gl, gm, env, vocab)
                    for t, gl, gm in zip(batch_trajs, batch_gt_lbl, batch_gt_mv)
                ]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD)
                opt.step()
                total_loss += loss.item()
                n_updates  += 1

            avg_loss = total_loss / max(n_updates, 1)
            print(f"Epoch {ep+1}/{PHASE1_EPOCHS}  avg_loss={avg_loss:.4f}  "
                f"updates={n_updates}  skipped={skipped}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({'state': model.state_dict()}, MODEL_PHASE1_OUT)
    print(f"Phase 1 model saved -> {MODEL_PHASE1_OUT}")

if __name__ == "__main__":
    main()

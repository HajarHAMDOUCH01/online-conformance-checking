import ast
import sys
import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pm4py
import yaml
from pm4py.objects.petri_net.semantics import ClassicSemantics

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
MODEL_PHASE1_OUT = _p2["model_phase1_out"]  
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

# ─────────────────────────────────────────────────────────────────────────────
#  GAE
# ─────────────────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones):
    T   = len(rewards)
    adv = [0.0] * T
    gae = 0.0
    nv  = 0.0
    for t in reversed(range(T)):
        not_done = 0.0 if dones[t] else 1.0
        delta    = rewards[t] + GAMMA * nv * not_done - values[t]
        gae      = delta + GAMMA * LAM * not_done * gae
        adv[t]   = gae
        nv       = values[t]
    ret = [a + v for a, v in zip(adv, values)]
    return adv, ret


# ─────────────────────────────────────────────────────────────────────────────
#  PPO update
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(env, model, opt, batch):
    flat_advs, flat_rets, flat_old = [], [], []
    for traj in batch:
        adv, ret = compute_gae(traj['rewards'], traj['values'], traj['dones'])
        flat_advs.extend(adv)
        flat_rets.extend(ret)
        flat_old.extend(traj['old_lps'])

    adv_t = torch.tensor(flat_advs, dtype=torch.float32)
    ret_t = torch.tensor(flat_rets, dtype=torch.float32)
    old_t = torch.tensor(flat_old,  dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        opt.zero_grad()
        ptr        = 0
        epoch_loss = torch.tensor(0.0)     

        for traj in batch:

            T        = len(traj['rewards'])
            traj_adv = adv_t[ptr:ptr + T]
            traj_ret = ret_t[ptr:ptr + T]
            traj_old = old_t[ptr:ptr + T]
            ptr     += T

            enc_out, h = model.encode(traj['src_ids'])
            new_lp = []
            new_v  = []
            entropies = []

            for t in range(T):
                mv     = traj['marks'][t]
                act_id = traj['act_ids'][t]
                pos    = traj['positions'][t]
                if traj['dones'][t - 1] if t > 0 else False:
                    break  
                label_logits, val, h, _ = model.decode_step(pos, mv, h, enc_out, act_id)
                ll = label_logits.clone()
                valid_mask = traj['valid_label_masks'][t].to(ll.device)
                ll[0][~valid_mask] = float('-inf')
                label_dist = torch.distributions.Categorical(torch.softmax(ll[0], -1))

                new_lp.append(label_dist.log_prob(torch.tensor(traj['labels'][t], device=ll.device)))
                new_v.append(val)
                entropies.append(label_dist.entropy())

            new_lp = torch.stack(new_lp)
            new_v  = torch.stack(new_v).squeeze(-1)

            ratio = torch.exp(new_lp - traj_old.detach())
            ratio = torch.clamp(ratio, 0.0, 10.0)
            s1 = ratio * traj_adv.detach()
            s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * traj_adv.detach()

            actor_loss = -torch.min(s1, s2).mean()
            value_loss =  0.5 * (new_v - traj_ret.detach()).pow(2).mean()
            entropy = torch.stack(entropies).mean()
            traj_loss   = (actor_loss + VF_COEF * value_loss - ENT_COEF * entropy) / len(batch)
            epoch_loss  = epoch_loss + traj_loss   

        epoch_loss.backward()                     
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
        opt.step()

    return epoch_loss.item()

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Data ──────────────────────────────────────────────────────────────────

    df = pd.read_csv(DS_CSV)
    print(df.shape)
    print(df.head())

    df["prefix_activities"] = df["prefix_activities"].apply(ast.literal_eval)
    df = df[df["aligned_prefix"].notna()]

    df["aligned_prefix"] = df["aligned_prefix"].apply(ast.literal_eval)
    df["step_types"]     = df["step_types"].apply(ast.literal_eval)

    train_cases = df["case_id"].unique()[:K_TRAIN]
    df    = df[df["case_id"].isin(train_cases)].reset_index(drop=True)
    cases = df["case_id"].unique()
    print(f"Training phase 2 (PPO) on {len(train_cases)} cases, {len(df)} rows")

    # ── Environment ───────────────────────────────────────────────────────────
    net, im, fm = pm4py.read_pnml(PNML_PATH)
    print("Initial marking:", im)

    sink_place = next(p for p in net.places if p.name == "sink")
    fm = pm4py.generate_marking(net, sink_place)
    print("Final marking  :", fm)
    labels      = [t.label for t in net.transitions if t.label is not None]
    env         = AlignmentEnv(net, im, labels)
    # run this once after loading the net, before any training
    print("Initial marking:", im)
    print("\nEnabled at initial marking:")
    sem = ClassicSemantics()
    for t in sem.enabled_transitions(net, im):
        print(f"  {t.label} ({t.name})")

    print("\nAll places and their input transitions:")
    for p in sorted(net.places, key=lambda x: x.name):
        in_trans  = [a.source.label or a.source.name for a in net.arcs if a.target == p]
        out_trans = [a.target.label or a.target.name for a in net.arcs if a.source == p]
        print(f"  {p.name}: in={in_trans} out={out_trans}")
    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab = Vocab()
    for label in env.LABEL_SPACE:
        vocab.add(label)

    # ── Model — load Phase 1 weights ──────────────────────────────────────────
    model = ActorCritic(len(vocab), env.n_places, len(env.LABEL_SPACE))

    # if MODEL_PHASE1_OUT and os.path.exists(MODEL_PHASE1_OUT):
    #     ckpt = torch.load(MODEL_PHASE1_OUT, map_location="cpu", weights_only=False)
    #     model.load_state_dict(ckpt["state"], strict=True)
    #     print(f"Loaded Phase 1 weights from {MODEL_PHASE1_OUT}")
    # else:
    #     print("Warning: no Phase 1 checkpoint found — training from scratch.")

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ── PPO training loop ─────────────────────────────────────────────────────
    for ep in range(EPISODES):
        np.random.shuffle(cases)

        batch      = []
        total_loss = 0.0
        n_updates  = 0
        skipped    = 0

        def _flush_batch():
            print("started ppo update")
            nonlocal total_loss, n_updates
            l = ppo_update(env, model, opt, batch)
            total_loss += l
            n_updates  += 1
            batch.clear()

        for idx, cid in enumerate(cases):
            case_df = df[df["case_id"] == cid].sort_values("prefix_length")

            for _, row in case_df.iterrows():
                prefix = row["prefix_activities"]
                GT_activity_labels = row['aligned_prefix'] 
                GT_move_types     = row['step_types']

                if not prefix:
                    continue

                src = torch.tensor([vocab.encode(prefix)])

                model.eval()
                # if idx % 10 == 0:
                print("prefix        :",  prefix)
                print("gt labels     : ", GT_activity_labels)
                print("gt move_types : ", GT_move_types)
                with torch.no_grad():
                    traj, n_invalid = model.generate(src, prefix, env, vocab, max_len=MAX_STEPS)
                model.train()

                if traj and traj['rewards']:  
                    batch.append(traj)
                if not traj:
                    skipped += 1
                    continue
                # if idx % 10 == 0:
                print("generated labels :", traj["labels_str"])
                print("corresponding move types :", traj["moves_str"])
                print()

                if len(batch) >= BATCH_SIZE:
                    _flush_batch()

        if batch:
            _flush_batch()

        avg_loss = total_loss / max(n_updates, 1)
        print(f"Episode {ep+1}/{EPISODES}  avg_loss={avg_loss:.4f}  "
              f"updates={n_updates}  skipped={skipped}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        "state": model.state_dict(),
        "vocab": vocab,
        }, PPO_OUT)
    print(f"Phase 2 model saved → {PPO_OUT}")

if __name__ == "__main__":
    main()

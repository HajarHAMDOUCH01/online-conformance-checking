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


# ─────────────────────────────────────────────────────────────────────────────
#  Episode collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_episode(model, env, prefix, src_ids, vocab, generated_alignment):
    if not prefix or not generated_alignment:
        return None

    data = dict(
        marks=[], moves=[], labels=[], old_lps=[], moves_str=[], labels_str=[],
        rewards=[], values=[], dones=[],
        src_ids=src_ids, act_ids=[]
    )

    mv = env.reset(prefix)
    h  = model.encode(src_ids)

    for move_str, label_str in zip(generated_alignment['moves_str'],
                                   generated_alignment['labels_str']):
        if move_str not in env.MOVE_SPACE:
            break
        move_id = env.MOVE_SPACE.index(move_str)

        if label_str not in env.LABEL_SPACE:
            break
        label_id = env.LABEL_SPACE.index(label_str)

        act    = env.current_activity()
        act_id = vocab.t2i.get(act, vocab.t2i["<UNK>"]) if act else 0
        data['act_ids'].append(act_id)

        with torch.no_grad():
            move_logits, label_logits, val, h = model.decode_step(mv, h, act_id)

        move_mask  = env.valid_move_mask()
        label_mask = env.valid_label_mask(move_str)

        ml = move_logits.clone()
        ml[0, ~move_mask] = -1e9
        move_dist = torch.distributions.Categorical(torch.softmax(ml[0], -1))

        ll = label_logits.clone()
        ll[0, ~label_mask] = -1e9
        label_dist = torch.distributions.Categorical(torch.softmax(ll[0], -1))

        move_lp  = move_dist.log_prob(torch.tensor(move_id))
        label_lp = label_dist.log_prob(torch.tensor(label_id))
        old_lp   = (move_lp + label_lp).item()

        reward, done = env.step(move_id, label_id)
        new_mv = env.marking_vec()

        data['marks'].append(mv.clone())
        data['moves'].append(move_id)
        data['labels'].append(label_id)
        data['moves_str'].append(move_str)
        data['labels_str'].append(label_str)
        data['old_lps'].append(old_lp)
        data['rewards'].append(float(reward))
        data['values'].append(val.item())
        data['dones'].append(done)

        mv = new_mv
        if done:
            break

    return data if data['rewards'] else None


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

def ppo_update(model, opt, batch):
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

    loss = torch.tensor(0.0)   

    for _ in range(PPO_EPOCHS):
        opt.zero_grad()
        ptr = 0

        for traj in batch:
            T        = len(traj['rewards'])
            traj_adv = adv_t[ptr:ptr + T]
            traj_ret = ret_t[ptr:ptr + T]
            traj_old = old_t[ptr:ptr + T]
            ptr     += T

            h      = model.encode(traj['src_ids'])
            new_lp = []
            new_v  = []

            for t in range(T):
                mv     = traj['marks'][t]
                act_id = traj['act_ids'][t]
                move_logits, label_logits, val, h = model.decode_step(mv, h, act_id)

                move_dist  = torch.distributions.Categorical(
                    torch.softmax(move_logits[0],  -1))
                label_dist = torch.distributions.Categorical(
                    torch.softmax(label_logits[0], -1))

                new_lp.append(
                    move_dist.log_prob(torch.tensor(traj['moves'][t])) +
                    label_dist.log_prob(torch.tensor(traj['labels'][t]))
                )
                new_v.append(val)

            new_lp = torch.stack(new_lp)
            new_v  = torch.stack(new_v).squeeze(-1)

            ratio = torch.exp(new_lp - traj_old.detach())
            s1 = ratio * traj_adv.detach()
            s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * traj_adv.detach()

            actor_loss = -torch.min(s1, s2).mean()
            value_loss =  0.5 * (new_v - traj_ret.detach()).pow(2).mean()
            entropy    = -new_lp.mean()

            loss = (actor_loss + VF_COEF * value_loss + ENT_COEF * entropy) / len(batch)
            loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
        opt.step()

    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(DS_CSV)
    df["prefix_activities"] = df["prefix_activities"].apply(ast.literal_eval)
    df = df[df["aligned_prefix"].notna()]

    train_cases = df["case_id"].unique()[:K_TRAIN]
    df    = df[df["case_id"].isin(train_cases)].reset_index(drop=True)
    cases = df["case_id"].unique().to_numpy()   
    print(f"Training phase 2 (PPO) on {len(train_cases)} cases, {len(df)} rows")

    # ── Environment ───────────────────────────────────────────────────────────
    net, im, fm = pm4py.read_pnml(PNML_PATH)
    labels      = [t.label for t in net.transitions if t.label is not None]
    env         = AlignmentEnv(net, im, labels)

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab = Vocab()
    for label in env.LABEL_SPACE:
        vocab.add(label)

    # ── Model — load Phase 1 weights ──────────────────────────────────────────
    model = ActorCritic(len(vocab), env.n_places, len(env.LABEL_SPACE))

    if MODEL_PHASE1_OUT and os.path.exists(MODEL_PHASE1_OUT):
        ckpt = torch.load(MODEL_PHASE1_OUT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state"], strict=True)
        print(f"Loaded Phase 1 weights from {MODEL_PHASE1_OUT}")
    else:
        print("Warning: no Phase 1 checkpoint found — training from scratch.")

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ── PPO training loop ─────────────────────────────────────────────────────
    for ep in range(EPISODES):
        np.random.shuffle(cases)

        batch      = []
        total_loss = 0.0
        n_updates  = 0
        skipped    = 0

        def _flush_batch():
            nonlocal total_loss, n_updates
            l = ppo_update(model, opt, batch)
            total_loss += l
            n_updates  += 1
            batch.clear()

        for cid in cases:
            case_df = df[df["case_id"] == cid].sort_values("prefix_length")

            for _, row in case_df.iterrows():
                prefix = row["prefix_activities"]
                if not prefix:
                    continue

                src = torch.tensor([vocab.encode(prefix)])

                model.eval()
                with torch.no_grad():
                    generated, n_invalid = model.generate(
                        src, prefix, env, vocab, max_len=MAX_STEPS
                    )
                model.train()

                if not generated:
                    skipped += 1
                    continue

                traj = collect_episode(
                    model, env, prefix, src, vocab,
                    generated_alignment=generated
                )
                if traj:
                    batch.append(traj)

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
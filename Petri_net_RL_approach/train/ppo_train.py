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
from pm4py.objects.log.importer.xes import importer as xes_importer

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
XES_PATH         = _paths["xes_path"]

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

    for i_ppo in range(PPO_EPOCHS):
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
            new_lp    = []
            new_v     = []
            entropies = []

            for t in range(T):
                mv       = traj['marks'][t]
                act_id   = traj['act_ids'][t]
                pos      = traj['positions'][t]
                # NEW — replay the exact loop_depth the policy saw during
                # rollout so the gradient flows through the same computation
                # graph that produced the action.
                loop_depth = traj['loop_depths'][t]

                if traj['dones'][t - 1] if t > 0 else False:
                    break
                remaining_cost = int(traj['costs'][t])
                label_logits, val, h, _ = model.decode_step(
                    pos,
                    mv,
                    h,
                    enc_out,
                    act_id,
                    loop_depth=loop_depth,
                    remaining_cost=remaining_cost
                )

                ll = label_logits.clone()
                valid_mask  = traj['valid_label_masks'][t].to(ll.device)
                policy_bias = traj['policy_biases'][t].to(ll.device)
                ll[0] = ll[0] + policy_bias
                ll[0][~valid_mask] = float('-inf')

                probs = torch.softmax(ll[0], -1)
                # skip degenerate distributions that somehow slipped through
                if torch.isnan(probs).any() or probs.sum() < 1e-9:
                    continue

                label_dist = torch.distributions.Categorical(probs)

                new_lp.append(label_dist.log_prob(
                    torch.tensor(traj['labels'][t], device=ll.device)
                ))
                new_v.append(val)
                entropies.append(label_dist.entropy())

            if not new_lp:
                continue

            new_lp = torch.stack(new_lp)
            new_v  = torch.stack(new_v).squeeze(-1)

            # slice adv/ret to match actual replayed steps (may be shorter
            # than T if we hit a done mid-trajectory)
            n_replayed = len(new_lp)
            traj_adv_r = traj_adv[:n_replayed]
            traj_ret_r = traj_ret[:n_replayed]
            traj_old_r = traj_old[:n_replayed]

            ratio = torch.exp(new_lp - traj_old_r.detach())
            ratio = torch.clamp(ratio, 0.0, 10.0)
            s1 = ratio * traj_adv_r.detach()
            s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * traj_adv_r.detach()

            actor_loss = -torch.min(s1, s2).mean()
            value_loss =  0.5 * (new_v - traj_ret_r.detach()).pow(2).mean()
            entropy    = torch.stack(entropies).mean()
            traj_loss  = (actor_loss + VF_COEF * value_loss - ENT_COEF * entropy) / len(batch)
            epoch_loss = epoch_loss + traj_loss

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

    log_all = xes_importer.apply(XES_PATH)
    traces_fitnes_list = []
    for trace in log_all:
        a = float(trace.attributes.get("trace_fitness"))
        traces_fitnes_list.append(a)

    threshold = 0.95
    train_cases = [
        trace for i, trace in enumerate(log_all)
        if 0.85 < traces_fitnes_list[i] < threshold
    ]
    print(log_all)
    train_cases_ids = [trace.attributes["concept:name"] for trace in train_cases]
    df["case_id"] = df["case_id"].astype(str)
    df    = df[df["case_id"].isin(train_cases_ids)].reset_index(drop=True)
    cases = df["case_id"].unique()
    print(f"Training (PPO) on {len(cases)} cases (between 0.85 and 0.95), {len(df)} rows")

    # ── Environment ───────────────────────────────────────────────────────────
    net, im, fm = pm4py.read_pnml(PNML_PATH)
    print("Initial marking:", im)

    sink_place = next(p for p in net.places if p.name == "sink")
    fm = pm4py.generate_marking(net, sink_place)
    print("Final marking  :", fm)

    labels = [t.label for t in net.transitions if t.label is not None]
    env    = AlignmentEnv(net, im, labels)

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab = Vocab()
    for label in env.LABEL_SPACE:
        vocab.add(label)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ActorCritic(len(vocab), env.n_places, len(env.LABEL_SPACE))
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=EPISODES
    )
    # ── PPO training loop ─────────────────────────────────────────────────────
    for ep in range(EPISODES):
        np.random.shuffle(cases)

        batch      = []
        total_loss = 0.0
        n_updates  = 0
        skipped    = 0

        def _flush_batch():
            nonlocal total_loss, n_updates
            l = ppo_update(env, model, opt, batch)
            total_loss += l
            n_updates  += 1
            batch.clear()

        for idx, cid in enumerate(cases):
            case_df = df[df["case_id"] == cid].sort_values("prefix_length")

            for _, row in case_df.iterrows():
                prefix             = row["prefix_activities"]
                GT_activity_labels = row['aligned_prefix']
                GT_move_types      = row['step_types']

                if not prefix:
                    continue

                src = torch.tensor([vocab.encode(prefix)])

                model.eval()
                if idx % 100 == 0:
                    print("prefix        :", prefix)
                    print("gt labels     :", GT_activity_labels)
                    print("gt move_types :", GT_move_types)

                with torch.no_grad():
                    traj, n_invalid = model.generate(
                        src, prefix, env, vocab,
                        dataset_path=None
                    )

                model.train()

                if traj and traj['rewards']:
                    batch.append(traj)
                if not traj:
                    skipped += 1
                    continue
                if idx % 100 == 0:
                    print("generated labels :", traj["labels_str"])
                    print("corresponding move types :", traj["moves_str"])

                if len(batch) >= BATCH_SIZE:
                    _flush_batch()
                    scheduler.step()
            # print("finished case idx : ", idx)

        if batch:
            _flush_batch()
            scheduler.step()
        avg_loss = total_loss / max(n_updates, 1)
        print(f"Episode {ep+1}/{EPISODES}  avg_loss={avg_loss:.4f}  "
              f"updates={n_updates}  skipped={skipped}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        "state": model.state_dict(),
        "vocab": vocab,
    }, PPO_OUT)
    print(f"Phase 2 model saved -> {PPO_OUT}")


if __name__ == "__main__":
    main()
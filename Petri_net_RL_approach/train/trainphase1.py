import ast
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pm4py
import sys 
sys.path.append(r"C:\Users\LENONVO\OneDrive\Desktop\model\Petri_net_RL_approach")

from ppo_env import AlignmentEnv
from model.ppo_model import ActorCritic
from model.model import Vocab

sys.modules['__main__'].Vocab = Vocab
sys.modules['model'].Vocab = Vocab
sys.modules['model.model'].Vocab = Vocab

DS_CSV             = r"C:\Users\LENONVO\OneDrive\Desktop\STAGE-PFE-CRAN\datasets\prefix_alignment_dataset_pdc.csv"
PNML_PATH          = r"C:\Users\LENONVO\OneDrive\Desktop\STAGE-PFE-CRAN\datasets\pdc2025_000000.pnml"
MODEL_PHASE1_OUT   = r"C:\Users\LENONVO\OneDrive\Desktop\STAGE-PFE-CRAN\model_phase1.pt"
PPO_OUT            = r"C:\Users\LENONVO\OneDrive\Desktop\STAGE-PFE-CRAN\model_phase_ppo.pt"

GAMMA      = 0.99
LAM        = 0.95
CLIP       = 0.2
ENT_COEF   = 0.01
VF_COEF    = 0.5
LR         = 1e-4          
MAX_GRAD   = 0.5
PPO_EPOCHS = 2
EPISODES   = 20
BATCH_SIZE = 16
MAX_STEPS  = 150


def collect_episode(model, env, prefix, src_ids, vocab, generated_alignment):

    if not prefix or not generated_alignment:
        return None

    data = dict(
        marks=[], moves=[], labels=[], old_lps=[],
        rewards=[], values=[], dones=[],
        src_ids=src_ids, act_ids=[]
    )

    mv = env.reset(prefix)          
    h  = model.encode(src_ids)

    for move_str, label_str in zip(generated_alignment['moves_str'], generated_alignment['labels_str']):

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

        move_t  = torch.tensor(move_id)
        label_t = torch.tensor(label_id)

        move_lp  = move_dist.log_prob(move_t)
        label_lp = label_dist.log_prob(label_t)
        old_lp   = (move_lp + label_lp).item()

        reward, done = env.step(move_id, label_id)
        new_mv = env.marking_vec()

        data['marks'].append(mv.clone())
        data['moves'].append(move_id)
        data['labels'].append(label_id)
        data['old_lps'].append(old_lp)
        data['rewards'].append(float(reward))
        data['values'].append(val.item())
        data['dones'].append(done)

        mv = new_mv
        if done:
            break

    return data if data['rewards'] else None


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

    for _ in range(PPO_EPOCHS):
        opt.zero_grad()
        ptr = 0
        for traj in batch:
            T        = len(traj['rewards'])
            traj_adv = adv_t[ptr:ptr+T]
            traj_ret = ret_t[ptr:ptr+T]
            traj_old = old_t[ptr:ptr+T]
            ptr     += T

            h = model.encode(traj['src_ids'])
            new_lp, new_v = [], []

            for t in range(T):
                mv     = traj['marks'][t]
                act_id = traj['act_ids'][t]
                move_logits, label_logits, val, h = model.decode_step(mv, h, act_id)

                move_dist  = torch.distributions.Categorical(
                    torch.softmax(move_logits[0],  -1))
                label_dist = torch.distributions.Categorical(
                    torch.softmax(label_logits[0], -1))

                move_t  = torch.tensor(traj['moves'][t])
                label_t = torch.tensor(traj['labels'][t])

                new_lp.append(move_dist.log_prob(move_t) + label_dist.log_prob(label_t))
                new_v.append(val)

            new_lp = torch.stack(new_lp)
            new_v  = torch.stack(new_v)

            ratio = torch.exp(new_lp - traj_old.detach())
            s1 = ratio * traj_adv.detach()
            s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * traj_adv.detach()

            actor_loss = -torch.min(s1, s2).mean()
            value_loss =  0.5 * (new_v - traj_ret.detach()).pow(2).mean()
            entropy    = -(new_lp).mean()

            loss = (actor_loss + VF_COEF * value_loss + ENT_COEF * entropy) / len(batch)
            loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
        opt.step()

    return loss.item()

K_TRAIN = 100

def main():
    df = pd.read_csv(DS_CSV)
    df["prefix_activities"] = df["prefix_activities"].apply(ast.literal_eval)
    df = df[df["aligned_prefix"].notna()]

    train_cases = df["case_id"].unique()[:K_TRAIN]
    df = df[df["case_id"].isin(train_cases)].reset_index(drop=True)
    print(f"Training on {len(train_cases)} cases, {len(df)} rows")

    net, im, fm = pm4py.read_pnml(PNML_PATH)
    labels = [t.label for t in net.transitions if t.label is not None]
    env    = AlignmentEnv(net, im, labels)

    # ckpt  = torch.load(MODEL_PT, map_location='cpu', weights_only=False)
    # vocab = ckpt["vocab"]
    vocab = Vocab()
    for label in env.LABEL_SPACE:
        vocab.add(label)

    model = ActorCritic(len(vocab), env.n_places, len(env.LABEL_SPACE))

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    cases = df['case_id'].unique()

    for ep in range(EPISODES):
        np.random.shuffle(cases)
        batch      = []
        total_loss = 0.0
        n_updates  = 0
        skipped    = 0

        for cid in cases:
            case_df = df[df['case_id'] == cid].sort_values('prefix_length')

            for _, row in case_df.iterrows():
                prefix = row['prefix_activities']
                if not prefix:
                    continue

                src = torch.tensor([vocab.encode(prefix)])

                model.eval()
                #here generate is called with torch.no_grad()
                with torch.no_grad():
                    generated, n_invalid = model.generate(
                        src, prefix, env, vocab, max_len=MAX_STEPS
                    )
                model.train()

                if not generated:
                    skipped += 1
                    continue
                # here traj data is given by collect_episode function
                traj = collect_episode(
                    model, env, prefix, src, vocab,
                    generated_alignment=generated
                )

                if traj:
                    batch.append(traj)

                if len(batch) >= BATCH_SIZE:
                    l = ppo_update(model, opt, batch)
                    total_loss += l
                    n_updates  += 1
                    batch       = []

        if batch:
            l = ppo_update(model, opt, batch)
            total_loss += l
            n_updates  += 1

        avg_loss = total_loss / max(n_updates, 1)
        print(f"Episode {ep+1}/{EPISODES}  avg_loss={avg_loss:.4f}  "
              f"updates={n_updates}  skipped={skipped}")

    torch.save({
        "state"   : model.state_dict(),
        "vocab"   : vocab,
        "n_places": env.n_places,
        "n_labels": len(env.LABEL_SPACE),
    }, MODEL_PHASE1_OUT)
    print(f"Saved → {MODEL_PHASE1_OUT}")

if __name__ == '__main__':
    main()
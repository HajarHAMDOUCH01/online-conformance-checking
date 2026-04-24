import ast
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pm4py

import sys 
sys.path.append("/content/Online-Conformance-Checking")

from model.train.ppo_env import AlignmentEnv
from model.model.ppo_model import ActorCritic
from model.model.model import Vocab


sys.modules['__main__'].Vocab = Vocab
sys.modules['model'].Vocab = Vocab
sys.modules['model.model'].Vocab = Vocab

BASE      = r"/content/drive/MyDrive/pdc2025"
DS_CSV    = BASE + r"/prefix_alignment_dataset_v_2.csv"
MODEL_PT  = BASE + r"/phase1_model.pt"
PNML_PATH = BASE + r"/pdc2025_000000.pnml"
PPO_OUT   = BASE + r"/ppo_model_epoch_1.pt"
# PPO_CHEKPOINT   = BASE + r"/ppo_model_epoch_5.pt"

GAMMA      = 0.99
LAM        = 0.95
CLIP       = 0.2
ENT_COEF   = 0.01
VF_COEF    = 0.5
LR         = 3e-4
MAX_GRAD   = 0.5
PPO_EPOCHS = 1
EPISODES   = 1
BATCH_SIZE = 32
MAX_STEPS  = 150


def compute_gae(rewards, values, dones):
    T = len(rewards)
    adv = [0.0] * T
    gae = 0.0
    nv  = 0.0
    for t in reversed(range(T)):
        not_done = 0.0 if dones[t] else 1.0
        delta = rewards[t] + GAMMA * nv * not_done - values[t]
        gae   = delta + GAMMA * LAM * not_done * gae
        adv[t] = gae
        nv = values[t]
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
    batch_loss = 0.0

    for _ in range(PPO_EPOCHS):
        opt.zero_grad()
        ptr = 0
        # print("in one batch : \n")
        for traj in batch:
            T = len(traj['rewards'])
            traj_adv = adv_t[ptr:ptr+T]
            traj_ret = ret_t[ptr:ptr+T]
            traj_old = old_t[ptr:ptr+T]
            ptr += T

            h = model.encode(traj['src_ids'])
            new_lp, new_v = [], []

            for t in range(T):
                mv     = traj['marks'][t]
                act_id = traj['act_ids'][t]  
                move_logits, label_logits, val, h = model.decode_step(mv, h, act_id)

                move_dist  = torch.distributions.Categorical(torch.softmax(move_logits[0],  -1))
                label_dist = torch.distributions.Categorical(torch.softmax(label_logits[0], -1))

                move  = torch.tensor(traj['moves'][t])
                label = torch.tensor(traj['labels'][t])

                new_lp.append(move_dist.log_prob(move) + label_dist.log_prob(label))
                new_v.append(val)

            new_lp = torch.stack(new_lp)
            new_v  = torch.stack(new_v)

            ratio = torch.exp(new_lp - traj_old.detach())
            s1 = ratio * traj_adv.detach()
            s2 = torch.clamp(ratio, 1-CLIP, 1+CLIP) * traj_adv.detach() # adjvantage should not be a function of params of the policy

            actor_loss = -torch.min(s1, s2).mean()
            value_loss = 0.5 * (new_v - traj_ret.detach()).pow(2).mean()
            entropy    = -(new_lp).mean()

            loss = (actor_loss + VF_COEF*value_loss + ENT_COEF*entropy) / len(batch)
            batch_loss += loss
            loss.backward()
        print(f"loss of this batch : {batch_loss}")
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
        opt.step()

    # return loss.item()


K_TRAIN = 400  

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

    ckpt  = torch.load(MODEL_PT, map_location='cpu', weights_only=False)
    from model.model.model import Vocab
    vocab = ckpt["vocab"]
    model = ActorCritic(len(vocab), env.n_places, len(env.LABEL_SPACE))
    model.load_from_supervised(MODEL_PT)
    for label in env.LABEL_SPACE:
      vocab.add(label)
    # continue_checkpoint_ppo_model = torch.load(PPO_CHEKPOINT, map_location='cpu', weights_only=False)
    # model.load_state_dict(continue_checkpoint_ppo_model["state"])
    # print("\nloaded ppo weights from checkpoint")

    



    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    cases = df['case_id'].unique()

    for ep in range(EPISODES):
        # print("testing with 1 epoch only")
        np.random.shuffle(cases)
        batch = []

        for idx, cid in enumerate(cases):
            case_df = df[df['case_id'] == cid].sort_values('prefix_length')

            for _, row in case_df.iterrows():
                prefix = row['prefix_activities']
                src    = torch.tensor([vocab.encode(prefix)])
                traj, _ = model.generate(src=src, prefix=prefix, env=env, vocab=vocab)
                # if idx % 10 == 0:
                #   print(f"\ncase : , {idx} and case_id = {cid}")
                #   print("\nprefix : ", prefix)
                #   print("\ngeerated alignement : ", traj['labels_str'])
                #   print("\ngeerated alignement : ", traj['moves_str'])
                if traj:
                    batch.append(traj)

                if len(batch) >= BATCH_SIZE:
                    np.random.shuffle(batch)
                    ppo_update(model, opt, batch)
                    batch = []

            if batch:
              ppo_update(model, opt, batch)
            if idx % 10 == 0 :
                print("\nfinished case : ", idx)
        print(f"Episode {ep+1}/{EPISODES} done")

    torch.save({"state": model.state_dict(), "vocab": vocab,
                "n_places": env.n_places, "n_labels": len(env.LABEL_SPACE)}, PPO_OUT)
    print(f"Saved → {PPO_OUT}")


if __name__ == '__main__':
    main()
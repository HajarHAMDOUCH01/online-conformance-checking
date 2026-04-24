import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, vocab_size: int, n_places: int, n_labels: int,
                 emb_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.enc = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.marking_proj = nn.Linear(n_places, emb_dim, bias=False)
        self.dec = nn.GRU(emb_dim, hidden_dim, batch_first=True)

        self.move_head  = nn.Linear(hidden_dim, 3)
        self.label_head = nn.Linear(hidden_dim, n_labels)
        self.critic     = nn.Linear(hidden_dim, 1)

        nn.init.orthogonal_(self.marking_proj.weight)
        nn.init.zeros_(self.move_head.bias)
        nn.init.zeros_(self.label_head.bias)
        nn.init.zeros_(self.critic.bias)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        _, h = self.enc(self.emb(src))
        return h

    def decode_step(self, mv: torch.Tensor, h: torch.Tensor, act_id: int = 0):

        marking_emb  = self.marking_proj(mv).view(1, 1, -1)
        activity_emb = self.emb(torch.tensor([[act_id]])).float()   # (1,1,emb_dim)
        inp = marking_emb + activity_emb                             # fuse

        out, new_h = self.dec(inp, h)
        hidden = out[:, 0, :]

        move_logits  = self.move_head(hidden)
        label_logits = self.label_head(hidden)
        value        = self.critic(hidden).squeeze(-1)

        return move_logits, label_logits, value, new_h

    def load_from_supervised(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        src  = ckpt['state']
        dst  = self.state_dict()

        key_map = {
            'emb.weight': 'emb.weight',
            'enc.weight_ih_l0': 'enc.weight_ih_l0',
            'enc.weight_hh_l0': 'enc.weight_hh_l0',
            'enc.bias_ih_l0': 'enc.bias_ih_l0',
            'enc.bias_hh_l0': 'enc.bias_hh_l0',
            'dec.weight_ih_l0': 'dec.weight_ih_l0',
            'dec.weight_hh_l0': 'dec.weight_hh_l0',
            'dec.bias_ih_l0': 'dec.bias_ih_l0',
            'dec.bias_hh_l0': 'dec.bias_hh_l0',
        }

        loaded, skipped = [], []
        for k, v in src.items():
            mapped = key_map.get(k, k)
            if mapped in dst and dst[mapped].shape == v.shape:
                dst[mapped] = v.clone()
                loaded.append(k)
            else:
                skipped.append(k)

        self.load_state_dict(dst)
        print(f"Loaded : {loaded}")
        print(f"Skipped: {skipped}")
        return ckpt['vocab']

    # @torch.no_grad()
    def generate(self, src: torch.Tensor, prefix: list, env, vocab, max_len: int = 150, train=True):
        self.train()
        data = dict(marks=[], moves=[], labels=[], moves_str=[], labels_str=[], old_lps=[],
                rewards=[], values=[], dones=[], src_ids=src, act_ids=[])
        h  = self.encode(src)
        mv = env.reset(prefix)
        
        n_invalid = 0

        for _ in range(1, max_len):
            move_mask = env.valid_move_mask()

            act = env.current_activity()
            act_id = vocab.t2i.get(act, vocab.t2i["<UNK>"]) if act else 0
            data['act_ids'].append(act_id)
            if train:
                move_logits, label_logits, value, h = self.decode_step(mv, h, act_id)
            else:
                with torch.no_grad():
                    move_logits, label_logits, value, h = self.decode_step(mv, h, act_id)
            ml = move_logits.clone()
            ml[0, ~move_mask] = -1e9
            move_dist = torch.distributions.Categorical(torch.softmax(ml[0], -1))
            move      = move_dist.sample()
            move_lp   = move_dist.log_prob(move)

            label_mask = env.valid_label_mask(env.MOVE_SPACE[move.item()])
            ll = label_logits.clone()
            ll[0, ~label_mask] = -1e9
            label_dist = torch.distributions.Categorical(torch.softmax(ll[0], -1))
            label      = label_dist.sample()
            label_lp   = label_dist.log_prob(label)
            move_str  = env.MOVE_SPACE[move]
            label_str = env.LABEL_SPACE[label]
            
            reward, done = env.step(move.item(), label.item())
            
            if move_str in ("S", "M") and reward == -2.0:
                n_invalid += 1

            new_mv = env.marking_vec()

            data['marks'].append(mv.clone())
            data['moves'].append(move.item())
            data['labels'].append(label.item())
            data['moves_str'].append(move_str)
            data['labels_str'].append(label_str)
            data['old_lps'].append((move_lp + label_lp).item())
            data['rewards'].append(float(reward))
            data['values'].append(value.item())
            data['dones'].append(done)

            mv = new_mv

            if done:
                break

        return data, n_invalid
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, vocab_size: int, n_places: int, n_labels: int,
                 emb_dim: int = 64, hidden_dim: int = 128,
                 prefix_attn_window: float = 2.0,
                 m_streak_penalty: float = 1.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.prefix_attn_window = prefix_attn_window
        # self.current_label_bias = current_label_bias
        self.m_streak_penalty = m_streak_penalty
        self.attn_scale = hidden_dim ** -0.5
        self.emb          = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.enc          = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.marking_proj = nn.Linear(n_places, emb_dim, bias=False)
        self.fuse_proj    = nn.Linear(emb_dim * 3, emb_dim, bias=False)
        self.dec          = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.pos_emb = nn.Embedding(70, self.emb_dim)
        self.attn_q       = nn.Linear(hidden_dim, hidden_dim, bias=False)  # query from decoder hidden
        self.attn_k       = nn.Linear(hidden_dim, hidden_dim, bias=False)  # key from encoder outputs
        self.attn_v       = nn.Linear(hidden_dim, hidden_dim, bias=False)  # value from encoder outputs
        self.attn_out     = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)  # fuse context + hidden
        self.state_norm   = nn.LayerNorm(hidden_dim)

        self.label_head   = nn.Linear(hidden_dim, n_labels)
        self.critic       = nn.Linear(hidden_dim, 1)

        nn.init.orthogonal_(self.marking_proj.weight)
        nn.init.zeros_(self.label_head.bias)
        nn.init.zeros_(self.critic.bias)
        nn.init.orthogonal_(self.fuse_proj.weight)

    def encode(self, src: torch.Tensor):
        enc_out, h = self.enc(self.emb(src))   # enc_out: (1, seq_len, hidden_dim)
        return enc_out, h

    def decode_step(self, pos, mv: torch.Tensor, h: torch.Tensor, enc_out: torch.Tensor, act_id: int = 0):
        # print("mv.shape =", mv.shape)
        # print("expected =", self.marking_proj.in_features)
        marking_emb  = self.marking_proj(mv).view(1, 1, -1)
        activity_emb = self.emb(torch.tensor([[act_id]], device=mv.device)).float()
        
        pos_id = min(pos, self.pos_emb.num_embeddings - 1)
        pos_emb = self.pos_emb(torch.tensor([[pos_id]], device=mv.device))

        inp = self.fuse_proj(
            torch.cat([
                marking_emb,
                activity_emb,
                pos_emb
            ], dim=-1)
        )

        # inp = self.fuse_proj(torch.cat([marking_emb, activity_emb], dim=-1)) # shape (1, 1, 64)

        out, new_h = self.dec(inp, h)

        readout, attn_weights = self.atten(out, enc_out, focus_pos=pos)
        new_h = self.state_norm(new_h + readout.transpose(0, 1))   # (1, 1, H)

        label_logits = self.label_head(readout.squeeze(1))
        value        = self.critic(readout.squeeze(1)).squeeze(-1)

        return label_logits, value, new_h, attn_weights
    
    def atten(self, decoder_hidden: torch.Tensor, enc_out: torch.Tensor,
              focus_pos: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # decoder_hidden: (1, 1, hidden_dim) : representing generated activities so far (markings trajectory and previous activities)
        # enc_out:        (1, seq_len, hidden_dim) : representing the prefix in hidden space 
        q = self.attn_q(decoder_hidden)                        # (1, 1, hidden_dim)
        k = self.attn_k(enc_out)                               # (1, seq_len, hidden_dim)
        v = self.attn_v(enc_out)                               # (1, seq_len, hidden_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.attn_scale  # (1, 1, seq_len)
        if focus_pos is not None:
            positions = torch.arange(enc_out.size(1), device=enc_out.device, dtype=scores.dtype)
            distance = positions - min(focus_pos, enc_out.size(1) - 1)
            local_bias = -0.5 * (distance / self.prefix_attn_window).pow(2)
            scores = scores + local_bias.view(1, 1, -1)
        weights = torch.softmax(scores, dim=-1)                # (1, 1, seq_len) : weights per original prefix activity for the generation in this step
        context = torch.bmm(weights, v)                        # (1, 1, hidden_dim)
        fused = torch.tanh(self.attn_out(
            torch.cat([decoder_hidden, context], dim=-1)
        ))                                                   # (1, 1, hidden_dim)
        return fused, weights.squeeze(0).squeeze(0)                        # weights for reward_function

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
        print(f"Note: fuse_proj initialised from scratch (not in Phase 1 checkpoint)")

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

    def prefix_policy_bias(self, env, valid_label_mask: torch.Tensor,
                           prev_moves: list[str], device) -> torch.Tensor:
        bias = torch.zeros(len(env.LABEL_SPACE), device=device)
        act = env.current_activity()

        m_streak = 0
        for move in reversed(prev_moves):
            if move == "M":
                m_streak += 1
            else:
                break

        if m_streak:
            for label_id, allowed in enumerate(valid_label_mask.tolist()):
                if allowed and env.infere_move_type(label_id) == env.MOVE_ID_SPACE["M"]:
                    bias[label_id] -= self.m_streak_penalty * m_streak

        return bias
    
    def generate(self, src, prefix, env, vocab, max_len=60, train=True):
        self.train(train)
        data = dict(
            marks=[], moves=[], labels=[],
            moves_str=[], labels_str=[],
            label_logits=[],rewards=[],
            old_lps=[], values=[], dones=[], src_ids=src, act_ids=[],
            positions=[], valid_label_masks=[], policy_biases=[]
        )
        enc_out, h = self.encode(src)
        mv = env.reset(prefix)
        n_invalid = 0

        for i in range(0, max_len):
            done = False
            act    = env.current_activity()
            act_id = vocab.t2i.get(act, vocab.t2i["<UNK>"]) if act else 0
            data['act_ids'].append(act_id)
            pos = env.pos
            data['positions'].append(pos)
            label_logits, value, h, attn_weights = self.decode_step(pos, mv, h, enc_out, act_id)
            position = 1
            moves_for_all_labels = [env.infere_move_type(i) for i in range(len(env.LABEL_SPACE))]
            data['label_logits'].append(label_logits[0]) # label_logits hsape (1, n_labels)

            valid_labels_mask = env.valid_label_mask()
            data['valid_label_masks'].append(valid_labels_mask.clone())
            policy_bias = self.prefix_policy_bias(
                env, valid_labels_mask, data['moves_str'], label_logits.device
            )
            data['policy_biases'].append(policy_bias.detach().cpu())

            ll = label_logits.clone()
            ll[0] = ll[0] + policy_bias
            ll[0][~valid_labels_mask] = float('-inf') 
            if torch.isnan(torch.softmax(ll[0], -1)).any():
                print(f"  [WARN] NaN in softmax at step {i}, using uniform")
                ll = torch.zeros_like(label_logits)

            label_dist = torch.distributions.Categorical(torch.softmax(ll[0], -1))
            # label      = label_dist.sample() if train else label_dist.probs.argmax()
            label        = label_dist.probs.argmax()

            old_lp = label_dist.log_prob(label).item() 
            position = i        
            move = env.infere_move_type(label.item())
            ### debug
            if move == 3:
                raise RuntimeError("move can only be L/M/S; Mask didn't work !")
            
            move_str  = env.MOVE_SPACE[move]
            
            label_str = env.LABEL_SPACE[label]
            label_str, move_str, reward, done = env.step(self, move, label.item(), list(data['moves']), list(data['labels']), label_logits[0], attn_weights, moves_for_all_labels, done)
            
            label = env.LABEL_ID_SPACE[label_str]
            move = env.MOVE_ID_SPACE[move_str]

            data['rewards'].append(float(reward))
            if reward == -20.0:
                done = True

            new_mv = env.marking_vec()
            data['marks'].append(mv.clone())
            data['moves'].append(move)
            data['labels'].append(label)
            data['moves_str'].append(move_str)
            data['labels_str'].append(label_str)
            data['old_lps'].append(old_lp)                     
            data['values'].append(value.item())
            data['dones'].append(done)
            mv = new_mv

            if done:
                # Pénaliser les trajectoires trop longues
                n_moves    = len(data['moves'])
                n_prefix   = len(prefix)
                n_m_moves  = data['moves_str'].count('M')
                
                # ratio de M-moves : idéalement proche de 0
                m_ratio = n_m_moves / max(n_moves, 1)
                
                # pénalité finale distribuée sur tous les steps
                length_penalty = -0.5 * m_ratio
                data['rewards'] = [r + length_penalty for r in data['rewards']]
                break

        return data, n_invalid

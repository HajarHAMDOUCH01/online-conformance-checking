import torch
import torch.nn as nn
import pm4py 
from pm4py.objects.petri_net.obj import Marking
from .dataset_utils import save_episode_transitions
from train.ppo_env import _m_tuple   
from baselines.A_start_baseline.dataset_generation_a_star import _astar_prefix_alignment

def normalize_marking_tuple(marking):

    if isinstance(marking, tuple):
        return marking

    if isinstance(marking, dict):
        return _m_tuple(marking)

    if isinstance(marking, Marking):
        return _m_tuple({
            p.name:v
            for p,v in marking.items()
            if v > 0
        })

    raise TypeError(type(marking))

class ActorCritic(nn.Module):
    def __init__(self, vocab_size: int, n_places: int, n_labels: int,
                 emb_dim: int = 64, hidden_dim: int = 128,
                 prefix_attn_window: float = 2.0,

                 # this makes the agent choose the first activity in the prefix, 
                 # without it it starts away from the prefix and gets lost and this prevents training , 
                 # training uses sampling anywway , so sometimes it still starts with an M move away from prefix 
                 # and learns from its actions
                 current_label_bias: float = 5.0,
                 
                 m_streak_penalty: float = 2.0,
                 max_loop_depth: int = 4): 
        super().__init__()
        self.emb_dim = emb_dim
        self.prefix_attn_window = prefix_attn_window
        self.current_label_bias = current_label_bias
        self.m_streak_penalty = m_streak_penalty
        self.max_loop_depth = max_loop_depth
        self.attn_scale = hidden_dim ** -0.5
        self.cost_emb = nn.Embedding(100, emb_dim)

        self.emb          = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.enc          = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.marking_proj = nn.Linear(n_places, emb_dim, bias=False)

        # fuse_proj now takes 4 × emb_dim:
        #   marking_emb | activity_emb | pos_emb | loop_depth_emb
        self.fuse_proj = nn.Linear(
            emb_dim * 5,
            emb_dim,
            bias=False
        )

        self.dec          = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.pos_emb      = nn.Embedding(70, emb_dim)

        # loop-depth embedding: 0 = never visited, 1..max_loop_depth = revisit count
        # index 0 -> first visit, index max_loop_depth -> "deep in a loop"
        self.loop_depth_emb = nn.Embedding(max_loop_depth + 1, emb_dim)

        self.attn_q   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_k   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_v   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_out = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.state_norm = nn.LayerNorm(hidden_dim)

        self.label_head = nn.Linear(hidden_dim, n_labels)
        self.critic     = nn.Linear(hidden_dim, 1)

        nn.init.orthogonal_(self.marking_proj.weight)
        nn.init.zeros_(self.label_head.bias)
        nn.init.zeros_(self.critic.bias)
        nn.init.orthogonal_(self.fuse_proj.weight)
        nn.init.normal_(
            self.cost_emb.weight,
            mean=0.0,
            std=0.01
        )
        # loop_depth_emb: initialise with small normal so depth=0 starts near zero
        nn.init.normal_(self.loop_depth_emb.weight, mean=0.0, std=0.01)

    # -------------------------------------------------------------------------
    def encode(self, src: torch.Tensor):
        enc_out, h = self.enc(self.emb(src))
        return enc_out, h

    # -------------------------------------------------------------------------
    def decode_step(
        self,
        pos,
        mv,
        h,
        enc_out,
        act_id=0,
        loop_depth=0,
        remaining_cost=0
    ):       
        """
        One decoder step.

        loop_depth : how many times the current (pos, marking) state has been
                     visited in this episode before this step (0 = first visit).
                     Clamped to self.max_loop_depth so the embedding table is never
                     out-of-bounds.  The critic learns to predict lower returns as
                     loop_depth grows; the actor learns to diversify its action
                     distribution when loop_depth is high.
        """
        device = mv.device

        marking_emb  = self.marking_proj(mv).view(1, 1, -1)          # (1,1,E)
        activity_emb = self.emb(torch.tensor([[act_id]], device=device)).float()  # (1,1,E)

        pos_id = min(pos, self.pos_emb.num_embeddings - 1)
        pos_emb = self.pos_emb(torch.tensor([[pos_id]], device=device))           # (1,1,E)

        depth_id = min(loop_depth, self.max_loop_depth)
        depth_emb = self.loop_depth_emb(
            torch.tensor([[depth_id]], device=device)
        )                                                              # (1,1,E)
        cost_id = min(
            remaining_cost,
            self.cost_emb.num_embeddings - 1
        )

        cost_emb = self.cost_emb(
            torch.tensor([[cost_id]], device=device)
        )
        inp = self.fuse_proj(
            torch.cat([
                marking_emb,
                activity_emb,
                pos_emb,
                depth_emb,
                cost_emb
            ], dim=-1)        
        )                                                              # (1,1,E)

        out, new_h = self.dec(inp, h)

        readout, attn_weights = self.atten(out, enc_out, focus_pos=pos)
        new_h = self.state_norm(new_h + readout.transpose(0, 1))

        label_logits = self.label_head(readout.squeeze(1))
        value        = self.critic(readout.squeeze(1)).squeeze(-1)

        return label_logits, value, new_h, attn_weights

    # -------------------------------------------------------------------------
    def atten(self, decoder_hidden: torch.Tensor, enc_out: torch.Tensor,
              focus_pos: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.attn_q(decoder_hidden)
        k = self.attn_k(enc_out)
        v = self.attn_v(enc_out)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.attn_scale
        if focus_pos is not None:
            positions = torch.arange(enc_out.size(1), device=enc_out.device,
                                     dtype=scores.dtype)
            distance  = positions - min(focus_pos, enc_out.size(1) - 1)
            local_bias = -0.5 * (distance / self.prefix_attn_window).pow(2)
            scores = scores + local_bias.view(1, 1, -1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, v)
        fused = torch.tanh(self.attn_out(
            torch.cat([decoder_hidden, context], dim=-1)
        ))
        return fused, weights.squeeze(0).squeeze(0)

    # -------------------------------------------------------------------------
    def load_from_supervised(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        src  = ckpt['state']
        dst  = self.state_dict()

        key_map = {
            'emb.weight':        'emb.weight',
            'enc.weight_ih_l0':  'enc.weight_ih_l0',
            'enc.weight_hh_l0':  'enc.weight_hh_l0',
            'enc.bias_ih_l0':    'enc.bias_ih_l0',
            'enc.bias_hh_l0':    'enc.bias_hh_l0',
            'dec.weight_ih_l0':  'dec.weight_ih_l0',
            'dec.weight_hh_l0':  'dec.weight_hh_l0',
            'dec.bias_ih_l0':    'dec.bias_ih_l0',
            'dec.bias_hh_l0':    'dec.bias_hh_l0',
        }
        print(
        "Note: fuse_proj, loop_depth_emb and cost_emb initialized from scratch"
        )
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

    # -------------------------------------------------------------------------
    def prefix_policy_bias(self, env, valid_label_mask, prev_moves, device,
                           current_label_bias=None):
        bias = torch.zeros(len(env.LABEL_SPACE), device=device)

        act = env.current_activity()
        if act in env.LABEL_ID_SPACE:
            cb = current_label_bias if current_label_bias is not None else self.current_label_bias
            if cb is not None:
                bias[env.LABEL_ID_SPACE[act]] += float(cb)

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
    


    # -------------------------------------------------------------------------
    def generate(
        self,
        src,
        prefix,
        env,
        vocab,
        max_len=60,
        train=False,
        compute_reward=True,
        dataset_path: str | None = None):

        # self.train(train)
        self.eval()

        data = dict(
            marks=[], moves=[], labels=[],
            moves_str=[], labels_str=[],
            label_logits=[], rewards=[],
            old_lps=[], values=[], dones=[], src_ids=src, act_ids=[],
            positions=[], valid_label_masks=[], policy_biases=[],
            loop_depths=[], costs=[] 
        )
        # Offline RL dataset storage
        offline_dataset = []
        enc_out, h = self.encode(src)
        mv = env.reset(prefix)
        n_invalid = 0
        done = False
        i = 0

        # ------------------------------------------------------------------
        # State-visit counter: (pos, marking_tuple) -> visit count.
        # This is the ground truth used by the reward function AND by the
        # loop_depth feature fed into decode_step.  They must be in sync.
        # ------------------------------------------------------------------
        _gen_visited: dict = {}

        while not done:
            act    = env.current_activity()
            act_id = vocab.t2i.get(act, vocab.t2i["<UNK>"]) if act else 0
            data['act_ids'].append(act_id)
            pos = env.pos

            current_marking = normalize_marking_tuple(
                env._marking_to_dict()
            )

            _, remaining_cost, _ = _astar_prefix_alignment(
                prefix=prefix,
                start_marking=current_marking,
                start_pos=pos
            )
            cost_id = min(int(remaining_cost), 99)

            # --- compute loop_depth BEFORE incrementing the counter --------
            # loop_depth = how many times we've already been in this state
            # before the current step, i.e. the count *prior* to this visit.
            state_key  = (pos, tuple(sorted(_m_tuple(env._marking_to_dict()))))
            loop_depth = _gen_visited.get(state_key, 0)
            # Save current state for offline RL
            offline_state = {
                "marking": current_marking,
                "position": pos,
                "activity_id": act_id,
                "loop_depth": loop_depth,
                "remaining_cost": cost_id
            }
            # --- hard structural exit: third revisit means we are stuck -----
            # (only as safety; the policy should learn to avoid this)
            # if loop_depth >= 3:
            #     break

            # increment visit counter for this step
            _gen_visited[state_key] = loop_depth + 1

            data['positions'].append(pos)
            data['loop_depths'].append(loop_depth)   # store for replay

            label_logits, value, h, attn_weights = self.decode_step(
                pos,
                mv,
                h,
                enc_out,
                act_id,
                loop_depth=loop_depth,
                remaining_cost=cost_id
            )
            moves_for_all_labels = [env.infere_move_type(i)
                                     for i in range(len(env.LABEL_SPACE))]
            data['label_logits'].append(label_logits[0])

            valid_labels_mask = env.valid_label_mask()
            data['valid_label_masks'].append(valid_labels_mask.clone())

            policy_bias = torch.zeros(len(env.LABEL_SPACE), device=label_logits.device)
            if i == 0:
                policy_bias = self.prefix_policy_bias(
                    env, valid_labels_mask, data['moves_str'],
                    label_logits.device, 50.0
                )
            data['policy_biases'].append(policy_bias.detach().cpu())

            ll = label_logits.clone()
            ll[0] = ll[0] + policy_bias
            ll[0][~valid_labels_mask] = float('-inf')

            # --- NaN guard: fires when valid_label_mask is all-False --------
            probs = torch.softmax(ll[0], -1)
            if torch.isnan(probs).any() or probs.sum() < 1e-9:
                break

            label_dist = torch.distributions.Categorical(probs)
            label      = label_dist.sample() 
            old_lp     = label_dist.log_prob(label).item()
            move       = env.infere_move_type(label.item())
            offline_action = {
                "move": move,
                "label": label.item()
            }
            if move == 3:
                raise RuntimeError("move can only be L/M/S; Mask didn't work!")

            move_str  = env.MOVE_SPACE[move]
            label_str = env.LABEL_SPACE[label]

            label_str, move_str, reward, done = env.step(
                self, valid_labels_mask, move, label.item(),
                list(data['moves']), list(data['labels']),
                label_logits[0], attn_weights, moves_for_all_labels,
                compute_reward=compute_reward
            )
            # Next state after transition
            next_marking = normalize_marking_tuple(
                env._marking_to_dict()
            )

            next_pos = env.pos

            next_act = env.current_activity()

            next_act_id = (
                vocab.t2i.get(next_act, vocab.t2i["<UNK>"])
                if next_act else 0
            )


            _, next_remaining_cost, _ = _astar_prefix_alignment(
                prefix=prefix,
                start_marking=next_marking,
                start_pos=next_pos
            )

            next_cost_id = min(
                int(next_remaining_cost),
                99
            )


            next_state_key = (
                next_pos,
                tuple(sorted(next_marking))
            )

            next_loop_depth = _gen_visited.get(
                next_state_key,
                0
            )


            offline_next_state = {
                "marking": next_marking,
                "position": next_pos,
                "activity_id": next_act_id,
                "loop_depth": next_loop_depth,
                "remaining_cost": next_cost_id
            }
            if dataset_path is not None:
                offline_dataset.append({
                    "prefix": prefix,
                    "state": offline_state,
                    "action": offline_action,
                    "next_state": offline_next_state,
                    "reward": float(reward),
                    "done": done
                })

            label = env.LABEL_ID_SPACE[label_str]
            move  = env.MOVE_ID_SPACE[move_str]

            data['rewards'].append(float(reward))
            new_mv = env.marking_vec()
            data['marks'].append(mv.clone())
            data['moves'].append(move)
            data['labels'].append(label)
            data['moves_str'].append(move_str)
            data['labels_str'].append(label_str)
            data['old_lps'].append(old_lp)
            data['values'].append(value.item())
            data['dones'].append(done)
            data['costs'].append(cost_id)

            mv = new_mv
            i += 1

            if i == max_len:
                print("at step, NOT forced to stop: ", i)
            # if i >= max_len:
            #     done = True    
            if done:
                break
        if dataset_path is not None and len(offline_dataset) > 0:
            save_episode_transitions(
                offline_dataset,
                dataset_path
            )
        return data, n_invalid  
    

import os 
def save_episode_transitions(transitions, path):

    if os.path.exists(path):
        dataset = torch.load(path)
    else:
        dataset = []

    dataset.extend(transitions)

    torch.save(dataset, path)
import torch
import torch.nn as nn
import pm4py 
from pm4py.objects.petri_net.obj import Marking
from .dataset_utils import save_episode_transitions
from train.ppo_env import _m_tuple   
from baselines.A_start_baseline.dataset_generation_a_star import _astar_prefix_alignment
from train.ppo_env import _m_tuple, MOVE_COST
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
    def __init__(self, vocab_size: int, n_places: int, n_labels: int, heuristic_net,
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
        self.heuristic_net = heuristic_net
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

        self.cost_proj = nn.Linear(1, emb_dim)    
        self.fuse_proj = nn.Linear(emb_dim * 5, emb_dim, bias=False)

        self.dec          = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.pos_emb      = nn.Embedding(70, emb_dim)
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

        cost_t   = torch.tensor([[[remaining_cost]]], device=device, dtype=torch.float32)
        cost_emb = self.cost_proj(cost_t)
        inp = self.fuse_proj(torch.cat(
            [marking_emb, activity_emb, pos_emb, depth_emb, cost_emb], dim=-1
        ))                                                            # (1,1,E)

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
        _gen_visited: dict = {}

        while not done:
            act    = env.current_activity()
            act_id = vocab.t2i.get(act, vocab.t2i["<UNK>"]) if act else 0
            data['act_ids'].append(act_id)
            pos = env.pos

            current_marking = normalize_marking_tuple(
                env._marking_to_dict()
            )
            state_key  = (pos, tuple(sorted(_m_tuple(env._marking_to_dict()))))
            loop_depth = _gen_visited.get(state_key, 0)
            # Save current state for offline RL
            offline_state = {
                "marking": current_marking,
                "position": pos,
                "activity_id": act_id,
                "loop_depth": loop_depth,
                # "remaining_cost": cost_id
            }
            # --- hard structural exit: third revisit means we are stuck -----
            # (only as safety; the policy should learn to avoid this)
            # if loop_depth >= 3:
            #     break

            # increment visit counter for this step
            _gen_visited[state_key] = loop_depth + 1

            data['positions'].append(pos)
            data['loop_depths'].append(loop_depth)   # store for replay

            remaining = prefix[pos:]
            src_remaining = (torch.tensor([vocab.encode(remaining)], device=mv.device) if remaining
                            else torch.zeros((1, 1), dtype=torch.long, device=mv.device))
            with torch.no_grad():
                predicted_cost = self.heuristic_net(mv, src_remaining).item()
            data['costs'].append(predicted_cost)        

            label_logits, value, h, attn_weights = self.decode_step(
                pos, mv, h, enc_out, act_id,
                loop_depth=loop_depth,
                remaining_cost=predicted_cost
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
                compute_reward=compute_reward,
                loop_depth=loop_depth        
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


            # _, next_remaining_cost, _ = _astar_prefix_alignment(
            #     prefix=prefix,
            #     start_marking=next_marking,
            #     start_pos=next_pos
            # )

            # next_cost_id = min(
            #     int(next_remaining_cost),
            #     99
            # )


            next_state_key = (
                next_pos,
                next_marking   
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



    @torch.no_grad()
    def generate_with_lookahead(
        self,
        src,
        prefix,
        env,
        vocab,
        top_k: int = 5,
        max_len: int = 60,
        compute_reward: bool = False,
    ):
        
        self.eval()

        data = dict(
            labels_str=[], moves_str=[], rewards=[],
            dones=[], positions=[], costs=[],
        )

        enc_out, h = self.encode(src)
        mv = env.reset(prefix)
        done = False
        i = 0
        _gen_visited: dict = {}

        while not done and i < max_len:
            act    = env.current_activity()
            act_id = vocab.t2i.get(act, vocab.t2i["<UNK>"]) if act else 0
            pos    = env.pos

            state_key  = (pos, tuple(sorted(_m_tuple(env._marking_to_dict()))))
            loop_depth = _gen_visited.get(state_key, 0)
            _gen_visited[state_key] = loop_depth + 1

            remaining = prefix[pos:]
            src_remaining = (
                torch.tensor([vocab.encode(remaining)], device=mv.device) if remaining
                else torch.zeros((1, 1), dtype=torch.long, device=mv.device)
            )
            predicted_cost = self.heuristic_net(mv, src_remaining).item()
            data['costs'].append(predicted_cost)

            # one decode_step per real step — h evolves independent of the
            # action that ends up chosen, so this is correct even though
            # the action isn't selected until after this call
            label_logits, value, h, attn_weights = self.decode_step(
                pos, mv, h, enc_out, act_id,
                loop_depth=loop_depth, remaining_cost=predicted_cost
            )

            valid_labels_mask = env.valid_label_mask()
            policy_bias = torch.zeros(len(env.LABEL_SPACE), device=label_logits.device)
            if i == 0:
                policy_bias = self.prefix_policy_bias(
                    env, valid_labels_mask, data['moves_str'], label_logits.device, 50.0
                )

            ll = label_logits.clone()
            ll[0] = ll[0] + policy_bias
            ll[0][~valid_labels_mask] = float('-inf')
            probs = torch.softmax(ll[0], -1)

            if torch.isnan(probs).any() or probs.sum() < 1e-9:
                break

            # restrict candidates to genuinely valid indices first, THEN
            # rank by probability — avoids any float-underflow tie issues
            # between low-prob valid actions and zeroed-out invalid ones
            valid_idx   = valid_labels_mask.nonzero(as_tuple=True)[0]
            valid_probs = probs[valid_idx]
            k = max(1, min(top_k, valid_idx.numel()))
            _, top_pos = torch.topk(valid_probs, k)
            candidate_ids = valid_idx[top_pos].tolist()

            current_m   = env._marking_to_dict()
            current_pos = env.pos

            best_score, best_label_id, best_move_id = None, None, None
            for label_id in candidate_ids:
                label_str = env.LABEL_SPACE[label_id]
                move_id   = env.infere_move_type(label_id)
                move_str  = env.MOVE_SPACE[move_id]

                new_m, new_pos = env._simulate_fire(current_m, current_pos, move_str, label_str)

                if new_pos >= len(prefix):
                    h_next = 0.0     # prefix fully consumed — true cost-to-go is 0
                else:
                    next_src = torch.tensor([vocab.encode(prefix[new_pos:])], device=mv.device)
                    h_next   = self.heuristic_net(env._vec_for_marking(new_m), next_src).item()

                score = MOVE_COST[move_str] + h_next
                if best_score is None or score < best_score:
                    best_score, best_label_id, best_move_id = score, label_id, move_id

            label_id, move_id = best_label_id, best_move_id
            move_str  = env.MOVE_SPACE[move_id]
            label_str = env.LABEL_SPACE[label_id]

            moves_for_all_labels = [env.infere_move_type(j) for j in range(len(env.LABEL_SPACE))]
            label_str, move_str, reward, done = env.step(
                self, valid_labels_mask, move_id, label_id,
                list(data['moves_str']), list(data['labels_str']),
                label_logits[0], attn_weights, moves_for_all_labels,
                compute_reward=compute_reward, loop_depth=loop_depth
            )

            data['positions'].append(pos)
            data['rewards'].append(float(reward))
            data['moves_str'].append(move_str)
            data['labels_str'].append(label_str)
            data['dones'].append(done)

            mv = env.marking_vec()
            i += 1
            if done:
                break

        return data 
    


from collections import defaultdict
import torch
import torch.nn as nn

class PetriHeuristicGNN(nn.Module):
    """
    Learns to predict A*-style remaining alignment cost from
    (marking, remaining_prefix) — no search at inference time.

    Topology (places, transitions, arcs) is fixed and built once;
    only the marking vector and remaining-prefix ids vary per call.
    """

    def __init__(self, net, place_list, place_idx, vocab_size,
                 hidden_dim=64, emb_dim=64, n_layers=3):
        super().__init__()
        self.place_list = place_list
        self.place_idx  = place_idx          # share env.place_idx, same Place objects
        self.n_places   = len(place_list)

        self.transitions = list(net.transitions)
        self.trans_idx    = {t: i for i, t in enumerate(self.transitions)}
        self.n_trans      = len(self.transitions)
        self.n_layers     = n_layers

        # ---- static topology, built once ----
        p2t = defaultdict(list)   # trans_i -> [(place_i, weight), ...]  (consumption)
        t2p = defaultdict(list)   # place_i -> [(trans_i, weight), ...]  (production)
        for arc in net.arcs:
            if arc.source in self.place_idx and arc.target in self.trans_idx:
                p2t[self.trans_idx[arc.target]].append(
                    (self.place_idx[arc.source], float(arc.weight))
                )
            elif arc.source in self.trans_idx and arc.target in self.place_idx:
                t2p[self.place_idx[arc.target]].append(
                    (self.trans_idx[arc.source], float(arc.weight))
                )
        self.p2t, self.t2p = p2t, t2p

        # ---- learned per-node identity (the topology is fixed, so each
        #      node can just learn "who it is") ----
        self.place_id_emb = nn.Embedding(self.n_places, hidden_dim)
        self.trans_id_emb = nn.Embedding(self.n_trans,  hidden_dim)
        self.marking_proj = nn.Linear(1, hidden_dim)     # token count -> hidden

        self.place_update = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                           nn.LayerNorm(hidden_dim))
            for _ in range(n_layers)
        ])
        self.trans_update = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                           nn.LayerNorm(hidden_dim))
            for _ in range(n_layers)
        ])

        # ---- remaining-prefix encoder (small, separate from the policy's) ----
        self.prefix_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.prefix_gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _aggregate(self, src_emb, neighbor_lists, n_dst, device):
        H = src_emb.size(-1)
        out = torch.zeros(n_dst, H, device=device)
        for dst_i, neighbors in neighbor_lists.items():
            if not neighbors:
                continue
            idxs = torch.tensor([n[0] for n in neighbors], device=device)
            w    = torch.tensor([n[1] for n in neighbors], device=device).unsqueeze(-1)
            out[dst_i] = (src_emb[idxs] * w).sum(0) / w.sum()
        return out

    def graph_embed(self, marking_vec: torch.Tensor) -> torch.Tensor:
        device  = marking_vec.device
        place_h = self.place_id_emb.weight + self.marking_proj(marking_vec.unsqueeze(-1))
        trans_h = self.trans_id_emb.weight

        for l in range(self.n_layers):
            trans_in     = self._aggregate(place_h, self.p2t, self.n_trans, device)
            new_trans_h  = self.trans_update[l](torch.cat([trans_h, trans_in], -1))
            place_in     = self._aggregate(trans_h, self.t2p, self.n_places, device)
            new_place_h  = self.place_update[l](torch.cat([place_h, place_in], -1))
            trans_h, place_h = new_trans_h, new_place_h

        total    = marking_vec.sum().clamp(min=1.0)
        weighted = (place_h * marking_vec.unsqueeze(-1)).sum(0) / total   # "where are the tokens"
        pooled   = place_h.mean(0)                                        # global structural context
        return torch.cat([weighted, pooled], -1)

    def forward(self, marking_vec: torch.Tensor, remaining_ids: torch.Tensor) -> torch.Tensor:
        g_emb = self.graph_embed(marking_vec)
        _, h  = self.prefix_gru(self.prefix_emb(remaining_ids))
        p_emb = h.squeeze(0).squeeze(0)
        return self.head(torch.cat([g_emb, p_emb], -1)).squeeze(-1)
    



from collections import deque
import random

class HeuristicBuffer:
    def __init__(self, capacity=20000):
        self.data = deque(maxlen=capacity)

    def add(self, marking_vec, remaining_prefix, true_cost):
        self.data.append((marking_vec.detach().cpu(), list(remaining_prefix), float(true_cost)))

    def sample(self, batch_size):
        return random.sample(self.data, k=min(batch_size, len(self.data)))


def train_heuristic_step(heuristic_net, opt, vocab, batch, device):
    heuristic_net.train()
    opt.zero_grad()
    preds, targets = [], []
    for marking_vec, remaining, true_cost in batch:
        src = (torch.tensor([vocab.encode(remaining)], device=device) if remaining
               else torch.zeros((1, 1), dtype=torch.long, device=device))
        preds.append(heuristic_net(marking_vec.to(device), src))
        targets.append(true_cost)
    loss = nn.functional.smooth_l1_loss(torch.stack(preds),
                                         torch.tensor(targets, device=device))
    loss.backward()
    nn.utils.clip_grad_norm_(heuristic_net.parameters(), 1.0)
    opt.step()
    return loss.item()



import os 
def save_episode_transitions(transitions, path):

    if os.path.exists(path):
        dataset = torch.load(path)
    else:
        dataset = []

    dataset.extend(transitions)

    torch.save(dataset, path)
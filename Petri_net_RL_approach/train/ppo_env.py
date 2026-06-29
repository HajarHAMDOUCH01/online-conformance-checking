from __future__ import annotations

import itertools
from collections import defaultdict, deque

import torch
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.semantics import ClassicSemantics
from baselines.A_start_baseline.dataset_generation_a_star import _astar_prefix_alignment
MOVE_COST = {"S": 0.0, "M": 1.0, "L": 2.0}
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

# helper used by both env and any external callers (including ppo_model.generate)
def _m_tuple(m_dict: dict) -> tuple:
    """
    Convert a marking dict to a canonical, hashable, sorted tuple.
    Accepts both Place-keyed (pm4py Marking) and str-keyed dicts.
    """
    return tuple(sorted(
        (p.name if hasattr(p, "name") else p, c)
        for p, c in m_dict.items()
        if c > 0
    ))

# ---------------------------------------------------------------------------
# AlignmentEnv
# ---------------------------------------------------------------------------

class AlignmentEnv:
    """
    RL environment for computing prefix alignments over a Petri net.
    """
    _shared_closure_cache: dict = {}
    _shared_labels_cache:  dict = {}
    _shared_path_cache:    dict = {}

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(self, net, im, labels: list[str]):
        self.net = net
        self.im  = im
        self.sem = ClassicSemantics()

        self.MOVE_SPACE     = ["S", "M", "L"]
        self.LABEL_SPACE    = list(labels)
        self.MOVE_ID_SPACE  = {move: i  for i, move  in enumerate(self.MOVE_SPACE)}
        self.ID_LABEL_SPACE = {i: label for i, label in enumerate(self.LABEL_SPACE)}
        self.LABEL_ID_SPACE = {label: i for i, label in enumerate(self.LABEL_SPACE)}
        self.ID_MOVE_SPACE  = {i: move  for i, move  in enumerate(self.MOVE_SPACE)}

        self.place_list = sorted(net.places, key=lambda p: p.name)
        self.place_idx  = {p: i for i, p in enumerate(self.place_list)}
        self.n_places   = len(self.place_list)

        self._in_arcs:  dict = defaultdict(list)
        self._out_arcs: dict = defaultdict(list)
        places      = set(net.places)
        transitions = set(net.transitions)
        for arc in net.arcs:
            if arc.source in places:
                self._in_arcs[arc.target].append((arc.source.name, arc.weight))
            elif arc.source in transitions:
                self._out_arcs[arc.source].append((arc.target.name, arc.weight))

        self._visible_transitions = [t for t in net.transitions if t.label is not None]
        self._silent_transitions  = [t for t in net.transitions if t.label is None]
        self._label_to_trans: dict = defaultdict(list)
        for t in self._visible_transitions:
            self._label_to_trans[t.label].append(t)

        self._sink_place_name = "sink"
        self._im_name_dict    = {p.name: cnt for p, cnt in im.items() if cnt > 0}

    # =========================================================================
    # Episode lifecycle
    # =========================================================================

    def reset(self, prefix: list) -> torch.Tensor:
        self.prefix  = list(prefix)
        self.pos     = 0
        # self.marking = Marking({p: v for p, v in self.im.items()})
        self.marking = {
            p.name:v
            for p,v in self.im.items()
            if v > 0
        }
        self._inserted_model_moves = []
        self.steps_without_progress = 0
        self.visited_states = {}
        return self.marking_vec()
    
    def _vec_for_marking(self, m) -> torch.Tensor:
        if isinstance(m, tuple):
            m = dict(m)
        v = torch.zeros(self.n_places)
        for pname, cnt in m.items():
            for p, idx in self.place_idx.items():
                if p.name == pname:
                    v[idx] = float(cnt)
                    break
        return v

    def marking_vec(self):
        v = torch.zeros(self.n_places)

        for pname, cnt in self._marking_to_dict().items():
            for p, idx in self.place_idx.items():
                if p.name == pname:
                    v[idx] = float(cnt)
                    break

        return v

    def current_activity(self):
        return self.prefix[self.pos] if self.pos < len(self.prefix) else None

    def is_done(self) -> bool:
        return self.pos >= len(self.prefix)

    # =========================================================================
    # Low-level Petri-net helpers
    # =========================================================================

    def _is_enabled(self, m_dict: dict, transition) -> bool:
        return all(
            m_dict.get(pname, 0) >= w
            for pname, w in self._in_arcs[transition]
        )

    def _fire(self, m_dict: dict, transition) -> dict:
        result = dict(m_dict)
        for pname, w in self._in_arcs[transition]:
            result[pname] = result.get(pname, 0) - w
            if result[pname] == 0:
                del result[pname]
        for pname, w in self._out_arcs[transition]:
            result[pname] = result.get(pname, 0) + w
        return result

    def _marking_to_dict(self) -> dict:
        return {
            p.name if hasattr(p, "name") else p: v
            for p, v in self.marking.items()
            if v > 0
        }

    def _pm4py_marking_to_name_dict(self, marking) -> dict:
        return {p.name: cnt for p, cnt in marking.items() if cnt > 0}

    def _enabled_visible(self) -> set:
        m_dict = self._marking_to_dict()

        return {
            t.label
            for t in self._visible_transitions
            if self._is_enabled(m_dict, t)
        }


    def real_enabled_visible(self) -> list:
        m_dict = self._marking_to_dict()

        return [
            t
            for t in self._visible_transitions
            if self._is_enabled(m_dict, t)
        ]

    # =========================================================================
    # Silent-closure subsystem
    # =========================================================================

    def _silent_reachable(self, m_dict: dict) -> frozenset:
        key    = _m_tuple(m_dict)
        cached = AlignmentEnv._shared_closure_cache.get(key)
        if cached is not None:
            return cached

        visited  = {key}
        frontier = [m_dict]
        while frontier:
            curr = frontier.pop()
            for t in self._silent_transitions:
                if self._is_enabled(curr, t):
                    nm   = self._fire(curr, t)
                    nkey = _m_tuple(nm)
                    if nkey not in visited:
                        visited.add(nkey)
                        frontier.append(nm)

        result = frozenset(visited)
        AlignmentEnv._shared_closure_cache[key] = result
        return result

    def _labels_enabled_after_silent(self, m_dict: dict) -> frozenset:
        if m_dict and hasattr(next(iter(m_dict)), "name"):
            m_dict = {p.name: v for p, v in m_dict.items() if v > 0}

        key    = _m_tuple(m_dict)
        cached = AlignmentEnv._shared_labels_cache.get(key)
        if cached is not None:
            return cached

        reachable = self._silent_reachable(m_dict)
        enabled   = set()
        for m_tup in reachable:
            m_tmp = dict(m_tup)
            for lbl, trans_list in self._label_to_trans.items():
                if lbl not in enabled:
                    for t in trans_list:
                        if self._is_enabled(m_tmp, t):
                            enabled.add(lbl)
                            break

        result = frozenset(enabled)
        AlignmentEnv._shared_labels_cache[key] = result
        return result

    def _silent_path_to(
        self, m_dict: dict, target_label: str
    ) -> tuple[list, object] | tuple[None, None]:
        key    = (_m_tuple(m_dict), target_label)
        cached = AlignmentEnv._shared_path_cache.get(key)
        if cached is not None:
            return cached

        visited = {_m_tuple(m_dict)}
        queue   = deque([(m_dict, [])])

        while queue:
            curr, path = queue.popleft()

            for t in self._label_to_trans.get(target_label, []):
                if self._is_enabled(curr, t):
                    result = (path, t)
                    AlignmentEnv._shared_path_cache[key] = result
                    return result

            for tau in self._silent_transitions:
                if self._is_enabled(curr, tau):
                    nm  = self._fire(curr, tau)
                    nk  = _m_tuple(nm)
                    if nk not in visited:
                        visited.add(nk)
                        queue.append((nm, path + [tau]))

        AlignmentEnv._shared_path_cache[key] = (None, None)
        return None, None

    def _replay_silent_path(self, m_dict: dict, silent_path: list) -> dict:
        result = dict(m_dict)
        for tau in silent_path:
            result = self._fire(result, tau)
        return result

    # =========================================================================
    # Decision interface
    # =========================================================================

    def infere_move_type(self, label) -> int:
        label_str   = self.LABEL_SPACE[label] if isinstance(label, int) else label
        act         = self.current_activity()
        m_dict      = self._marking_to_dict()
        all_enabled = self._labels_enabled_after_silent(m_dict)

        if label_str == act:
            return 0 if label_str in all_enabled else 2
        else:
            if label_str in all_enabled:
                return 1
            else:
                return 3

    def valid_label_mask(self) -> torch.Tensor:
        m_dict      = self._marking_to_dict()
        current_tup = _m_tuple(m_dict)
        act         = self.current_activity()

        safe_fireable: set[str] = set()
        for lbl in self._label_to_trans:
            silent_path, matching_t = self._silent_path_to(m_dict, lbl)
            if silent_path is None or matching_t is None:
                continue

            target_m = self._replay_silent_path(m_dict, silent_path)
            new_m    = self._fire(target_m, matching_t)
            new_tup  = _m_tuple(new_m)

            is_current_activity = lbl == act
            if new_tup != current_tup and (is_current_activity or self._sink_place_name not in new_m):
                safe_fireable.add(lbl)

        mask = [
            (lbl == act) or (lbl in safe_fireable)
            for lbl in self.LABEL_SPACE
        ]
        result = torch.tensor(mask, dtype=torch.bool)
        return result

    # =========================================================================
    # Step
    # =========================================================================

    def step(self, model, i, valid_labels_mask, move_id, label_id,
            prev_moves, prev_labels, labels_logits, attn_weights,
            moves_for_all_labels, compute_reward=True, loop_depth=0):


        current_marking = self.marking
        current_pos = self.pos

        move  = self.MOVE_SPACE[move_id]
        label = self.LABEL_SPACE[label_id]

        if self.is_done():
            return label, move, 0.0, True

        if move == "L":
            self.pos += 1

        if move in ("S", "M"):
            current_m   = self._marking_to_dict()
            current_tup = _m_tuple(current_m)
            fired       = False

            # Stage 1: direct firing
            for t in self.real_enabled_visible():
                if t.label != label:
                    continue
                new_m   = self._fire(current_m, t)
                new_tup = _m_tuple(new_m)
                if new_tup == current_tup:
                    continue
                if self._sink_place_name in new_m and move == "M":
                    continue
                self.marking = new_m
                # self.marking = self.sem.weak_execute(t, self.net, self.marking)
                fired = True
                break

            if not fired:
                silent_path, matching_t = self._silent_path_to(current_m, label)
                if silent_path is not None and matching_t is not None:
                    target_m = self._replay_silent_path(current_m, silent_path)
                    new_m    = self._fire(target_m, matching_t)
                    new_tup  = _m_tuple(new_m)

                    # if the agent samples a model move that takes it to a marking in the final place, don't fire that move
                    # valid = (new_tup != current_tup) and (
                    #     self._sink_place_name not in new_m or move != "M"
                    # )
                    # if valid:
                
                    target_m = self._replay_silent_path(
                    current_m,
                    silent_path
                    )

                    self.marking = self._fire(
                        target_m,
                        matching_t
                    )
                    fired = True
                        # for tau in silent_path:
                        #     self.marking = self.sem.weak_execute(
                        #         tau, self.net, self.marking
                        #     )
                        # self.marking = self.sem.weak_execute(
                        #     matching_t, self.net, self.marking
                        # )
                        
                    # else: 
                    #     self.marking = current_marking
                    #     self.pos = current_pos
                    #     return None, None, None, None


            if move == "M":
                jumped_to_final_marking_without_finishing_prefix = (self._sink_place_name == new_m)
                # for looping behaviour , reward already handles it
            if move == "S" and fired:
                self.pos += 1

        prev_moves.append(move)  
        prev_labels.append(label)

        if compute_reward:
            total_reward, force_done = self.reward_function(
                prev_moves, prev_labels, labels_logits, attn_weights,
                moves_for_all_labels, current_marking, current_pos,
                loop_depth=loop_depth, jumped_to_final_marking_without_finishing_prefix=jumped_to_final_marking_without_finishing_prefix, position=i
            )
        else:
            total_reward = 0.0
            force_done = False

        done = self.is_done() or force_done

        return (
            prev_labels[-1],
            prev_moves[-1],
            total_reward,
            done
        )


    def _simulate_fire(self, m_dict: dict, pos: int, move: str, label: str):
        
        new_pos = pos
        new_m   = m_dict

        if move == "L":
            new_pos = pos + 1

        if move in ("S", "M"):
            current_tup = _m_tuple(m_dict)
            fired = False

            for t in self._label_to_trans.get(label, []):
                if not self._is_enabled(m_dict, t):
                    continue
                candidate = self._fire(m_dict, t)
                if _m_tuple(candidate) == current_tup:
                    continue
                if self._sink_place_name in candidate and move == "M":
                    continue
                new_m = candidate
                fired = True
                break

            if not fired:
                silent_path, matching_t = self._silent_path_to(m_dict, label)
                if silent_path is not None and matching_t is not None:
                    target_m  = self._replay_silent_path(m_dict, silent_path)
                    candidate = self._fire(target_m, matching_t)
                    valid = (_m_tuple(candidate) != current_tup) and (
                        self._sink_place_name not in candidate or move != "M"
                    )
                    if valid:
                        new_m = candidate
                        fired = True

            if move == "S":
                new_pos = pos + 1

        return new_m, new_pos

    # =========================================================================
    # Reward
    # =========================================================================

    def reward_function(self, new_moves, new_labels, labels_logits, attn_weights,
                        moves_for_all_labels, current_marking, current_pos,
                        loop_depth=0, jumped_to_final_marking_without_finishing_prefix=False, position=0):

        label = new_labels[-1]
        move  = new_moves[-1]
        reward = 0.0
        original_prefix      = self.prefix
        after_this_step_marking = self.marking
        after_this_step_pos     = self.pos

        current_marking = normalize_marking_tuple(current_marking)
        after_this_step_marking = normalize_marking_tuple(after_this_step_marking)
        
        
        # print("current_marking type:", type(current_marking))
        # print("current_marking:", current_marking)

        # print("after_marking type:", type(after_this_step_marking))
        # print("after_marking:", after_this_step_marking)
        alignment_before, cost_before, _ = _astar_prefix_alignment(
            prefix=original_prefix,
            start_marking=current_marking,
            start_pos=current_pos
        )
        alignment_after, cost_after, _ = _astar_prefix_alignment(
            prefix=original_prefix,
            start_marking=after_this_step_marking,
            start_pos=after_this_step_pos
        )

        if getattr(self, "heuristic_buffer", None) is not None:
            self.heuristic_buffer.add(
                self._vec_for_marking(current_marking),
                original_prefix[current_pos:], cost_before
            )
            self.heuristic_buffer.add(
                self._vec_for_marking(after_this_step_marking),
                original_prefix[after_this_step_pos:], cost_after
            )
        # print("alignement before : ", alignment_before)
        # print("alignement after : ", alignment_after)
        # ------------------------------------------------------------------
        # Base reward: A* cost improvement 
        # ------------------------------------------------------------------
        
        delta_h      = cost_before - cost_after        
        move_cost    = MOVE_COST[move]                  
        inefficiency = move_cost - delta_h              
        reward       = -inefficiency


        # ------------------------------------------------------------------
        # State-visit accounting
        # ------------------------------------------------------------------
        prev_state_key = (
            current_pos,
            current_marking
        )

        new_state_key = (
            after_this_step_pos,
            after_this_step_marking
        )

        prev_visit_count = self.visited_states.get(prev_state_key, 0)
        new_state_is_novel = new_state_key not in self.visited_states

        self.visited_states[new_state_key] = (
            self.visited_states.get(new_state_key, 0) + 1
        )
        new_visit_count = self.visited_states[new_state_key]
        terminate = False
        # revisit penalty
        reward -= 0.5 * (new_visit_count - 1)
        # reward -= 0.5 * loop_depth

        # completion
        if after_this_step_pos == len(original_prefix):
            reward += 15.0
            terminate = True

        if jumped_to_final_marking_without_finishing_prefix:
            reward -= 15.0
            terminate = True
        if position >= 60:
            reward -= 15.0
            terminate = True

        print(
            move,
            reward
        )

        return reward, terminate
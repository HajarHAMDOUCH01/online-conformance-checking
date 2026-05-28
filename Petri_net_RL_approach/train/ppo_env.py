from __future__ import annotations

import heapq
import random
import itertools
from collections import defaultdict

import torch
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.semantics import ClassicSemantics

LOOKAHEAD_LAMBDA = 0.5

_dijkstra_counter = itertools.count()


class AlignmentEnv:
    def __init__(self, net, im, labels: list[str]):
        self.net = net
        self.im  = im
        self.sem = ClassicSemantics()

        self.MOVE_SPACE  = ["S", "M", "L"]
        self.MOVE_ID_SPACE = {move: i for i, move in enumerate(self.MOVE_SPACE)}
        self.ID_MOVE_SPACE = {i: move for i, move in enumerate(self.MOVE_SPACE)}
        self.LABEL_SPACE = list(labels)

        self.place_list = sorted(net.places, key=lambda p: p.name)
        self.place_idx  = {p: i for i, p in enumerate(self.place_list)}
        self.n_places   = len(self.place_list)

        self._in_arcs  = defaultdict(list)
        self._out_arcs = defaultdict(list)
        for arc in net.arcs:
            if isinstance(arc.source, PetriNet.Place):
                self._in_arcs[arc.target].append((arc.source.name, arc.weight))
            else:
                self._out_arcs[arc.source].append((arc.target.name, arc.weight))

        self._visible_transitions = [t for t in net.transitions if t.label is not None]
        self._silent_transitions  = [t for t in net.transitions if t.label is None]
        self._label_to_trans      = defaultdict(list)
        for t in self._visible_transitions:
            self._label_to_trans[t.label].append(t)

        self._im_name_dict = {p.name: cnt for p, cnt in im.items() if cnt > 0}

    def reset(self, prefix: list) -> torch.Tensor:
        self.prefix  = list(prefix)
        self.pos     = 0
        self.marking = Marking({p: v for p, v in self.im.items()})
        self._inserted_model_moves: list[str] = []
        # self._tau_closure()
        return self.marking_vec()

    def marking_vec(self) -> torch.Tensor:
        v = torch.zeros(self.n_places)
        for p, cnt in self.marking.items():
            if p in self.place_idx:
                v[self.place_idx[p]] = float(cnt)
        return v

    def current_activity(self):
        return self.prefix[self.pos] if self.pos < len(self.prefix) else None

    def is_done(self) -> bool:
        return self.pos >= len(self.prefix)

    def valid_move_mask(self) -> torch.Tensor:
        mask = torch.zeros(len(self.MOVE_SPACE), dtype=torch.bool)

        if self.is_done():
            mask[2] = True
            return mask

        act     = self.current_activity()
        enabled = self._enabled_visible()

        mask[2] = True  

        if act is not None:
            if any(t.label == act for t in enabled):
                mask[0] = True  

        if len(enabled) > 0:
            mask[1] = True  

        return mask

    def valid_label_mask(self, move: str) -> torch.Tensor:
        mask    = torch.zeros(len(self.LABEL_SPACE), dtype=torch.bool)
        act     = self.current_activity()
        enabled = self._enabled_visible()

        if act is None:
            return mask

        if move == "S":
            if act in self.LABEL_SPACE:
                mask[self.LABEL_SPACE.index(act)] = True

        elif move == "M":
            """
            For M move: 
            True for all labels that correspond 
            to currently enabled visible transitions. 
            The model can fire any enabled activity, 
            regardless of what the trace shows."""
            enabled_labels = {t.label for t in enabled if t.label is not None}
            for i, l in enumerate(self.LABEL_SPACE):
                if l in enabled_labels:
                    mask[i] = True

        elif move == "L":
            if act in self.LABEL_SPACE:
                mask[self.LABEL_SPACE.index(act)] = True

        return mask

    def _intermediate_prefix(self) -> list[str]:
        return self._inserted_model_moves + self.prefix[self.pos:]

    def step(self, move_id: int, label_id: int) -> tuple[float, bool]:
        move  = self.MOVE_SPACE[move_id]
        label = self.LABEL_SPACE[label_id]

        if self.is_done():
            return 0.0, True

        # marking_before = self._pm4py_marking_to_name_dict(self.marking)
        # prefix_before  = self._intermediate_prefix()
        # c_before = self._dijkstra_remaining_cost(marking_before, prefix_before)

        act     = self.current_activity()
        enabled = self._enabled_visible()
        base_reward = 0.0

        if move == "L":
            base_reward = -1.0
            self.pos += 1

        elif move in ("S", "M"):
            fired = False
            for t in enabled:
                if t.label == label:
                    self.marking = self.sem.weak_execute(t, self.net, self.marking)
                    fired = True
                    break

            if not fired:
                # This should never happen with masking - indicates a exception
                raise RuntimeError(f"Invalid {move}-move: label '{label}' not enabled. "
                                f"Enabled: {[t.label for t in enabled]}. "
                                f"Current activity: {self.current_activity()}")

            if move == "S":
                base_reward = +1.0
                self.pos += 1
            elif move == "M":
                base_reward = -1.0
                self._inserted_model_moves.append(label)

        else:
            base_reward = -1.0

        # self._tau_closure()

        # marking_after = self._pm4py_marking_to_name_dict(self.marking)
        # prefix_after  = self._intermediate_prefix()
        # c_after = self._dijkstra_remaining_cost(marking_after, prefix_after)

        # shaping      = LOOKAHEAD_LAMBDA * (c_before - c_after)
        total_reward = base_reward 

        return total_reward, self.is_done()

    def _dijkstra_remaining_cost(
        self,
        start_marking_dict: dict,
        remaining_prefix: list,
    ) -> float:
        if not remaining_prefix:
            return 0.0

        goal_pos   = len(remaining_prefix)
        start_tup  = tuple(sorted(
            (p, c) for p, c in start_marking_dict.items() if c > 0
        ))
        init_state = (start_tup, 0)

        dist  = {init_state: 0}
        heap  = [(0, next(_dijkstra_counter), start_tup, 0)]

        while heap:
            cost, _, m_tup, pos = heapq.heappop(heap)
            state = (m_tup, pos)

            if cost > dist.get(state, float('inf')):
                continue

            if pos == goal_pos:
                # print("\n dijkstra remaining cost = ", cost)
                return float(cost)

            act    = remaining_prefix[pos]
            m_dict = dict(m_tup)

            def _push(new_m_dict, new_pos, new_cost, _s=state):
                new_tup = tuple(sorted(
                    (p, c) for p, c in new_m_dict.items() if c > 0
                ))
                ns = (new_tup, new_pos)
                if new_cost < dist.get(ns, float('inf')):
                    dist[ns] = new_cost
                    heapq.heappush(
                        heap, (new_cost, next(_dijkstra_counter), new_tup, new_pos)
                    )

            # for reachable_m in self._tau_closure_dict(m_dict):
            reachable_markings = self._tau_closure_dict(m_dict)
            reachable_m = random.choice(reachable_markings)
            # sync
            for t in self._label_to_trans.get(act, []):
                if self._dijk_enabled(reachable_m, t):
                    _push(self._dijk_fire(reachable_m, t), pos + 1, cost)
            # model
            for t in self._visible_transitions:
                if self._dijk_enabled(reachable_m, t):
                    _push(self._dijk_fire(reachable_m, t), pos, cost + 1)
            # log
            _push(reachable_m, pos + 1, cost + 1)
        
        return float('inf')

    def _dijk_enabled(self, m_dict: dict, transition) -> bool:
        return all(
            m_dict.get(pname, 0) >= w
            for pname, w in self._in_arcs[transition]
        )

    def _dijk_fire(self, m_dict: dict, transition) -> dict:
        result = dict(m_dict)
        for pname, w in self._in_arcs[transition]:
            result[pname] = result.get(pname, 0) - w
            if result[pname] == 0:
                del result[pname]
        for pname, w in self._out_arcs[transition]:
            result[pname] = result.get(pname, 0) + w
        return result

    def _pm4py_marking_to_name_dict(self, marking) -> dict:
        return {p.name: cnt for p, cnt in marking.items() if cnt > 0}

    def _enabled_visible(self):
        return [
            t for t in self.sem.enabled_transitions(self.net, self.marking)
            if t.label is not None
        ]

    def _tau_closure(self):
        changed = True
        safety  = 0
        while changed and safety < 200:
            changed = False
            safety += 1
            for t in self.sem.enabled_transitions(self.net, self.marking):
                if t.label is None:
                    self.marking = self.sem.weak_execute(t, self.net, self.marking) # this keeps the alst obtained marking 
                    changed = True
                    break

    def _tau_closure_dict(self, m_dict: dict) -> list[dict]:
        """Return all markings reachable via silent transitions from m_dict."""
        visited = set()
        queue   = [m_dict]
        result  = []

        while queue:
            cur = queue.pop()
            key = tuple(sorted(cur.items()))
            if key in visited:
                continue
            visited.add(key)
            result.append(cur)
            for t in self._silent_transitions:
                if self._dijk_enabled(cur, t):
                    queue.append(self._dijk_fire(cur, t))
        return result
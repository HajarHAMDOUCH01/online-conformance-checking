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
    
    def infere_move_type(self, label):
        act     = self.current_activity()
        enabled = self._enabled_visible()
        label_str = self.LABEL_SPACE[label]
        if label_str is not None:
            if (label_str in enabled):
                move_id = 1
                if any(t == act for t in enabled):
                    move_id =  0
            else: 
                move_id = 2
        return move_id

    def valid_label_mask(self) -> torch.Tensor:
        enabled = self._enabled_visible()
        mask = torch.tensor([label in enabled for label in self.LABEL_SPACE], dtype=torch.bool)
        return mask

    def _intermediate_prefix(self) -> list[str]:
        return self._inserted_model_moves + self.prefix[self.pos:]

    def step(self, move_id: int, label_id: int) -> tuple[float, bool]:
        move  = self.MOVE_SPACE[move_id]
        label = self.LABEL_SPACE[label_id]

        if self.is_done():
            return 0.0, True

        act     = self.current_activity()
        enabled = self.real_enabled_visible()
        base_reward = 0.0

        if move == "L":
            base_reward = -1.0
            self.pos += 1

        elif move in ("S", "M"):
            fired = False
            for t in enabled:
                if t.label == label:
                    # may not be deterministic if multiple transitions share the same label => to dooo. 
                    self.marking = self.sem.weak_execute(t, self.net, self.marking)
                    fired = True
                    break     
            # print(f"Current activity: {self.current_activity()}")

            if move == "S":
                base_reward = +1.0
                self.pos += 1
            elif move == "M":
                base_reward = -1.0
                self._inserted_model_moves.append(label)

        else:
            base_reward = -1.0
        total_reward = base_reward 

        return total_reward, self.is_done()

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
            t.label for t in self.sem.enabled_transitions(self.net, self.marking)
            if t.label is not None
        ]
    
    def real_enabled_visible(self):
        return [
            t for t in self.sem.enabled_transitions(self.net, self.marking)
            if t.label is not None
        ]

# to dooo => tau function
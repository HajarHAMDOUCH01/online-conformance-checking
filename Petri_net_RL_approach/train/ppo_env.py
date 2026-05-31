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
        self.LABEL_SPACE = list(labels)
        self.MOVE_ID_SPACE = {move: i for i, move in enumerate(self.MOVE_SPACE)}
        self.ID_LABEL_SPACE = {i: label for i, label in enumerate(self.LABEL_SPACE)}
        self.LABEL_ID_SPACE = {label: i for i, label in enumerate(self.LABEL_SPACE)}
        self.ID_MOVE_SPACE = {i: move for i, move in enumerate(self.MOVE_SPACE)}
        

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

    def infere_move_type(self, label, position):
        enabled = self._enabled_visible()
        label_str = self.LABEL_SPACE[label]
        act = self.current_activity()  # prefix[self.pos]
        next_act = self.prefix[self.pos + 1] if (self.pos + 1) < len(self.prefix) else act

        # check current pos first, then next
        if label_str == act and label_str in enabled:
            return 0  # S
        elif label_str == act and label_str not in enabled:
            return 2  # L
        else:
            return 1  # M : label in enabled but != act

    def valid_label_mask(self) -> torch.Tensor:
        enabled = self._enabled_visible()
        act = self.current_activity()
        mask = torch.tensor(
            [label in enabled or label == act for label in self.LABEL_SPACE],
            dtype=torch.bool
        )
        return mask
    
    def step(self, model, move_id: int, label_id: int, prev_moves, prev_labels, labels_logits, attn_weights, moves_for_all_labels):
        """
        before reward design : 
            1. checks prefix 
            2. applies this masks
                L => label = act 
                M => label = any(enabled)
                S => label = any(enabled == act)
            3. one reward per move type (S -> +1 ; M or L -> -1)
            4. PPO learns reward of the whole path and minimizes the cost of the alignment
        efter reward design :
            1. predict a label randomly 
            2. infere its move type based on the generated alignement 
            3. depending on predicted activity move type :
                reaward = attendance_to_the_prefix + exploration_rate + alpha / beta / gamma * label's logit
                * attendance_to_the_prefix applies attention mechanism to the priginal prefix up to current pos 
                * exploration rate uses M labels logits 
        """
        move  = self.MOVE_SPACE[move_id]
        label = self.LABEL_SPACE[label_id]

        if self.is_done():
            return label, move, 0.0, True
        act     = self.current_activity()
        enabled = self.real_enabled_visible()
        enabled_str = self._enabled_visible()
        total_reward = 0.0 
        if move == "S":
            self.pos += 1   
        if move == "L":
            self.pos += 1
        if move in ("S", "M"):

            fired = False
            for t in enabled:
                if t.label == label:
                    # to do : fix : may not be deterministic if multiple transitions share the same label => to dooo. 
                    self.marking = self.sem.weak_execute(t, self.net, self.marking)
                    fired = True
                    break 
        # new state :
            # state includes : prefix and its current position (note we can do marking for prefix too , but it's not used in this environement modelization)
            # state includes : generated alignment and its marking 
        prev_moves.append(move)
        prev_labels.append(label)
        new_moves  = prev_moves
        new_labels = prev_labels
        total_reward = self.reward_function(new_moves, new_labels, labels_logits, attn_weights, moves_for_all_labels)
        return new_labels[-1], new_moves[-1], total_reward, self.is_done()

    def reward_function(self, new_moves, new_labels, labels_logits, attention_weights_to_prefix, moves_for_all_labels):
        alpha = 0.3
        beta  = 0.3
        gamma = 0.3
        move  = new_moves[-1]
        label = new_labels[-1]

        label_id    = self.LABEL_SPACE.index(label)
        label_logit = torch.sigmoid(labels_logits[label_id]).item()

        model_moves_labels_logits = [
            labels_logits[i] for i, mt in enumerate(moves_for_all_labels) if mt == 2
        ]
        ratio_of_high_model_moves_labels_logits = (
            len([l for l in model_moves_labels_logits if l.item() > 0.5]) / len(labels_logits)
        ) if len(labels_logits) > 0 else 0.0

        did_you_attend = alpha * attention_weights_to_prefix.max().item()
        did_you_explore = beta  * ratio_of_high_model_moves_labels_logits
        label_reward    = gamma * label_logit

        reward = did_you_attend + did_you_explore + label_reward
        return reward
    
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

# to do => tau function for handling silent transitions
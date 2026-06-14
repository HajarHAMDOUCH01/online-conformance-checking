from __future__ import annotations

import itertools
from collections import defaultdict, deque

import torch
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.semantics import ClassicSemantics

LOOKAHEAD_LAMBDA = 0.5
_dijkstra_counter = itertools.count()

# helper used by both env and any external callers
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

    Silent-transitions : 
    ------------------------
    Three class-level caches persist across all episodes and all env
    instances that share the same Petri net.  
    They are keyed by marking tuple; which depends only on net structure.

    _shared_closure_cache : marking_tup -> frozenset[marking_tup]
        Every marking reachable from a given marking via silent
        transitions

    _shared_labels_cache  : marking_tup -> frozenset[str]
        Every visible label enabled in any reachable marking.

    _shared_path_cache    : (marking_tup, label_str) -> (path, t) | (None, None)
        The shortest silent-transition path (BFS order) that makes
        label_str fireable, together with the matching visible transition.
        Early-terminates as soon as the label is found
    """
    _shared_closure_cache: dict = {}   # marking_tup  -> frozenset[marking_tup]
    _shared_labels_cache:  dict = {}   # marking_tup  -> frozenset[str]
    _shared_path_cache:    dict = {}   # (m_tup, lbl) -> (path, t) | (None, None)

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(self, net, im, labels: list[str]):
        self.net = net
        self.im  = im
        self.sem = ClassicSemantics()

        self.MOVE_SPACE  = ["S", "M", "L"]
        self.LABEL_SPACE = list(labels)
        self.MOVE_ID_SPACE  = {move: i  for i, move  in enumerate(self.MOVE_SPACE)}
        self.ID_LABEL_SPACE = {i: label for i, label in enumerate(self.LABEL_SPACE)}
        self.LABEL_ID_SPACE = {label: i for i, label in enumerate(self.LABEL_SPACE)}
        self.ID_MOVE_SPACE  = {i: move  for i, move  in enumerate(self.MOVE_SPACE)}

        self.place_list = sorted(net.places, key=lambda p: p.name)
        self.place_idx  = {p: i for i, p in enumerate(self.place_list)}
        self.n_places   = len(self.place_list)

        self._in_arcs:  dict = defaultdict(list)   # transition -> [(place_name, weight)]
        self._out_arcs: dict = defaultdict(list)   # transition -> [(place_name, weight)]
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
        self.marking = Marking({p: v for p, v in self.im.items()})
        self._inserted_model_moves = []
        # Shared caches are NOT cleared: they depend only on the net,
        # which never changes between episodes.
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

    # =========================================================================
    # Low-level Petri-net helpers
    # =========================================================================

    def _is_enabled(self, m_dict: dict, transition) -> bool:
        """True iff all input-arc token requirements of transition are satisfied."""
        return all(
            m_dict.get(pname, 0) >= w
            for pname, w in self._in_arcs[transition]
        )

    def _fire(self, m_dict: dict, transition) -> dict:
        """
        Return a new marking dict after firing transition.
        Does not mutate the input dict.
        """
        result = dict(m_dict)
        for pname, w in self._in_arcs[transition]:
            result[pname] = result.get(pname, 0) - w
            if result[pname] == 0:
                del result[pname]
        for pname, w in self._out_arcs[transition]:
            result[pname] = result.get(pname, 0) + w
        return result

    def _marking_to_dict(self) -> dict:
        """Convert self.marking (pm4py Marking, Place-keyed) to a str-keyed dict."""
        return {p.name: v for p, v in self.marking.items() if v > 0}

    def _pm4py_marking_to_name_dict(self, marking) -> dict:
        return {p.name: cnt for p, cnt in marking.items() if cnt > 0}

    def _enabled_visible(self) -> set:
        """Labels of all currently enabled visible transitions."""
        return {
            t.label
            for t in self.sem.enabled_transitions(self.net, self.marking)
            if t.label is not None
        }

    def real_enabled_visible(self) -> list:
        """All currently enabled visible transition *objects*."""
        return [
            t for t in self.sem.enabled_transitions(self.net, self.marking)
            if t.label is not None
        ]

    # =========================================================================
    # Silent-closure subsystem
    # =========================================================================

    def _silent_reachable(self, m_dict: dict) -> frozenset:
        """
        Return a frozenset of every marking-tuple reachable from m_dict by
        firing zero or more silent transitions (m_dict itself is included).

        Stored in _shared_closure_cache.  At most one BFS per unique marking
        for the lifetime of the process.
        """
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
        """
        Return frozenset of every visible label enabled in any marking
        reachable from m_dict via silent transitions (m_dict itself included).

        Stored in _shared_labels_cache.  Built on top of _silent_reachable,
        so both caches benefit from each other.

        Accepts both Place-keyed (pm4py Marking) and str-keyed dicts.
        """
        # Normalise to str-keyed dict
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
        """
        Targeted BFS over silent transitions.

        Terminates as soon as target_label becomes fireable

        Returns
        -------
        (silent_path, matching_transition)
            silent_path is the ordered list of silent transitions to fire
            before matching_transition.  An empty list means target_label
            is already enabled at m_dict with no silent preamble needed.
        (None, None)
            target_label is unreachable via any sequence of silent moves.

        Stored in _shared_path_cache keyed by (marking_tup, target_label).
        """
        key    = (_m_tuple(m_dict), target_label)
        cached = AlignmentEnv._shared_path_cache.get(key)
        if cached is not None:
            return cached

        visited = {_m_tuple(m_dict)}
        queue   = deque([(m_dict, [])])   # (current_marking_dict, path_so_far)

        while queue:
            curr, path = queue.popleft()

            # Check if target_label is fireable at this intermediate marking
            for t in self._label_to_trans.get(target_label, []):
                if self._is_enabled(curr, t):
                    result = (path, t)
                    AlignmentEnv._shared_path_cache[key] = result
                    return result

            # Expand via silent transitions
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
        """
        Fire a sequence of silent transitions from m_dict and return the
        resulting marking dict.  Does not touch self.marking.
        """
        result = dict(m_dict)
        for tau in silent_path:
            result = self._fire(result, tau)
        return result

    # =========================================================================
    # Decision interface
    # =========================================================================

    def infere_move_type(self, label) -> int:
        """
        Infer move type for label given the current marking and prefix position.

          0  S  synchronous  — label == current activity AND label is enabled
          1  M  model-only   — label != current activity AND label is enabled
          2  L  log-only     — label == current AND label is not enabled 

        Uses a single cached _labels_enabled_after_silent call.
        """
        label_str   = self.LABEL_SPACE[label] if isinstance(label, int) else label
        act         = self.current_activity()
        m_dict      = self._marking_to_dict()
        all_enabled = self._labels_enabled_after_silent(m_dict)

        if label_str == act:
            return 0 if label_str in all_enabled else 2 # either S if enabled or L 
        else: 
            if label_str in all_enabled:
                return 1
            else: 
                return 3
        # not enabled and not the current prefix position are masked

    def valid_label_mask(self) -> torch.Tensor:
        """
        Boolean mask over LABEL_SPACE.  A label passes if:

            it equals the current prefix activity — always allowed so the
              agent can always produce a log-only or synchronous move, OR
            it can be fired as a non-cyclic, non-sink move from some
              marking in the silent reachability set of the current marking.

        Single _silent_reachable call (cached).  Inner loop short-circuits
        per label as soon as one valid firing is confirmed.
        """
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
        # Fallback: if everything is masked out, open all labels so the agent
        # always has at least one valid action
        if not result.any():
            # Forcer log-only : seul le current activity est autorisé
            # Si act n'est pas dans LABEL_SPACE, autoriser tous les labels
            # mais marquer comme L obligatoire
            if act in self.LABEL_SPACE:
                result = torch.zeros(len(self.LABEL_SPACE), dtype=torch.bool)
                result[self.LABEL_SPACE.index(act)] = True
            else:
                # Label inconnu → log-only forcé, on garde le fallback
                result = torch.ones(len(self.LABEL_SPACE), dtype=torch.bool)
        return result

    # =========================================================================
    # Step
    # =========================================================================

    def step(
        self,
        model,
        move_id:              int,
        label_id:             int,
        prev_moves:           list,
        prev_labels:          list,
        labels_logits:        torch.Tensor,
        attn_weights:         torch.Tensor,
        moves_for_all_labels: list,
        done=False,
    ):
        """
        Execute one alignment step.

        Firing strategy
        ---------------
        0. The decoder predicts a label
        1. Try direct firing (no silent preamble) via real_enabled_visible().
           If multiple transitions share the same label only the first valid
           one (non-cycle, non-sink) is used.
        2. If that fails, call _silent_path_to for a targeted BFS that stops
           as soon as the label becomes fireable.  Replay the silent path,
           then fire the visible transition.

        Both stages skip a firing if it would:
          • produce the same marking as the current one (cycle)
          • deposit a token in sink0 during an M-move
        """
        move  = self.MOVE_SPACE[move_id]
        label = self.LABEL_SPACE[label_id]

        if self.is_done() or done == True:
            return label, move, 0.0, True

        if move == "L":
            self.pos += 1

        if move in ("S", "M"):
            current_m   = self._marking_to_dict()
            current_tup = _m_tuple(current_m)
            fired       = False
            print("label = ", label)
            # ------------------------------------------------------------------
            # Stage 1: direct firing
            # ------------------------------------------------------------------
            for t in self.real_enabled_visible():
                
                if t.label != label:
                    continue
                new_m   = self._fire(current_m, t)
                new_tup = _m_tuple(new_m)
                if new_tup == current_tup:
                    continue            
                if self._sink_place_name in new_m and move == "M":
                    continue            
                self.marking = self.sem.weak_execute(t, self.net, self.marking)
                fired = True
                break
            if not fired:
                silent_path, matching_t = self._silent_path_to(current_m, label)
                print("matching t = ", matching_t)
                
                if silent_path is not None and matching_t is not None:
                    target_m = self._replay_silent_path(current_m, silent_path)
                    new_m    = self._fire(target_m, matching_t)
                    new_tup  = _m_tuple(new_m)

                    valid = (new_tup != current_tup) and (
                        self._sink_place_name not in new_m or move != "M"
                    )
                    if valid:
                        for tau in silent_path:
                            self.marking = self.sem.weak_execute(
                                tau, self.net, self.marking
                            )
                        self.marking = self.sem.weak_execute(
                            matching_t, self.net, self.marking
                        )
                        # print("in marking : ", current_m)
                        
                        fired = True

            if move == "S":
                print("pos+1")
                self.pos += 1

        prev_moves.append(move)
        prev_labels.append(label)
        total_reward = self.reward_function(
            prev_moves, prev_labels, labels_logits, attn_weights,
            moves_for_all_labels,
        )
        return prev_labels[-1], prev_moves[-1], total_reward, self.is_done()

    # =========================================================================
    # Reward
    # =========================================================================
    def reward_function(self, new_moves, new_labels, labels_logits,
                        attention_weights_to_prefix, moves_for_all_labels):
        
        label = new_labels[-1]
        move  = new_moves[-1]
        step  = len(new_moves)
        reward = 0.0

        # # ── Récompense par type de move ──────────────────────────────────
        if move == "S":
            reward += 3.0                    # sync = parfait
            # bonus si trouvé rapidement
            reward += max(0, 5 - 0.5 * step)

        elif move == "L":
            reward += 1.0                    # log-only = acceptable

        m_streak = 0
        for m in reversed(new_moves):
            if m == "M":
                m_streak += 1
            else:
                break
        reward -= 0.3 * (m_streak ** 2)

        # # ── Récompense récupération après M-moves ─────────────────────────
        if len(new_moves) >= 2 and new_moves[-2] == "M":
            if move == "S":
                reward += 2.0               # bien récupéré
            elif move == "L":
                reward += 0.5
            elif move == "M":
                reward -= 0.3               # continue à errer

        # ── Confidence du logit ───────────────────────────────────────────
        label_id    = self.LABEL_SPACE.index(label)
        label_logit = torch.sigmoid(labels_logits[label_id]).item()
        reward += 0.1 * label_logit

        # ── Attention sur le préfixe ──────────────────────────────────────
        # reward += 0.3 * attention_weights_to_prefix.max().item()
        attended_pos = attention_weights_to_prefix.argmax().item()

        distance = abs(attended_pos - self.pos)

        reward += 1.5 * (1.0 / (1.0 + distance))
        return reward

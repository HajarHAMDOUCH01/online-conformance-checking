import os
import sys
import heapq
import itertools
import time
from collections import defaultdict

import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.importer.xes import importer as xes_importer

# ── File paths ────────────────────────────────────────────────────────────────
DS_CSV   = (
    r"C:\Users\LENONVO\OneDrive\Desktop\STAGE-PFE-CRAN\datasets\STAGE_data"
    r"\data_event_log\data\data\heuristics_miner"
    r"\heuristics_dedup_1min_dep080_and080_loop090_prefix_alignments_dataset.csv"
)
PNML_PATH = (
    r"C:\Users\LENONVO\OneDrive\Desktop\STAGE-PFE-CRAN\datasets\STAGE_data"
    r"\data_event_log\data\dicovery_models_imgs\heuristics_miner"
    r"\heuristics_dedup_1min_dep080_and080_loop090.pnml"
)
XES_PATH = (
    r"C:\Users\LENONVO\OneDrive\Desktop\STAGE-PFE-CRAN\datasets\STAGE_data"
    r"\data_event_log\data\data"
    r"\ordered_cleaned_event_log_normalized_with_conformance.xes"
)

# ── Move costs  ──────────────────────
COST_SYNC  = 0   # synchronous move  - log and model agree
COST_LOG   = 1   # log-only move     - event in log, not in model
COST_MODEL = 1   # model-only move   - transition fired, no log event
COST_SILENT = 0  # silent (tau) transition - always free

# =============================================================================
# Load Petri net
# =============================================================================
print("=" * 70)
print("Loading Petri net")
print("=" * 70)

net, im, fm = pm4py.read_pnml(PNML_PATH)
print("Initial marking:", im)

sink_place = next(p for p in net.places if p.name == "sink0")
fm = pm4py.generate_marking(net, {sink_place: 1})
print("Final marking  :", fm)

print(f"Places           : {len(net.places)}")
print(f"Transitions      : {len(net.transitions)}")
print(f"Unique labels    : {len({t.label for t in net.transitions if t.label})}")
print(f"Silent trans.    : {sum(1 for t in net.transitions if t.label is None)}")

# =============================================================================
# Pre-compute arc look-up tables (immutable after this point)
# =============================================================================
# _in_arcs[transition]  -> [(place_name, weight), ...]   (input arcs)
# _out_arcs[transition] -> [(place_name, weight), ...]   (output arcs)
_in_arcs:  dict = defaultdict(list)
_out_arcs: dict = defaultdict(list)

for arc in net.arcs:
    if isinstance(arc.source, PetriNet.Place):
        _in_arcs[arc.target].append((arc.source.name, arc.weight))
    else:
        _out_arcs[arc.source].append((arc.target.name, arc.weight))

_visible_transitions = [t for t in net.transitions if t.label is not None]
_silent_transitions  = [t for t in net.transitions if t.label is None]
_label_to_trans: dict = defaultdict(list)
for t in _visible_transitions:
    _label_to_trans[t.label].append(t)

# Set of all activity labels modelled in the net (used by heuristic)
_modelled_labels: frozenset = frozenset(_label_to_trans.keys())

# =============================================================================
# Petri-net firing helpers
# =============================================================================

def _marking_to_tuple(marking) -> tuple:
    """Convert a pm4py Marking object -> hashable sorted tuple."""
    return tuple(
        (p.name, cnt)
        for p, cnt in sorted(marking.items(), key=lambda x: x[0].name)
        if cnt > 0
    )

_IM_TUPLE: tuple = _marking_to_tuple(im)


def _is_enabled(m_dict: dict, transition) -> bool:
    """True iff all input-arc requirements are satisfied."""
    return all(m_dict.get(pname, 0) >= w for pname, w in _in_arcs[transition])


def _fire(m_dict: dict, transition) -> dict:
    """Return a new marking dict after firing transition"""
    result = dict(m_dict)
    for pname, w in _in_arcs[transition]:
        result[pname] = result.get(pname, 0) - w
        if result[pname] == 0:
            del result[pname]
    for pname, w in _out_arcs[transition]:
        result[pname] = result.get(pname, 0) + w
    return result


def _m_tuple(m_dict: dict) -> tuple:
    """Convert marking dict -> hashable tuple (inline, avoids repeated sorts)."""
    return tuple(sorted((p, c) for p, c in m_dict.items() if c > 0))


# =============================================================================
# Pre-compute silent-closure reachability (for the heuristic)
# =============================================================================
# For the admissible heuristic we want to know, from any marking, which labels
# could become enabled after zero or more silent transitions.
# Pre-computing the full reachability is too expensive.  Instead we pre-compute
# the "silent-closure" of each individual marking lazily (cached per marking).


from pm4py.objects.petri_net.semantics import ClassicSemantics
from pm4py.objects.petri_net.obj import Marking, PetriNet

def _enabled_visible(m_dict, net):
    marking = Marking({p: v for p, v in m_dict.items()})
    sem = ClassicSemantics()
    return [
        t.label for t in sem.enabled_transitions(net, marking)
        if t.label is not None
    ]

_silent_closure_cache: dict = {}


def _silent_closure(m_dict: dict) -> frozenset:
    """
    Return the set of all markings (as tuples) reachable from m_dict by firing
    only silent transitions, including m_dict itself.  Cached by marking tuple.
    """
    key = _m_tuple(m_dict)
    if key in _silent_closure_cache:
        return _silent_closure_cache[key]

    visited  = {key}
    frontier = [m_dict]
    while frontier:
        curr = frontier.pop()
        for t in _silent_transitions:
            if _is_enabled(curr, t):
                nm   = _fire(curr, t)
                nkey = _m_tuple(nm)
                if nkey not in visited:
                    visited.add(nkey)
                    frontier.append(nm)

    result = frozenset(visited)
    _silent_closure_cache[key] = result
    return result


def _labels_enabled_after_silent(m_dict: dict) -> frozenset:
    """
    Return the set of visible labels that have at least one enabled transition
    in any marking reachable via silent moves from m_dict.
    """
    enabled = set()
    for m_tup in _silent_closure(m_dict):
        m_tmp = dict(m_tup)
        for lbl, trans_list in _label_to_trans.items():
            if lbl not in enabled:
                for t in trans_list:
                    if _is_enabled(m_tmp, t):
                        enabled.add(lbl)
                        break
    return frozenset(enabled)


# =============================================================================
# heuristic 
# =============================================================================

def _heuristic(m_dict: dict, pos: int, prefix: list) -> int:
    """
    Admissible lower bound on future alignment cost for a prefix alignment.

    For each remaining activity in prefix[pos:], if it cannot be executed
    as a synchronous move from any marking reachable via silent transitions
    from m_dict, it will cost at least 1 (either a log-move, or a model-move
    that eventually enables a sync at extra cost).
    Counting these activities gives h ≤ true remaining cost -> admissible.

    Complexity: O(|prefix[pos:]| * |silent_closure|) per call; but silent
    closure is cached, so repeated calls with the same marking are cheaper.
    """
    if pos >= len(prefix):
        return 0
    enabled = _labels_enabled_after_silent(m_dict)
    # Count activities in the remaining prefix that are definitely not sync-able
    h = sum(1 for act in prefix[pos:] if act not in enabled)
    return h


# =============================================================================
# A* prefix-alignment algorithm
# =============================================================================

def _astar_prefix_alignment(prefix: list) -> tuple[list, int]:
    """
    Compute a cost-optimal prefix alignment using A*.

    State space
    -----------
    A search state is (marking_tuple, pos) where:
        marking_tuple : current Petri-net marking
        pos           : number of log activities consumed so far  (0 … len(prefix))

    Goal
    ----
    pos == len(prefix)  (all log activities have been aligned)

    Moves
    --------------------------------
    ('S',   label)  synchronous move   - cost 0 (advance pos, change marking)
    ('tau', None)   silent transition  - cost 0 (pos unchanged, change marking)
    ('L',   label)  log-only move      - cost 1 (advance pos, marking unchanged)
    ('M',   label)  model-only move    - cost 1 (fire visible t, pos unchanged)

    Priority queue entry
    --------------------
    (f, tie_break, g, m_tup, pos)
    where f = g + h  (total estimated cost).

    Closed set
    ----------
    Essential for A* correctness: once a state is popped with its optimal g-value
    it is marked closed and never re-expanded.  (Pure Dijkstra omits this and
    instead uses a lazy check `cost > dist[state]`, which is equivalent but
    allocates more heap entries; A* with a closed set is cleaner and faster when
    the heuristic is consistent / monotone, which the used heuristic here is.)

    Consistency of h
    ----------------
    h is consistent (monotone) if  h(n) ≤ c(n, n') + h(n')  for every edge n -> n'
      * Sync / silent moves: pos advances or marking changes with cost 0.
        h can only decrease or stay equal
      * Log-move: pos advances by 1, cost 1.  h decreases by at most 1 
      * Model-move: marking changes, pos unchanged, cost 1.
        h(new_m, pos) ≤ h(old_m, pos) + 1 because enabling more transitions
        can only reduce the count of "unmatchable" activities 
    Consistent heuristics guarantee A* never re-expands a closed node.

    Returns
    -------
    (path, total_cost) where path is a list of (move_type, label) pairs.
    """
    visited_states = 0
    queued_states = 1
    traversed_arcs = 0
    goal_pos = len(prefix)

    # g[state] = best known cost to reach state
    g: dict = {}
    parent:  dict = {}
    dict_for_mem_metric: dict = {}

    init_state = (_IM_TUPLE, 0)
    g[init_state]      = 0
    parent[init_state] = None

    closed: set = set()
    counter     = itertools.count()

    # Compute initial f = g + h
    init_m_dict = dict(_IM_TUPLE)
    h0          = _heuristic(init_m_dict, 0, prefix)
    heapq.heappush(heap := [], (h0, next(counter), 0, _IM_TUPLE, 0))

    while heap:
        f, _, g_curr, m_tup, pos = heapq.heappop(heap)

        state = (m_tup, pos)

        # ── Skip if already closed ──────────────────────
        if state in closed:
            continue

        # ── Optimality check (should match because h is consistent) ─────────
        if g_curr > g.get(state, float("inf")):
            continue

        closed.add(state)
        visited_states += 1
        # ── Goal test ───────────────────────────────────────────────────────
        if pos == goal_pos:
            # Reconstruct the path
            path, cur = [], state
            while parent[cur] is not None:
                par, mv = parent[cur]
                path.append(mv)
                cur = par
            path.reverse()
            dict_for_mem_metric["visited_states"]  =visited_states
            dict_for_mem_metric["queued_states"]  =queued_states
            dict_for_mem_metric["traversed_arcs"]  =traversed_arcs

            return path, g_curr, dict_for_mem_metric

        # ── Expand state ────────────────────────────────────────────────────
        m_dict = dict(m_tup)
        act    = prefix[pos] if pos < goal_pos else None

        def _push(new_m_dict: dict, new_pos: int, move_cost: int,
                  move: tuple, _s=state, _g=g_curr):
            """Helper: compute f = g + h and push if better."""
            nonlocal queued_states
            nonlocal traversed_arcs
            traversed_arcs += 1
            new_g   = _g + move_cost
            new_tup = _m_tuple(new_m_dict)
            ns      = (new_tup, new_pos)
            if ns in closed:
                return
            if new_g < g.get(ns, float("inf")):
                g[ns]      = new_g
                parent[ns] = (_s, move)
                h_val      = _heuristic(new_m_dict, new_pos, prefix)
                f_val      = new_g + h_val
                heapq.heappush(heap, (f_val, next(counter), new_g, new_tup, new_pos))
                queued_states += 1
        # 1. Silent transitions (tau): cost 0, pos unchanged
        for t in _silent_transitions:
            if _is_enabled(m_dict, t):
                _push(_fire(m_dict, t), pos, COST_SILENT, ("tau", None))

        # 2. Synchronous moves: cost 0, pos advances
        if act is not None:
            for t in _label_to_trans.get(act, []):
                if _is_enabled(m_dict, t):
                    _push(_fire(m_dict, t), pos + 1, COST_SYNC, ("S", act))

        # 3. Model-only moves: cost 1, pos unchanged (fire visible t)
        for t in _visible_transitions:
            if _is_enabled(m_dict, t):
                _push(_fire(m_dict, t), pos, COST_MODEL, ("M", t.label))

        # 4. Log-only move: cost 1, pos advances, marking unchanged
        if act is not None:
            _push(m_dict, pos + 1, COST_LOG, ("L", act))

    return [], float("inf"), dict_for_mem_metric


# =============================================================================
# align_prefix
# =============================================================================

def align_prefix(activities: list) -> dict:
    """
    Compute the optimal prefix alignment for the given activity list.

    Parameters
    ----------
    activities : list of str
        Ordered sequence of observed activity labels.

    Returns
    -------
    dict with keys:
        aligned_prefix  – list of matched labels (sync + model moves)
        step_types      – list of move types ('S', 'L', 'M') - tau excluded
        cost            – total alignment cost (int)
        sync_moves      – count of synchronous moves
        log_moves       – count of log-only moves
        model_moves     – count of model-only moves
        is_conforming   – True iff cost == 0 (no deviations)
        error           – None or error string
    """
    clean = [str(a) for a in activities]

    try:
        start = time.perf_counter()
        raw_alignment, total_cost, dict_for_mem_metric = _astar_prefix_alignment(clean)
        alignment_time = time.perf_counter() - start
    except Exception as exc:
        return {"error": str(exc)}

    visited_states = dict_for_mem_metric["visited_states"]
    traversed_arcs = dict_for_mem_metric["traversed_arcs"]
    queued_states = dict_for_mem_metric["queued_states"]

    if total_cost == float("inf"):
        return {"error": "no_path_found"}

    aligned:    list = []
    step_types: list = []

    for mv_type, label in raw_alignment:
        if mv_type == "tau":          # silent moves are internal - omit
            continue
        aligned.append(label)
        step_types.append(mv_type)

    total_sync  = step_types.count("S")
    total_log   = step_types.count("L")
    total_model = step_types.count("M")

    return dict(
        queued_states = queued_states,
        traversed_arcs = traversed_arcs,
        visited_states = visited_states,
        alignment_time_sec = alignment_time,
        aligned_prefix = aligned,
        step_types     = step_types,
        cost           = total_cost,
        sync_moves     = total_sync,
        log_moves      = total_log,
        model_moves    = total_model,
        is_conforming  = (total_log == 0 and total_model == 0),
        error          = None,
    )
    

# =============================================================================
# Load XES log, filter low-fitness traces
# =============================================================================
print("\n" + "=" * 70)
print("Loading XES event log")
print("=" * 70)

log_all = xes_importer.apply(XES_PATH)

neg_traces = [
    trace for trace in log_all
    if float(trace.attributes.get("trace_fitness", 1.0)) < 0.7
]

print(f"Total traces              : {len(log_all)}")
print(f"Low-fitness traces (<0.7) : {len(neg_traces)}")

# =============================================================================
# Dataset generation loop
# =============================================================================
print("\n" + "=" * 70)
print("Dataset generation (A* prefix alignments)")
print("=" * 70)

os.makedirs(os.path.dirname(DS_CSV), exist_ok=True)
write_header   = not os.path.exists(DS_CSV)
t_start        = time.time()
total_prefixes = 0
errors         = 0

for idx, trace in enumerate(neg_traces):
    case_id    = str(trace.attributes.get("concept:name", f"case_{idx}"))
    activities = [str(ev["concept:name"]) for ev in trace]
    rows       = []

    for k in range(1, len(activities) + 1):

        result = align_prefix(activities[:k])

        if result.get("error"):
            errors += 1
        


        rows.append({
            "queued_states" : result.get("queued_states"),
            "traversed_arcs" : result.get("traversed_arcs"),
            "visited_states" : result.get("visited_states"),
            "alignment_time_sec": result.get("alignment_time_sec"),
            "case_id":           case_id,
            "prefix_length":     k,
            "prefix_activities": str(activities[:k]),
            "aligned_prefix":    str(result.get("aligned_prefix")),
            "step_types":        str(result.get("step_types")),
            "cost":              result.get("cost"),
            "sync_moves":        result.get("sync_moves"),
            "log_moves":         result.get("log_moves"),
            "model_moves":       result.get("model_moves"),
            "is_conforming":     result.get("is_conforming"),
            "error":             result.get("error"),

        })
        # print(rows[-1]["prefix_activities"])
        # print(rows[-1]["step_types"])

    pd.DataFrame(rows).to_csv(
        DS_CSV, mode="a", header=write_header, index=False
    )
    write_header    = False
    total_prefixes += len(rows)

    elapsed = time.time() - t_start
    avg_s   = elapsed / (idx + 1)
    eta_s   = avg_s * (len(neg_traces) - idx - 1)

    if (idx + 1) % 50 == 0 or idx == 0:
        print(
            f"  [{idx+1:4d}/{len(neg_traces)}]  "
            f"case={case_id!r}  prefixes={len(rows)}  "
            f"elapsed={elapsed:.1f}s  ETA={eta_s:.0f}s  "
            f"errors={errors}"
        )

print(f"\nDone. Total prefixes written : {total_prefixes}")
print(f"      Errors                 : {errors}")
print(f"      Saved -> {DS_CSV}")
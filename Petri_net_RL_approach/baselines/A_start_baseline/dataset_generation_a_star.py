import os
import sys
import yaml
import heapq
import itertools
import time
from collections import defaultdict
from collections import defaultdict, deque

import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.importer.xes import importer as xes_importer

# ── Load config ───────────────────────────────────────────────────────────────
_CFG_PATH = "/content/online-conformance-checking/Petri_net_RL_approach/train/config.yaml"

with open(_CFG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

_paths = _cfg["paths"]

# ── Paths ─────────────────────────────────────────────────────────────────────
DS_CSV           = _paths["ds_csv"]
PNML_PATH        = _paths["pnml_path"]
XES_PATH         = _paths["xes_path"]



# ── Move costs  ──────────────────────
COST_SYNC  = 0   # synchronous move  - log and model agree
COST_LOG   = 2   # log-only move     - event in log, not in model
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

sink_place = next(p for p in net.places if p.name == "sink")
fm = pm4py.generate_marking(net, sink_place)
print("Final marking  :", fm)

print(f"Places           : {len(net.places)}")
print(f"Transitions      : {len(net.transitions)}")
print(f"Unique labels    : {len({t.label for t in net.transitions if t.label})}")
print(f"Silent trans.    : {sum(1 for t in net.transitions if t.label is None)}")
place_by_name = {p.name: p for p in net.places}
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
from pm4py.objects.petri_net.semantics import ClassicSemantics
sem = ClassicSemantics()
enabled = sem.enabled_transitions(net, im)



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
from pm4py.objects.petri_net.semantics import ClassicSemantics
from pm4py.objects.petri_net.obj import Marking, PetriNet


# =============================================================================
# Module-level caches (mirrors AlignmentEnv._shared_* but standalone)
# =============================================================================
_shared_closure_cache: dict = {}   # marking_tup  -> frozenset[marking_tup]
_shared_labels_cache:  dict = {}   # marking_tup  -> frozenset[str]
_shared_path_cache:    dict = {}   # (m_tup, lbl) -> (path, t) | (None, None)

# =============================================================================
# Silent-closure helpers (standalone, no class dependency)
# =============================================================================

def _silent_reachable(m_dict: dict) -> frozenset:
    """
    Return a frozenset of every marking-tuple reachable from m_dict by
    firing zero or more silent transitions (m_dict itself is included).
    Cached in _shared_closure_cache.
    """
    key    = _m_tuple(m_dict)
    cached = _shared_closure_cache.get(key)
    if cached is not None:
        return cached

    visited  = {key}
    frontier = [m_dict]
    while frontier:
        curr = frontier.pop()
        
        # tau = next(iter(_silent_transitions))
        # print(_in_arcs[tau])
        for t in _silent_transitions:
            if _is_enabled(curr, t):
                nm   = _fire(curr, t)
                nkey = _m_tuple(nm)
                if nkey not in visited:
                    visited.add(nkey)
                    frontier.append(nm)

    result = frozenset(visited)
    _shared_closure_cache[key] = result
    return result

def _labels_enabled_after_silent(m_dict: dict) -> frozenset:
    """
    Return frozenset of every visible label enabled in any marking
    reachable from m_dict via silent transitions (m_dict itself included).
    Cached in _shared_labels_cache.
    Accepts both Place-keyed (pm4py Marking) and str-keyed dicts.
    """
    # Normalise to str-keyed dict
    if m_dict and hasattr(next(iter(m_dict)), "name"):
        m_dict = {p.name: v for p, v in m_dict.items() if v > 0}

    key    = _m_tuple(m_dict)
    cached = _shared_labels_cache.get(key)
    if cached is not None:
        return cached

    reachable = _silent_reachable(m_dict)
    enabled   = set()
    for m_tup in reachable:
        m_tmp = dict(m_tup)
        for lbl, trans_list in _label_to_trans.items():
            if lbl not in enabled:
                for t in trans_list:
                    if _is_enabled(m_tmp, t):
                        enabled.add(lbl)
                        break

    result = frozenset(enabled)
    _shared_labels_cache[key] = result
    return result

def _silent_path_to(m_dict: dict, target_label: str):
    """
    Targeted BFS over silent transitions.
    Terminates as soon as target_label becomes fireable.

    Returns
    -------
    (silent_path, matching_transition)  or  (None, None)
    Cached in _shared_path_cache.
    """
    key    = (_m_tuple(m_dict), target_label)
    cached = _shared_path_cache.get(key)
    if cached is not None:
        return cached

    visited = {_m_tuple(m_dict)}
    queue   = deque([(m_dict, [])])

    while queue:
        curr, path = queue.popleft()

        for t in _label_to_trans.get(target_label, []):
            if _is_enabled(curr, t):
                result = (path, t)
                _shared_path_cache[key] = result
                return result

        for tau in _silent_transitions:
            req = _in_arcs[tau]
            ok = all(curr.get(p, 0) >= w for p, w in req)

            # if ok:
            #     print("ENABLED:", tau)
            if _is_enabled(curr, tau):
                nm  = _fire(curr, tau)
                nk  = _m_tuple(nm)
                if nk not in visited:
                    visited.add(nk)
                    queue.append((nm, path + [tau]))

    _shared_path_cache[key] = (None, None)
    return None, None

def _replay_silent_path(m_dict: dict, silent_path: list) -> dict:
    """Fire a sequence of silent transitions and return the resulting marking dict."""
    result = dict(m_dict)
    for tau in silent_path:
        result = _fire(result, tau)
    return result

# =============================================================================
# heuristic 
# =============================================================================


def _marking_to_dict(marking) -> dict:
        """Convert self.marking (pm4py Marking, Place-keyed) to a str-keyed dict."""
        return {p.name: v for p, v in marking.items() if v > 0}

def _enabled_visible(net, marking):
    sem = ClassicSemantics()
    # print("marking type:", type(marking))
    # print("marking:", marking)
    return {
        t
        for t in sem.enabled_transitions(net, marking)
        if t.label is not None
    }

def _dict_to_marking(m_dict):
    return Marking({
        place_by_name[name]: cnt
        for name, cnt in m_dict.items()
    })

def _heuristic(m_dict: dict, pos: int, prefix: list) -> int:
    """
    Admissible: count remaining prefix activities that are unreachable
    from m_dict via ANY number of tau steps.
    
    These can never become S moves regardless of usage of tau closure,
    so each costs at least 1 (L). This never overestimates because:
    - if act IS reachable via taus → true cost could be 0 (S) → h ignores it ✓
    - if act is NOT reachable via taus → true cost >= 1 → h counts 1 ✓
    
    Consistent because tau/S moves (cost 0) can only grow or maintain
    the silent-reachable set, never shrink it.
    """
    if pos >= len(prefix):
        return 0
    enabled = _labels_enabled_after_silent(m_dict)
    return sum(1 for act in prefix[pos:] if act not in enabled)


# =============================================================================
# A* prefix-alignment algorithm
# =============================================================================

def _astar_prefix_alignment(
    prefix,
    start_marking=None,
    start_pos=0
):
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

    if start_marking is None:
        start_marking = _IM_TUPLE


    if isinstance(start_marking, dict):
        start_marking = _m_tuple(start_marking)

    elif isinstance(start_marking, Marking):
        start_marking = _m_tuple(
            {p.name:v for p,v in start_marking.items() if v > 0}
        )
    init_state = (start_marking, start_pos)
    g[init_state] = 0
    parent[init_state] = None

    closed: set = set()
    counter     = itertools.count()

    h0 = _heuristic(
        dict(start_marking),
        start_pos,
        prefix
    )

    heapq.heappush(
        heap := [],
        (h0, next(counter), 0, start_marking, start_pos)
    )
    
    while heap:
        
        f, _, g_curr, m_tup, pos = heapq.heappop(heap)
        # print("popped state:", m_tup, pos)
        state = (m_tup, pos)
        m_dict = dict(m_tup)
        
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
        
        def _push(new_m_dict: dict, new_pos: int, move_cost: int,
                  move: tuple, _s=state, _g=g_curr):
            """Helper: compute f = g + h and push if better."""
            nonlocal queued_states
            nonlocal traversed_arcs
            traversed_arcs += 1
            new_g   = _g + move_cost

            try:
                new_tup = _m_tuple(new_m_dict)
            except Exception as exc:
                # print("exceptionnnnn in _m_tuple call : ", exc)
                return {"error": str(exc)}
            ns      = (new_tup, new_pos)
            if ns in closed:
                return
            if new_g < g.get(ns, float("inf")):
                g[ns]      = new_g
                parent[ns] = (_s, move)
                try:
                    h_val      = _heuristic(new_m_dict, new_pos, prefix)
                except Exception as exc:
                    print("exceptionnnnn in heuristic call : ", exc)
                f_val      = new_g + h_val

# prefix        : ['Blood_tests_biologie_delocalisee', 'Nettoyage_soins_nasopharynges', 'TDR']

# gt labels     : ['Blood_tests_biologie_delocalisee', 'Urine_bandelette', 'Nettoyage_soins_nasopharynges', 'TDR']

# gt move_types : ['S', 'M', 'S', 'S']

# generated labels : ['Blood_tests_biologie_delocalisee', 'TDR', 'Inhalation_bronchodilatateurs_aerosols', 'TDR', 'Urine_bandelette', 'Surveillance_prise_de_la_temperature', 'TDR', 'Blood_tests_biologie_delocalisee', 'Dispensation_instantane', 'Nettoyage_soins_nasopharynges', 'Surveillance_prise_de_la_temperature', 'Inhalation_bronchodilatateurs_aerosols', 'Surveillance_prise_de_la_temperature', 'Blood_tests_biologie_delocalisee', 'TDR']

# corresponding move types : ['S', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'S', 'M', 'M', 'M', 'M', 'S']
                try:
                    heapq.heappush(heap, (f_val, next(counter), new_g, new_tup, new_pos))
                    # print("pushed to heap: ", f_val, new_g)
                    queued_states += 1
                except Exception as exc:
                    print("exceptionnnnn in heappush call : ", exc)
        # ── Expand state ─────────────────────────────────────────────────────
        
        m_dict = dict(m_tup)
        act    = prefix[pos] if pos < goal_pos else None
        if act is not None:
            silent_path, matching_t = _silent_path_to(m_dict, act)
            
            if matching_t is not None:

                m_after_tau = _replay_silent_path(
                    m_dict,
                    silent_path
                )
                m_after_sync = _fire(
                    m_after_tau,
                    matching_t
                )
                _push(
                    m_after_sync,
                    pos + 1,
                    COST_SYNC,
                    ("S", act)
                )

        reachable_labels = _labels_enabled_after_silent(m_dict)
        allow_model_moves = pos > start_pos
        if allow_model_moves:
            for lbl in reachable_labels:

                silent_path, matching_t = _silent_path_to(
                    m_dict,
                    lbl
                )

                if matching_t is None:
                    continue

                m_after_tau = _replay_silent_path(
                    m_dict,
                    silent_path
                )

                m_after_model = _fire(
                    m_after_tau,
                    matching_t
                )

                _push(
                    m_after_model,
                    pos,
                    COST_MODEL,
                    ("M", lbl)
                )

        # 3. Log-only move
        if act is not None:
            _push(
                m_dict,
                pos + 1,
                COST_LOG,
                ("L", act)
            )

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
        # print("BEFOREEEEEE A* CALL")
        raw_alignment, total_cost, dict_for_mem_metric = _astar_prefix_alignment(clean)
        # print("AFTEEEEEEER A* CALL") 
        alignment_time = time.perf_counter() - start
    # except Exception as exc:
    #     print("exception : ", exc)
    #     return {"error": str(exc)}

    except Exception:
        import traceback
        traceback.print_exc()
        raise

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
traces_fitnes_list = []
for trace in log_all:
    a = float(trace.attributes.get("trace_fitness"))
    traces_fitnes_list.append(a)

threshold = 0.95
neg_traces = [
    trace for i, trace in enumerate(log_all)
    if 0.85 < traces_fitnes_list[i] < threshold
]
print(f"training set : {len(neg_traces)} traces in (0.85, {threshold}) out of {len(log_all)} total")

max_trace = max(neg_traces, key=len)

print(f"Total traces              : {len(log_all)}")
print(f"0.85 < Low-fitness traces < {threshold} : {len(neg_traces)}")
print(f"Maximum trace length      : {len(max_trace)}")
print(f"Longest trace             : {max_trace}")

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
        # for step_type in rows[-1]["step_types"]:
        #     if step_type == "M":

        #         print("original prefix : ", rows[-1]["prefix_activities"])
        #         print("aligned prefix : ", rows[-1]["aligned_prefix"])
        #         print("alignement move types : ", rows[-1]["step_types"])

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
import os, sys, heapq, itertools, time, yaml
from collections import defaultdict

import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.log.importer.xes import importer as xes_importer

# ── Load config ───────────────────────────────────────────────────────────────
_CFG_PATH = r"C:\Users\LENONVO\OneDrive\Desktop\model\Petri_net_RL_approach\train\config.yaml"

with open(_CFG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

_paths = _cfg["paths"]

# ── Paths ─────────────────────────────────────────────────────────────────────
DS_CSV           = _paths["ds_csv"]
PNML_PATH        = _paths["pnml_path"]
XES_PATH         = _paths["xes_path"]



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

sink_place = next(p for p in net.places if p.name == "sink")
fm = pm4py.generate_marking(net, sink_place)
print("Final marking  :", fm)

print(f"Places           : {len(net.places)}")
print(f"Transitions      : {len(net.transitions)}")
print(f"Unique labels    : {len({t.label for t in net.transitions if t.label})}")
print(f"Silent trans.    : {sum(1 for t in net.transitions if t.label is None)}")
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer

# gviz = pn_visualizer.apply(net, im, fm)
# pn_visualizer.view(gviz)

log_all = xes_importer.apply(XES_PATH)

neg_traces = [
    trace for trace in log_all
    if float(trace.attributes.get("trace_fitness", 1.0)) < 0.7
]


_in_arcs  = defaultdict(list)
_out_arcs = defaultdict(list)
for arc in net.arcs:
    if isinstance(arc.source, PetriNet.Place):
        _in_arcs[arc.target].append((arc.source.name, arc.weight))
    else:
        _out_arcs[arc.source].append((arc.target.name, arc.weight))

_visible_transitions = [t for t in net.transitions if t.label is not None]
_silent_transitions  = [t for t in net.transitions if t.label is None]
_label_to_trans      = defaultdict(list)
for t in _visible_transitions:
    _label_to_trans[t.label].append(t)

def _marking_to_tuple(marking) -> tuple:
    return tuple(
        (p.name, cnt)
        for p, cnt in sorted(marking.items(), key=lambda x: x[0].name)
        if cnt > 0
    )

_IM_TUPLE = _marking_to_tuple(im)

def _tuple_to_dict(m_tup: tuple) -> dict:
    return dict(m_tup)

def _is_enabled(m_dict: dict, transition) -> bool:
    return all(m_dict.get(pname, 0) >= w for pname, w in _in_arcs[transition])

def _fire(m_dict: dict, transition) -> dict:
    result = dict(m_dict)
    for pname, w in _in_arcs[transition]:
        result[pname] = result.get(pname, 0) - w
        if result[pname] == 0:
            del result[pname]
    for pname, w in _out_arcs[transition]:
        result[pname] = result.get(pname, 0) + w
    return result

def _dijkstra_prefix_alignment(prefix: list) -> tuple:
    """Fresh alignment from scratch for the full prefix."""
    goal_pos   = len(prefix)
    dist       = {}
    parent     = {}
    init_state = (_IM_TUPLE, 0)          # always starting from initial marking

    dist[init_state]   = 0
    parent[init_state] = None
    _counter           = itertools.count()

    heap = [(0, next(_counter), _IM_TUPLE, 0)]

    while heap:
        cost, _, m_tup, pos = heapq.heappop(heap)
        state = (m_tup, pos)

        if cost > dist.get(state, float('inf')):
            continue

        if pos == goal_pos:
            path, cur = [], state
            while parent[cur] is not None:
                par, mv = parent[cur]
                path.append(mv)
                cur = par
            path.reverse()
            return path, cost

        act    = prefix[pos] if pos < goal_pos else None
        m_dict = _tuple_to_dict(m_tup)

        def _push(new_m_dict, new_pos, new_cost, move, _s=state):
            new_tup = tuple(sorted(
                (p, c) for p, c in new_m_dict.items() if c > 0
            ))
            ns = (new_tup, new_pos)
            if new_cost <= dist.get(ns, float('inf')):
                dist[ns]   = new_cost
                parent[ns] = (_s, move)
                heapq.heappush(heap, (new_cost, next(_counter), new_tup, new_pos))

        for t in _silent_transitions:
            if _is_enabled(m_dict, t):
                _push(_fire(m_dict, t), pos, cost, ('tau', None))

        if act is not None:
            for t in _label_to_trans.get(act, []):
                if _is_enabled(m_dict, t):
                    _push(_fire(m_dict, t), pos + 1, cost, ('S', act))

        for t in _visible_transitions:
            if _is_enabled(m_dict, t):
                _push(_fire(m_dict, t), pos, cost + 1, ('M', t.label))

        if act is not None:
            _push(m_dict, pos + 1, cost + 1, ('L', act))

    return [], float('inf')


def align_prefix(activities: list) -> dict:
    """Align prefix[:k] from first activity each time."""
    clean = [str(a) for a in activities]

    try:
        raw_alignment, total_cost = _dijkstra_prefix_alignment(clean)
    except Exception as e:
        raise RuntimeError(e)

    if total_cost == float('inf'):
        return {"error": "no_path_found"}

    aligned    = []
    step_types = []

    for mv_type, label in raw_alignment:
        if mv_type == 'tau':
            continue
        aligned.append(label)
        step_types.append(mv_type)

    total_sync  = step_types.count('S')
    total_log   = step_types.count('L')
    total_model = step_types.count('M')

    return dict(
        aligned_prefix = aligned,
        step_types     = step_types,
        cost           = total_cost,
        sync_moves     = total_sync,
        log_moves      = total_log,
        model_moves    = total_model,
        is_conforming  = (total_log == 0 and total_model == 0),
        error          = None,
    )

print("\n" + "=" * 70)
print("STEP 6 — Dataset generation")
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
    # for k in range(1, len(activities) + 1, 2):
        result = align_prefix(activities[:k])   # fresh each time, no state passed

        if result.get("error"):
            errors += 1

        rows.append({
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
        print(rows)


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

print(f"\nDone. Total prefixes: {total_prefixes}  Errors: {errors}")
print(f"Saved → {DS_CSV}")

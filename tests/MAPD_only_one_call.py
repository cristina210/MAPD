import sys, os, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
from graph_utils.upload_graph import make_grid_graph
from extended_time_graph import TimeExpandedGraph
from Simulator import Simulator
from MAPD_solver import MAPD_solver
import matplotlib.pyplot as plt
import networkx as nx
from utils import *
from utils_per_result import *


"""
MAPD comparison test — single MAPD call per instance.

For N_ITER random instances (random initial AMR positions, random pre-existing
_progress for 0-4 agents, random missions with waypoints), runs MAPD_solver once
with each of PP, CBS, BCBS and checks:
  - success/failure (including crashes)
  - conflict-freedom of the resulting _progress (check_progress_conflicts)
  - whether all requested waypoints are reached in order (check_waypoints_in_progress)
  - total path cost (sum of _progress lengths over agents with waypoints)

All the three solvers are tested
"""

def check_waypoints_in_progress(missions, sim):
    """
    For each agent with waypoints, checks that all its target waypoints
    (excluding the start node, missions[i][1][0]) appear in sim._progress[agent_id]
    in order (as a subsequence). Note: _progress never contains the start node.

    Returns (all_ok, per_agent_dict).
    """
    per_agent = {}
    all_ok = True

    for agent_id, waypoints in missions:
        targets = waypoints[1:]  # exclude start node

        if not targets:
            per_agent[agent_id] = True
            continue

        progress = sim._progress.get(agent_id, [])
        idx = 0
        ok = True
        for i in range(0, len(targets)):
            target = targets[i]
            if waypoints[i] == waypoints[i+1]:
                continue
            found = False
            while idx < len(progress):
                if progress[idx] == target:
                    found = True
                    idx += 1
                    break
                idx += 1
            if not found:
                ok = False
                break

        per_agent[agent_id] = ok
        if not ok:
            all_ok = False

    return all_ok, per_agent


N_ITER  = 100
SOLVERS   = ["PP", "CBS", "BCBS"]
MAX_PROGRESS_LEN = 8
OUTPUT_DIR = "result"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mapd_test_results_one_call.txt")


os.makedirs(OUTPUT_DIR , exist_ok=True)

G = make_grid_graph(rows=4, cols=4, step=1.0)
nodes = list(G.nodes())

stats = {
    s: {"success": 0, "failed": 0, "conflicts": 0, "waypoints_ok": 0,
        "total_cost": 0.0, "valid_runs": 0}
    for s in SOLVERS
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for it in range(N_ITER):
        random.seed(it)

        # random initial positions (4 distinct nodes)
        initial_positions = random.sample(nodes, 4)

        # randomly choose how many agents (0..4) start with a non-empty _progress
        n_with_progress = random.randint(0, 4)
        agents_with_progress = set(random.sample(range(4), n_with_progress))

        base_progress = {
            agent_id: (random_progress(G) if agent_id in agents_with_progress else [])
            for agent_id in range(4)
        }

        # base simulator just to derive missions consistently with base_progress
        base_sim = Simulator(graph=G, initial_positions=initial_positions)
        for agent_id in range(4):
            if base_progress[agent_id]:
                base_sim._progress[agent_id] = list(base_progress[agent_id])

        missions = random_missions(base_sim)

        if it in range(N_ITER-5,N_ITER):

            f.write(f"\n{'='*70}\n")
            f.write(f"ITERATION {it}\n")
            f.write(f"{'='*70}\n")
            f.write(f"initial_positions: {initial_positions}\n")
            f.write(f"initial _progress: {base_progress}\n")
            f.write(f"missions: {missions}\n\n")

        for solver_name in SOLVERS:
            # fresh simulator with identical initial state for each solver
            sim = Simulator(graph=G, initial_positions=initial_positions)
            for agent_id in range(4):
                if base_progress[agent_id]:
                    sim._progress[agent_id] = list(base_progress[agent_id])

            mapd = MAPD_solver(sim, solver_name=solver_name)

            if it in range(N_ITER-5,N_ITER):
                f.write(f"  -- {solver_name} --\n")

            try:
                success = mapd.compute_paths(copy.deepcopy(missions))
            except Exception as e:
                stats[solver_name]["failed"] += 1
                if it in range(N_ITER-5,N_ITER):
                    f.write(f"  CRASHED: {e}\n\n")
                continue
            if it in range(N_ITER-5,N_ITER):
                f.write(f"  success: {success}\n")

            if not success:
                stats[solver_name]["failed"] += 1
                f.write("\n")
                continue

            stats[solver_name]["success"] += 1

            if it in range(N_ITER-5,N_ITER):
                for agent_id, path in sim._progress.items():
                    f.write(f"    AMR {agent_id} progress: {path}\n")

            # conflicts (note: pre-existing conflicts in random base_progress
            # among non-planned agents are reported too, not only MAPD-induced ones)
            ok, conflicts = check_progress_conflicts(sim)
            if not ok:
                stats[solver_name]["conflicts"] += 1
                if it in range(N_ITER-5,N_ITER):
                    f.write(f"  conflicts: {conflicts}\n")
            else:
                if it in range(N_ITER-5,N_ITER):
                    f.write("  conflicts: none\n")

            # waypoints check
            wp_ok, per_agent = check_waypoints_in_progress(missions, sim)
            if wp_ok:
                stats[solver_name]["waypoints_ok"] += 1
            if it in range(N_ITER-5,N_ITER):
                f.write(f"  waypoints reached: {wp_ok} ({per_agent})\n")

            # cost
            cost = total_cost(sim, missions)
            stats[solver_name]["total_cost"] += cost
            stats[solver_name]["valid_runs"] += 1
            if it in range(N_ITER-5,N_ITER):
                f.write(f"  cost (total path length): {cost}\n\n")

    # final summary
    f.write(f"\n{'='*70}\n")
    f.write("SUMMARY\n")
    f.write(f"{'='*70}\n")
    for solver_name in SOLVERS:
        s = stats[solver_name]
        avg_cost = s["total_cost"] / s["valid_runs"] if s["valid_runs"] > 0 else 0.0
        f.write(f"\n-- {solver_name} --\n")
        f.write(f"  success:         {s['success']}/{N_ITER}\n")
        f.write(f"  failed:          {s['failed']}/{N_ITER}\n")
        f.write(f"  conflicts found: {s['conflicts']}\n")
        f.write(f"  waypoints ok:    {s['waypoints_ok']}/{s['success']}\n")
        f.write(f"  avg cost:        {avg_cost:.2f}\n")

print(f"Results written to {OUTPUT_FILE}")


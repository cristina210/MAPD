import os

import random
import os
import copy
from graph_utils.upload_graph import make_grid_graph
from Simulator import Simulator
from MAPD_solver import MAPD_solver
from utils import check_progress_conflicts
N_ITER  = 100
SOLVERS   = ["PP", "CBS", "BCBS"]
MAX_PROGRESS_LEN = 8
OUTPUT_DIR = "result"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mapd_test_results_one_call.txt")


def random_progress(graph, max_len=MAX_PROGRESS_LEN):
    """Random non-empty _progress list (length 1..max_len) with random nodes from graph."""
    length = random.randint(1, max_len)
    nodes = list(graph.nodes())
    return [random.choice(nodes) for _ in range(length)]


def random_missions(sim, n_waypoints_range=(1, 3), idle_prob=0.2):
    """
    Builds random missions for each agent.
    start_node = last node in _progress if non-empty, else amr_positions.
    With probability idle_prob the agent gets no waypoints (idle, len==1).
    """
    nodes = list(sim.graph.nodes())
    missions = []
    for agent_id in range(sim.N_AMR):
        if len(sim._progress.get(agent_id, [])) > 0:
            start_node = sim._progress[agent_id][-1]
        else:
            start_node = sim.amr_positions[agent_id]

        if random.random() < idle_prob:
            missions.append((agent_id, [start_node]))
            continue

        n_wp = random.randint(*n_waypoints_range)
        waypoints = [random.choice(nodes) for _ in range(n_wp)]
        missions.append((agent_id, [start_node] + waypoints))
    return missions



def total_cost(sim, missions):
    """Sum of len(_progress[agent_id]) over agents that had at least one waypoint."""
    total = 0
    for agent_id, waypoints in missions:
        if len(waypoints) > 1:
            total += len(sim._progress.get(agent_id, []))
    return total


def save_comparison_results(filepath, stats, valid_runs, N_ITER, N_AGENTS,
                            GRID_ROWS, GRID_COLS, last_iterations_data):
    """
    Writes a clean summary of CBS/BCBS/PP comparison results to a txt file.
    """

    avg_pp_vs_cbs = (
        stats["cost_pp_minus_cbs"] / valid_runs if valid_runs > 0 else 0.0
    )
    avg_bcbs_vs_cbs = (
        stats["cost_bcbs_minus_cbs"] / valid_runs if valid_runs > 0 else 0.0
    )

    # crea cartella result se non esiste
    os.makedirs("result", exist_ok=True)

    # forza il salvataggio dentro result/
    filepath = os.path.join("result", filepath)

    W = 65

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * W + "\n")
        f.write(f" RESULTS over {N_ITER} iterations\n")
        f.write(f" Agents: {N_AGENTS} | Grid: {GRID_ROWS}x{GRID_COLS}\n")
        f.write("=" * W + "\n")

        f.write("\n-- Failures --\n")
        f.write(f"{'CBS failed':<35}: {stats['cbs_failed']}\n")
        f.write(f"{'BCBS failed':<35}: {stats['bcbs_failed']}\n")
        f.write(f"{'PP failed':<35}: {stats['pp_failed']}\n")

        f.write("\n-- Correctness --\n")
        f.write(f"{'CBS conflicts found':<35}: {stats['cbs_conflicts']}\n")
        f.write(f"{'BCBS conflicts found':<35}: {stats['bcbs_conflicts']}\n")
        f.write(f"{'PP conflicts found':<35}: {stats['pp_conflicts']}\n")

        f.write("\n-- Optimality / comparisons --\n")
        f.write(f"{'PP better than CBS [BUG]':<35}: {stats['pp_better_than_cbs']}\n")
        f.write(f"{'CBS better than PP':<35}: {stats['cbs_better_than_pp']}\n")
        f.write(f"{'CBS = PP':<35}: {stats['pp_equals_cbs']}\n")
        f.write(f"{'CBS better than BCBS':<35}: {stats['cbs_better_than_bcbs']}\n")
        f.write(f"{'BCBS = CBS':<35}: {stats['bcbs_equals_cbs']}\n")

        f.write("\n-- Average cost difference --\n")
        f.write(f"{'PP - CBS':<35}: {avg_pp_vs_cbs:.4f}\n")
        f.write(f"{'BCBS - CBS':<35}: {avg_bcbs_vs_cbs:.4f}\n")

        f.write("\n-- Success rate --\n")
        f.write(
            f"{'Valid runs':<35}: {valid_runs}/{N_ITER} "
            f"({100*valid_runs/N_ITER:.1f}%)\n"
        )

        f.write("=" * W + "\n")

        f.write("\n" + "=" * W + "\n")
        f.write(" PATHS — last iterations\n")
        f.write("=" * W + "\n")

        for data in last_iterations_data:
            f.write(f"\n--- Iteration {data['iter']:03d} ---\n")
            f.write(f"  start/goal:  {data['diz_start_and_goal']}\n")
            f.write(f"  start times: {data['diz_time_start']}\n")

            for agent_id, paths in data["agents"].items():
                f.write(f"  agent {agent_id}:\n")
                f.write(f"    CBS  (len {len(paths['cbs'])}):  {paths['cbs']}\n")
                f.write(f"    BCBS (len {len(paths['bcbs'])}): {paths['bcbs']}\n")
                f.write(f"    PP   (len {len(paths['pp'])}):   {paths['pp']}\n")
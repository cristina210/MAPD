import sys, os, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graph_utils.upload_graph import make_grid_graph
from fleet_utils.upload_fleet import make_random_fleet
from MAPF_algorithm.solvers.CBS_mapd import CBSSolver_mapd
from MAPF_algorithm.solvers.BCBS_mapd import BCBSSolver_mapd
from MAPF_algorithm.solvers.PP_mapd import PrioritizedPlanner_mapd
from utils import *
from utils_per_result import *
import copy

"""
MAPF algorithms comparison — CBS vs BCBS vs PP, asynchronous starts + obstacles.
For N_ITER random instances (random fleet with N_AGENTS, asynchronous start
times mapf_sincrono=False), builds a TEG and injects external constraints:
  - other agents' start positions as vertex obstacles (mapf_no_obstacle off)
  - extra random vertex/edge constraints (simulated environment obstacles)

Runs CBSSolver_mapd, BCBSSolver_mapd and PrioritizedPlanner_mapd on the same
instance/constraints and checks:
  - success/failure of each solver
  - conflict-freedom and constraint satisfaction (verify_solution_with_constraints)
  - sum-of-costs comparison: CBS (optimal) vs BCBS vs PP, flagging any case
    where PP beats CBS (should never happen)
"""


N_ITER    = 500
N_AGENTS  = 6
GRID_ROWS = 4
GRID_COLS = 4

# numero di iterazioni finali per cui salvare i path completi
N_LAST_TO_PRINT = 10

stats = {
    "cbs_failed": 0,
    "bcbs_failed": 0,
    "pp_failed": 0,

    "pp_better_than_cbs": 0,
    "bcbs_better_than_cbs": 0,

    "cbs_conflicts": 0,
    "bcbs_conflicts": 0,
    "pp_conflicts": 0,

    "cbs_better_than_pp": 0,
    "cbs_better_than_bcbs": 0,
    "pp_equals_cbs": 0,
    "bcbs_equals_cbs": 0,

    "ok": 0,

    "cost_pp_minus_cbs": 0.0,
    "cost_bcbs_minus_cbs": 0.0,
}

valid_runs = 0
last_iterations_data = []

for i in range(N_ITER):
    print(i)
    random.seed(i*10)
    G  = make_grid_graph(rows=GRID_ROWS, cols=GRID_COLS,  step=1.0)
    fleet = make_random_fleet(G, num_agents=N_AGENTS, mapf_sincrono=False)
    T = 2 * compute_T_min(fleet, graph=G)
    teg = TimeExpandedGraph(G, T)

    diz_obstacle_from_start_pos = {}   # contiene gli ostacoli/vincoli aggiuntivi, vengono inseriti anche vincoli di start iniziali degli agent

    for aid in range(0,fleet.num_agents()):
        diz_obstacle_from_start_pos[aid] = set()
        for aid_other in range(0,fleet.num_agents()):
            if aid != aid_other:
                diz_obstacle_from_start_pos[aid].add(teg.get_expanded_id_from_original_id(fleet.agents[aid_other].start, fleet.agents[aid_other].start_t))


    vertex_constraints = diz_obstacle_from_start_pos
    vertex_constraints = {aid: set() for aid in fleet.agents}
    edge_constraints = {aid: set() for aid in fleet.agents}


    # vincoli vertex random aggiuntivi
    all_nodes = list(G.nodes())
    all_edges = list(G.edges())

    num_extra_vertex = 8
    num_extra_edge = 4

    for aid in fleet.agents:
        for _ in range(num_extra_vertex):
            rand_node = random.choice(all_nodes)
            rand_t = random.randint(0, T - 1)
            # evito di bloccare lo start dell'agente stesso al suo start_t
            agent = fleet.agents[aid]
            if rand_node == agent.start and rand_t == agent.start_t:
                continue
            exp_node = teg.get_expanded_id_from_original_id(rand_node, rand_t)
            vertex_constraints[aid].add(exp_node)

        for _ in range(num_extra_edge):
            rand_edge = random.choice(all_edges)
            src, dst = rand_edge
            rand_t = random.randint(0, T - 2)  # T-2 perché l'arco va da t a t+1
            exp_src = teg.get_expanded_id_from_original_id(src, rand_t)
            exp_dst = teg.get_expanded_id_from_original_id(dst, rand_t + 1)
            edge_constraints[aid].add((exp_src, exp_dst))

    # costruisci diz_start_and_goal e diz_time_start dalla fleet
    diz_start_and_goal = {
        aid: (agent.start, agent.goal)
        for aid, agent in fleet.agents.items()
    }
    diz_time_start = {
        aid: agent.start_t
        for aid, agent in fleet.agents.items()
    }

    result_cbs = CBSSolver_mapd(
        teg=teg,
        diz_constr_vertex=copy.deepcopy(vertex_constraints),
        diz_constr_edge=copy.deepcopy(edge_constraints)
    ).plan(diz_start_and_goal, diz_time_start)

    result_bcbs = BCBSSolver_mapd(
        teg=teg,
        diz_constr_vertex=copy.deepcopy(vertex_constraints),
        diz_constr_edge=copy.deepcopy(edge_constraints)
    ).plan(diz_start_and_goal, diz_time_start)

    result_pp = PrioritizedPlanner_mapd(
        teg=teg,
        diz_constr_vertex=copy.deepcopy(vertex_constraints),
        diz_constr_edge=copy.deepcopy(edge_constraints)
    ).plan(diz_start_and_goal, diz_time_start)
    # unpack solver returns
    cbs_paths, cbs_ok = result_cbs
    bcbs_paths, bcbs_ok = result_bcbs
    pp_paths, pp_ok = result_pp


    # ── failures ──────────────────────────────────────────────────────────────

    if not cbs_ok:
        stats["cbs_failed"] += 1
        print(f"[{i:03d}] CBS failed")
        continue

    if not pp_ok:
        stats["pp_failed"] += 1
        print(f"[{i:03d}] PP failed")
        continue

    if not bcbs_ok:
        stats["bcbs_failed"] += 1
        print(f"[{i:03d}] BCBS failed")
        continue


    # ── conflict checks ───────────────────────────────────────────────────────

    cbs_valid, cbs_conf = verify_solution_with_constraints( cbs_paths, fleet, teg, vertex_constraints, edge_constraints)

    bcbs_valid, bcbs_conf = verify_solution_with_constraints( bcbs_paths, fleet, teg, vertex_constraints, edge_constraints)

    pp_valid, pp_conf = verify_solution_with_constraints(pp_paths, fleet, teg, vertex_constraints, edge_constraints)



    if not cbs_valid:
        stats["cbs_conflicts"] += 1
        print(f"[{i:03d}] BUG CBS — conflicts: {cbs_conf}")
        continue


    if not bcbs_valid:
        stats["bcbs_conflicts"] += 1
        print(f"[{i:03d}] BUG BCBS — conflicts: {bcbs_conf}")
        continue


    if not pp_valid:
        stats["pp_conflicts"] += 1
        print(f"[{i:03d}] BUG PP — conflicts: {pp_conf}")
        continue


    # ── costs ─────────────────────────────────────────────────────────────────

    cost_cbs = sum(len(path) for path in cbs_paths.values())
    cost_bcbs = sum(len(path) for path in bcbs_paths.values())
    cost_pp = sum(len(path) for path in pp_paths.values())


    # ── optimality bug ────────────────────────────────────────────────────────

    if cost_pp < cost_cbs:
        stats["pp_better_than_cbs"] += 1
        print(
            f"[{i:03d}] BUG PP < CBS: "
            f"CBS={cost_cbs} PP={cost_pp}"
        )
        continue


    # ── comparison ────────────────────────────────────────────────────────────

    if cost_cbs < cost_pp:
        stats["cbs_better_than_pp"] += 1
    else:
        stats["pp_equals_cbs"] += 1


    if cost_cbs < cost_bcbs:
        stats["cbs_better_than_bcbs"] += 1
    else:
        stats["bcbs_equals_cbs"] += 1


    stats["cost_bcbs_minus_cbs"] += cost_bcbs - cost_cbs
    stats["cost_pp_minus_cbs"] += cost_pp - cost_cbs

    stats["ok"] += 1
    valid_runs += 1


    if cost_pp > cost_cbs:
        print(
            f"[{i:03d}] PP suboptimal: "
            f"CBS={cost_cbs} PP={cost_pp}"
        )

    # ── collect paths for the last N iterations ──────────────────────────────
    if i >= N_ITER - N_LAST_TO_PRINT:
        # convert expanded node ids to original node ids (after constraint checks)
        cbs_paths_orig = {
            aid: [teg.get_original_id_from_expanded(n) for n in path]
            for aid, path in cbs_paths.items()
        }
        bcbs_paths_orig = {
            aid: [teg.get_original_id_from_expanded(n) for n in path]
            for aid, path in bcbs_paths.items()
        }
        pp_paths_orig = {
            aid: [teg.get_original_id_from_expanded(n) for n in path]
            for aid, path in pp_paths.items()
        }

        agents_data = {}
        for agent_id in cbs_paths_orig:
            agents_data[agent_id] = {
                "cbs": cbs_paths_orig.get(agent_id),
                "bcbs": bcbs_paths_orig.get(agent_id),
                "pp": pp_paths_orig.get(agent_id),
            }
        last_iterations_data.append({
            "iter": i,
            "diz_start_and_goal": diz_start_and_goal,
            "diz_time_start": diz_time_start,
            "agents": agents_data,
        })

# ─────────────────────────────────────────────────────────────
# SAVE CLEAN SUMMARY TO FILE
# ─────────────────────────────────────────────────────────────

save_comparison_results(
    "result_comparing_MAPF_algorithm_with_obstacles_and_asynchronous.txt",
    stats,
    valid_runs,
    N_ITER,
    N_AGENTS,
    GRID_ROWS,
    GRID_COLS,
    last_iterations_data,
)

print(f"\nSummary written to result_comparison4.txt ({valid_runs}/{N_ITER} valid runs)")
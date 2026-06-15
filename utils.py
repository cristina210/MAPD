import random
from extended_time_graph import TimeExpandedGraph
import matplotlib.pyplot as plt
import networkx as nx
from instance.Network_graph import NetworkGraph
import os
random.seed(40)

def compute_T_min(fleet, graph) -> int:
    ''' Compute makespan if each agent plan its shortest path neglecting conflicts with others'''
    shortest_path_dix = {}   # cache riusata in _validate
    max_len = 0
    for a in fleet.agents.values():
        if a.goal is None:
            continue
        path = nx.shortest_path(graph, a.start, a.goal)
        if path is None:
            raise ValueError(f"Agent {a.id}: no path from {a.start} to {a.goal}")
        shortest_path_dix[a.id] = path
        travel_time = len(path) - 1
        arrival_time = a.start_t + travel_time
        max_len = max(max_len, arrival_time)
    return max_len


def verify_solution_with_constraints(result: dict ,fleet: Fleet ,teg: TimeExpandedGraph, vertex_constraints: dict, edge_constraints: dict) -> tuple[bool, list]:
    '''usare nodi espansi perchè si parte da t diversi'''
    paths = result
    agent_ids = list(paths.keys())
    conflicts = []

    # 1. CHECK MAPF CONFLICTS
    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            ai, aj = agent_ids[i], agent_ids[j]
            path_i = paths[ai]   # devo per forza usare l'espanso perchè possono portare in momenti diversi
            path_j = paths[aj]
            # vertex conflict
            if set(path_i) & set(path_j):
                conflicts.append({"type": "vertex", "agents": (ai, aj)})

            # swap edge conflict
            edges_i = {(path_i[z], path_i[z+1]) for z in range(len(path_i)-1)}
            edges_j = {(path_j[z], path_j[z+1]) for z in range(len(path_j)-1)}
            for ei in edges_i:
                if ei in teg.swap_pairs and teg.swap_pairs[ei] in edges_j:
                    conflicts.append({"type": "swap edge", "agents": (ai, aj)})

    # 2. CHECK EXTERNAL VERTEX CONSTRAINTS
    for agent_id, constraints in vertex_constraints.items():
        path = paths[agent_id]
        for node in path:
            if node in constraints:
                conflicts.append({ "type": "external obstacle", "agents": (agent_id,)})               


    # 3. CHECK EXTERNAL EDGE CONSTRAINTS
    for agent_id, constraints in edge_constraints.items():
        path = paths[agent_id]
        for i in range(0,len(path)-1):
            edge = (path[i], path[i + 1])
            if edge in constraints:
                conflicts.append({"type": "edge_constraint","agents": (agent_id,)})
                print("node in external edges, agent: ", agent_id, "in", teg.get_original_id_from_expanded(path[i]), teg.get_original_id_from_expanded(path[j]),"in time", path[i] % teg.T)

    return (len(conflicts) == 0), conflicts

    
def check_progress_conflicts(sim) -> tuple[bool, list]:
    """
    Checks sim._progress for vertex and edge (swap) conflicts between agents.

    Agents with no scheduled future path (idle/stationary, i.e. empty
    sim._progress[agent_id]) are excluded from the check entirely —
    treated as "disappeared at target", consistent with MAPF
    disappear-at-goal semantics.

    sim._progress[agent_id] contains the FUTURE scheduled nodes (relative time,
    index 0 = next step, t=1). The current position (t=0) is sim.amr_positions[agent_id]
    and is prepended to build the full timeline for each agent.

    Checks:
    - vertex conflict: two agents occupy the same node at the same timestep
    - edge conflict (swap): agent i goes u->v while agent j goes v->u
      in the same timestep interval [t, t+1]

    Args:
        sim: Simulator object (uses sim.amr_positions and sim._progress)
    Returns:
        (True, [])             if no conflicts are found
        (False, conflict_list) if conflicts are found
    """
    conflicts = []

    # only agents with a non-empty future path are checked
    agent_ids = [
        aid for aid in range(sim.N_AMR)
        if aid in sim._progress and len(sim._progress[aid]) > 0
    ]

    full_paths = {
        aid: [sim.amr_positions[aid]] + list(sim._progress[aid])
        for aid in agent_ids
    }

    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            ai, aj = agent_ids[i], agent_ids[j]
            path_i = full_paths[ai]
            path_j = full_paths[aj]

            min_len = min(len(path_i), len(path_j))

            for t in range(min_len):
                if path_i[t] == path_j[t]:
                    conflicts.append({
                        "type": "vertex",
                        "agents": (ai, aj),
                        "timestep": t,
                        "node": path_i[t],
                    })

                if t + 1 < min_len:
                    if path_i[t] == path_j[t + 1] and path_j[t] == path_i[t + 1]:
                        conflicts.append({
                            "type": "edge_swap",
                            "agents": (ai, aj),
                            "timestep": t,
                            "nodes": (path_i[t], path_i[t + 1]),
                        })

    return (len(conflicts) == 0), conflicts
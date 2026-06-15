from extended_time_graph import TimeExpandedGraph


def _get_conflicting_pairs(solution: dict, T: int, internal_to_orig: dict) -> set:
    """
    Returns the set of agent pairs (ai, aj) that have at least one conflict
    (vertex or edge) in the given solution.
    Used as a base for h2 and h3 heuristics.
    """
    conflicting_pairs = set()
    agent_ids = list(solution.keys())

    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            ai, aj = agent_ids[i], agent_ids[j]
            path_i, path_j = solution[ai], solution[aj]

            for t in range(min(len(path_i), len(path_j)) - 1):
                or_i_t  = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_i[t], T)]
                or_j_t  = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_j[t], T)]
                or_i_t1 = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_i[t + 1], T)]
                or_j_t1 = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_j[t + 1], T)]

                if or_i_t == or_j_t or (or_i_t == or_j_t1 and or_j_t == or_i_t1):
                    conflicting_pairs.add((ai, aj))
                    break

            if (ai, aj) not in conflicting_pairs:
                last_t = min(len(path_i), len(path_j)) - 1
                l_i = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_i[last_t], T)]
                l_j = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_j[last_t], T)]
                if l_i == l_j:
                    conflicting_pairs.add((ai, aj))

    return conflicting_pairs


def h1_num_conflicts(solution: dict, T: int, internal_to_orig: dict) -> int:
    """
    h1: total number of conflicts across all pairs and all timesteps.
    Counts every (pair, timestep) occurrence — most granular heuristic.
    """
    agent_ids = list(solution.keys())
    count = 0

    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            ai, aj = agent_ids[i], agent_ids[j]
            path_i, path_j = solution[ai], solution[aj]

            for t in range(min(len(path_i), len(path_j)) - 1):
                or_i_t  = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_i[t],     T)]
                or_j_t  = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_j[t],     T)]
                or_i_t1 = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_i[t + 1], T)]
                or_j_t1 = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_j[t + 1], T)]

                if or_i_t == or_j_t:
                    count += 1
                if or_i_t == or_j_t1 and or_j_t == or_i_t1:
                    count += 1

            last_t = min(len(path_i), len(path_j)) - 1
            l_i = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_i[last_t], T)]
            l_j = internal_to_orig[TimeExpandedGraph.get_internal_id_from_expanded(path_j[last_t], T)]
            if l_i == l_j:
                count += 1

    return count


def h2_num_conflicting_agents(solution: dict, T: int, internal_to_orig: dict) -> int:
    """
    h2: number of agents involved in at least one conflict.
    """
    pairs = _get_conflicting_pairs(solution, T, internal_to_orig)
    conflicting_agents = {agent for pair in pairs for agent in pair}
    return len(conflicting_agents)


def h3_num_conflicting_pairs(solution: dict, T: int, internal_to_orig: dict) -> int:
    """
    h3: number of pairs of agents with at least one conflict.
    """
    return len(_get_conflicting_pairs(solution, T, internal_to_orig))


def h1_low(g_c_current: float, neighbor: int, congestion_dict: dict) -> float:
    """Low-level h1: total conflicts along partial path."""
    return g_c_current + congestion_dict.get(neighbor, 0)

def h2_low(g_c_current: float, neighbor: int, congestion_dict: dict) -> float:
    """Low-level h2: number of congested nodes visited."""
    return g_c_current + (1 if congestion_dict.get(neighbor, 0) > 0 else 0)

def h3_low(g_c_current: float, neighbor: int, congestion_dict: dict) -> float:
    """Low-level h3: conflicting pairs — equivalent to h1 at low level."""
    return g_c_current + congestion_dict.get(neighbor, 0)

HEURISTICS_LOW_L = {
    "h1": h1_low,
    "h2": h2_low,
    "h3": h3_low,
}

HEURISTICS_HIGH_L = {
    "h1": h1_num_conflicts,
    "h2": h2_num_conflicting_agents,
    "h3": h3_num_conflicting_pairs,
}
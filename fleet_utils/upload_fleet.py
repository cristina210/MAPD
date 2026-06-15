import random
from instance.Network_graph import NetworkGraph as nx # o dal path corretto
import networkx as nx
from fleet_utils.fleet import Fleet

def make_random_fleet(G, num_agents: int = 3, mapf_sincrono: bool = True,
                      min_dist: int = 4, max_start_t: int = None) -> Fleet:
    """
    Generate a fleet of agents with initial position and goals chosen from nodes of a graph.

    Guarantees:
    - All start nodes are distinct.
    - All goal nodes are distinct.
    - No goal coincides with any start node (avoids trivial t=0 conflicts).
    - Start and goal of each agent are at least min_dist hops apart (shortest path).
      If no candidate satisfies min_dist, the constraint is relaxed gracefully.
    - In async mode, start_t values are in [0, max_start_t] and never exceed T/4
      to ensure all agents can start within the time horizon.

    Args:
        G:            graph (NetworkX or NetworkGraph)
        num_agents:   number of agents to generate
        min_dist:     minimum shortest-path distance between start and goal (default: 4)
        max_start_t:  upper bound for start_t in async mode.
                      Defaults to max(1, num_agents // 2) if not provided.
    Returns:
        Fleet object
    """
    all_nodes = list(G.nodes)

    assert len(all_nodes) >= 2 * num_agents, (
        f"Graph has only {len(all_nodes)} nodes — not enough for "
        f"{num_agents} agents with distinct starts and goals."
    )

    # ── starts: all distinct ──────────────────────────────────────────────────
    starts = random.sample(all_nodes, num_agents)
    print("Starts node:")
    print(starts)
    used_starts = set(starts)

    # ── goals: distinct, not coinciding with any start ────────────────────────
    goals = []
    used_goals = set(starts)   # block starts too, so goal != any start

    for start in starts:
        # prefer nodes that are at least min_dist hops away
        candidates = [
            n for n in all_nodes
            if n not in used_goals
            and nx.shortest_path_length(G, start, n) >= min_dist
        ]

        if not candidates:
            # relax min_dist: just avoid used goals
            candidates = [n for n in all_nodes if n not in used_goals]

        if not candidates:
            raise RuntimeError(
                f"Could not assign a goal for agent starting at {start}: "
                f"no free nodes left. Reduce num_agents or increase the graph."
            )

        goal = random.choice(candidates)
        goals.append(goal)
        used_goals.add(goal)

    # ── start times ───────────────────────────────────────────────────────────
    ids = list(range(num_agents))

    if mapf_sincrono:
        t_start = [0 for _ in ids]
    else:
        if max_start_t is None:
            max_start_t = max(1, num_agents // 2)
        t_start = [random.randint(0, max_start_t) for _ in ids]
        print("t_starts:")
        print(t_start)

    return Fleet(
        ids=ids,
        starts=starts,
        goals=goals,
        start_ts=t_start
    )

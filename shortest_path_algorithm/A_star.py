import heapq
import math
from instance.Network_graph import NetworkGraph
from MAPF_algorithm.heuristicsBCBS import HEURISTICS_LOW_L


def a_star(G: NetworkGraph, start: int, goal: int, extended = False, heuristic = None) -> list | None:
    """
    Standard A* pathfinding on a NetworkGraph.
    Supports two modes:
    - Standard (extended=False): finds the shortest path between two node ids in the original graph.
    - Extended (extended=True): operates on a time-expanded graph. The goal is matched by the 'original_id' node attribute, so the agent can reach the
      goal at any feasible timestep. The heuristic uses the spatial coordinates of the goal at t=0 as a reference.

    Duplicate entries in the heap are handled in the following way: when a node is popped, it is skipped if a better path was already found
    (checked via f_current > f[current]).

    Args:
        G: directed NetworkGraph (standard or time-expanded)
        start: expanded node id at a certain t (extended) or original node id (standard)
        goal: original_id to reach (extended) or goal node id (standard)
        extended: if True, uses time-expanded graph logic
        heuristic: admissible heuristic function h(G, node, goal_ref). Default: h_manhattan
    Returns:
        list of node ids representing the optimal path, or None if no path exists.
        In extended mode, ids are expanded node ids.
    """
    if heuristic is None:
        heuristic = h_manhattan

    # early exit if start == goal
    if not extended and start == goal:
        return [start]
    if extended and G.nodes[start]["original_id"] == goal:
        return [start]

    # in extended mode: find the expanded node (goal_original_id, t=0)
    # its spatial coords are used by the heuristic across all timesteps

    if extended:
        goal_ref = None
        for nid, data in G.nodes(data=True):
            if data["original_id"] == goal:
                goal_ref = nid
                break
        if goal_ref is None:
            return None  # il goal non esiste in nessun timestep del TEG (caso davvero irraggiungibile)
    else:
        goal_ref = goal



    # node_to_predecessor[n] = predecessor of n on the current best path (shortest) from start
    node_to_predecessor = {}    

    # g[n] = cost of the best known path from start to n
    g = {start: 0}

    # f[n] = g[n] + h(n, goal_ref): estimated total cost from start to goal through n
    f = {start: heuristic(G, start, goal_ref)}

    # OPEN: min-heap of (f_score, node_id)
    heap = [(f[start], start)]   

    while heap:   # ciclo heap

        f_current, current = heapq.heappop(heap)

        # skip if a better path to current was already found
        if f_current > f.get(current, float("inf")):
            continue

        # check if goal is reached
        if extended and G.nodes[current]["original_id"] == goal:
            return reconstruct_path(node_to_predecessor, current)
        if not extended and current == goal:
            return reconstruct_path(node_to_predecessor, current)


        # explore outgoing edges
        for _, neighbor, edge_attrs in G.out_edges(current, data=True):

            # optimal path for reaching neighbor passing from the current node
            possible_g = g[current] + edge_attrs["distance"]

            if possible_g < g.get(neighbor, float("inf")):
                # better path from start to neighbor found through current
                node_to_predecessor[neighbor] = current   # update optimal path to neighbor
                g[neighbor] = possible_g   # update best cost found so far 
                f[neighbor] = possible_g + heuristic(G, neighbor, goal_ref)   # update estimated cost for optimal path to the goal
                heapq.heappush(heap, (f[neighbor], neighbor))

    return None



def a_star_with_focal_search(G: NetworkGraph, start: int, goal: int, congestion_dict: dict,
                              extended: bool = False, heuristic=None, conflict_heuristic="h3", w: float = 1) -> list | None:
    """
    A* with focal search — low-level planner for BCBS.
    Maintains two lists:
    - OPEN: list of all discovered nodes, ordered by f = g + h.
      f_min (minimum f in OPEN) defines the suboptimality threshold.
    - FOCAL: subset of OPEN with f <= w * f_min, ordered by g_c
      (accumulated congestion cost). The node with lowest g_c in focal is expanded,
      minimising conflicts with other agents within the cost bound.

    Key difference from standard A*:
    Instead of always expanding the node with lowest f, focal search
    expands the node with lowest g_c among those within the cost bound.
    This trades a small cost increase (bounded by w) for fewer conflicts.
    When w=1, FOCAL = {nodes with f == f_min} and the behaviour matches
    standard A* when all g_c values are equal (no congestion).
    Duplicate handling: OPEN is a plain list. When a better path to a node
    is found, the old entry is removed before appending the new one.

    Args:
        G: directed NetworkGraph (standard or time-expanded)
        start: start node id
        goal:goal node id (standard) or original_id (extended)
        congestion_dict: {expanded_node_id: count} — number of other agents
        passing through each node in the current CT solution. Used to compute g_c incrementally during expansion.
        extended: if True, uses time-expanded graph logic
        heuristic: spatial heuristic h(G, node, goal_ref) (default: h_manhattan)
        conflict_heuristic: heuristic for congestion use for ordering of FOCAL
        w: suboptimality factor.
        w=1  -> optimal (FOCAL = f_min nodes only).
        w>1  -> bounded suboptimal (cost <= w * C*), faster.
    Returns:
        list of node ids representing the path, or None if no path exists.
    """
    if heuristic is None:
        heuristic = h_manhattan

    heuristic_function_c = HEURISTICS_LOW_L[conflict_heuristic]

    # early exit: agent already at goal
    if not extended and start == goal:
        return [start]
    if extended and G.nodes[start]["original_id"] == goal:
        return [start]

    if extended:
        goal_ref = next((nid for nid, d in G.nodes(data=True)
                        if d["original_id"] == goal), None)
        if goal_ref is None:
            return None  # il goal non esiste in nessun timestep del TEG
    else:
        goal_ref = goal
    

    # --- data structures ---

    # node_to_predecessor[n] = predecessor of n on the current best path from start
    node_to_predecessor = {}

    # g[n]: best known travel cost from start to n
    g = {start: 0}

    # g_c[n]: accumulated congestion cost along the best known path from start to n
    # incremented by congestion_dict[n] each time n is reached via a better path
    g_c = {start: heuristic_function_c(0, start, congestion_dict)}

    # f[n] = g[n] + h(n,goal): estimated total travel cost from start to goal through n
    f = {start: heuristic(G, start, goal_ref)}

    # OPEN: list of (f, g_c, node) entries
    open_list = [(f[start], g_c[start], start)]

    while open_list:


        # f_min: lowest f value currently in OPEN
        min_tuple = min(open_list, key=lambda t: t[0])
        f_min = min_tuple[0]

        threshold = f_min * w

        # build FOCAL: nodes within the suboptimality bound, ordered by g_c
        focal = get_focal_list(open_list, threshold)

        # extract node with lowest g_c from FOCAL
        f_current, g_c_current, current = focal.pop(0)

        if f_current > f.get(current, float("inf")):
            continue
        
        # remove current from OPEN
        open_list.remove((f_current, g_c_current, current))    # delete from set of open heap

        # goal check 
        if extended and G.nodes[current]["original_id"] == goal:
            return reconstruct_path(node_to_predecessor, current)
        if not extended and current == goal:
            return reconstruct_path(node_to_predecessor, current)

        # expand current: explore all outgoing edges
        for _, neighbor, edge_attrs in G.out_edges(current, data=True):

            # tentative travel cost to reach neighbor through current
            possible_g = g[current] + edge_attrs["distance"]

            # update only if a better path to neighbor is found
            if possible_g < g.get(neighbor, float("inf")):

                # remove old entry for neighbor from OPEN if present
                old_entry = (f.get(neighbor), g_c.get(neighbor), neighbor)
                if old_entry in open_list:
                    open_list.remove(old_entry)

                node_to_predecessor[neighbor] = current  # record best predecessor
                g[neighbor]   = possible_g               # update best travel cost
                f[neighbor]   = possible_g + heuristic(G, neighbor, goal_ref)  # update f

                # g_c propagates the total number of conflicts along the path
                g_c[neighbor] = heuristic_function_c(g_c[current], neighbor, congestion_dict)

                # push to OPEN — duplicates handled by lazy deletion via expanded set
                open_list.append((f[neighbor], g_c[neighbor], neighbor))

    return None  # OPEN exhausted: no path exists


def get_focal_list(listt: list, threshold: float) -> list:
    """
    Builds the FOCAL list from OPEN for focal search.
    Filters OPEN to include only nodes with f <= threshold (within the
    suboptimality bound), then sorts by (g_c, f) so the node with the
    lowest conflict cost is first.
    Args:
        open_list: OPEN as a list of (f, g_c, node) entries
        threshold: maximum f value to enter FOCAL (= w * f_min)
    Returns:
        sorted list of eligible entries ordered by (g_c, f)
    """
    focal_list = []
    for f_score, g_c_score, node in listt:
        if f_score <= threshold:
            focal_list.append((f_score, g_c_score, node))
    focal_list.sort(key=lambda x: (x[1], x[0]))
    return focal_list



def h_euclidean(G: NetworkGraph, n: int, goal_node: int) -> float:
    """
    Euclidean distance heuristic between node n and goal_node.
    Admissible only if edge weights equal Euclidean distances.
    goal_node must already be resolved to a node id in G.

    Args:
        G:         the graph (standard or time-expanded)
        n:         current node id
        goal_node: goal node id, already resolved
    Returns:
        Euclidean distance from n to goal_node
    """
    x_n,y_n = G.nodes[n]["x"], G.nodes[n]["y"]
    x_goal, y_goal = G.nodes[goal_node]["x"], G.nodes[goal_node]["y"]
    return math.sqrt((x_goal - x_n) ** 2 + (y_goal - y_n) ** 2)

def h_manhattan(G: NetworkGraph, n: int, goal_node: int) -> float:
    """
    Manhattan distance heuristic between node n and goal_node.

    Args:
        G: the graph
        n: current node id
        goal_node: goal node id (già risolto, non original_id)
    Returns:
        Manhattan distance from n to goal_node
    """
    x_n,y_n = G.nodes[n]["x"], G.nodes[n]["y"]
    x_goal, y_goal = G.nodes[goal_node]["x"], G.nodes[goal_node]["y"]
    return abs(x_goal - x_n) + abs(y_goal - y_n)


def reconstruct_path(node_to_predecessor, current):
    """
    Reconstructs the optimal path from start to current by following
    the predecessor chain stored during A* expansion.

    Args:
        node_to_predecessor: dict {node: predecessor} built during search
        current: goal node from which to backtrack
    Returns:
        list of node ids from start to goal (inclusive)
    """
    total_path = []
    while current in node_to_predecessor:
        total_path.append(current)
        current = node_to_predecessor[current]
    total_path.append(current)  
    total_path.reverse()
    return total_path
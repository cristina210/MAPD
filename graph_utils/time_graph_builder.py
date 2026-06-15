import networkx as nx


def build_id_remapping(G):
    """
    Creates a bijection between the arbitrary node ids of G and the range 0..N-1.
    This is necessary because the TEG formula  expanded_id = orig_id * T + t
    requires orig_id to be a consecutive integer starting from 0.
    Args:
        G: NetworkGraph whose node ids can be arbitrary integers
    Returns:
        orig_to_internal: dict  {original_id -> internal_id}   (e.g. {1->0, 3->1, 7->2, ...})
        internal_to_orig: dict  {internal_id -> original_id}   (inverse mapping)
    """
    orig_ids = sorted(G.nodes())
    orig_to_internal = {orig: idx for idx, orig in enumerate(orig_ids)}
    internal_to_orig = {idx: orig for idx, orig in enumerate(orig_ids)}
    return orig_to_internal, internal_to_orig


# TEG construction helpers 

def build_teg_mappings(G, T, orig_to_internal):
    """
    Precomputes the mapping  internal_id -> [expanded_id_t0, expanded_id_t1, ...] and the swap-pair dict for edge constraints.
    Args:
        G original graph (arbitrary node ids)
        T:  number of timesteps
        orig_to_internal: mapping from original id to internal consecutive id
    Returns:
        internal_to_expanded: list where internal_to_expanded[internal_id][t] = expanded_id
        swap_pairs: dict {(u_exp_t, v_exp_t1): (v_exp_t, u_exp_t1)}
    """
    N = len(G.nodes())

    # internal_to_expanded[i][t] = i * T + t
    internal_to_expanded = []

    for i in range(N):
        expanded = [i * T + t for t in range(T)]
        internal_to_expanded.append(expanded)

    # build swap pairs: for each directed edge (src, dst) in the original graph,
    # register the temporal swap counterpart for every timestep
    swap_pairs = {}
    for src_orig, dst_orig, _ in G.edges(data=True):
        src_int = orig_to_internal[src_orig]
        dst_int = orig_to_internal[dst_orig]
        for t in range(T - 1):
            u_t  = internal_to_expanded[src_int][t]
            v_t1 = internal_to_expanded[dst_int][t + 1]
            v_t  = internal_to_expanded[dst_int][t]
            u_t1 = internal_to_expanded[src_int][t + 1]

            # forward arc -> its swap counterpart
            # add the swap-pair inverse if it exists in the original graph
            # (only bidirectional edges have a swap pair)
            if G.has_edge(dst_orig, src_orig):
                swap_pairs[(u_t, v_t1)] = (v_t, u_t1)

    return internal_to_expanded, swap_pairs


def create_edges_constraints(swap_pairs, edge_constraints_set):
    """
    Expands a set of edge constraints by adding the swap-pair inverse of each edge.
    Forbidding (u_t, v_t+1) must also forbid (v_t, u_t+1) to prevent swap conflicts.
    Args:
        swap_pairs:            dict {(u_exp_t, v_exp_t1): (v_exp_t, u_exp_t1)}
        edge_constraints_set:  set of (src_exp, dst_exp) pairs to forbid
    Returns:
        expanded set of edge constraints including swap-pair inverses
    """
    if edge_constraints_set is None:
        return set()
    edge_constraints_final = set()
    for (src, dst) in edge_constraints_set:
        edge_constraints_final.add((src, dst))
        if (src, dst) in swap_pairs:
            edge_constraints_final.add(swap_pairs[(src, dst)])
    return edge_constraints_final


def build_expanded_graph(G, T, orig_to_internal, internal_to_expanded, vertex_constraints, edge_constraints):
    """
    Builds the time-expanded graph (TEG).
    Each node stores: original_id, t, x, y.
    Two edge types:
        - Wait edges:  (node, t) -> (node, t+1)   weight=1
        - Move edges:  (u, t)   -> (v, t+1)       weight=1

    Constraints are enforced by skipping forbidden nodes and edges (neglecting).

    Args:
        G: original graph (arbitrary node ids)
        T: number of timesteps
        orig_to_internal:  dict {orig_id -> internal_id}
        internal_to_expanded: list[list] — internal_to_expanded[i][t] = expanded_id
        vertex_constraints: set of expanded node ids to exclude
        edge_constraints: set of (src_exp, dst_exp) pairs to exclude
    Returns:
        G_expanded: nx.DiGraph (TEG)
    """
    if vertex_constraints is None:
        vertex_constraints = set()
    if edge_constraints is None:
        edge_constraints = set()

    G_expanded = nx.DiGraph()

    # add nodes and wait edges
    for orig_id, node_attrs in G.nodes(data=True):
        int_id = orig_to_internal[orig_id]
        for t in range(T):
            exp_id = internal_to_expanded[int_id][t]
            if exp_id not in vertex_constraints:
                # store original_id so A* can check goal matching by original id
                G_expanded.add_node(exp_id, x=node_attrs["x"], y=node_attrs["y"], t=t, original_id=orig_id)
            # wait edge from t-1 to t
            if t > 0:
                exp_prev = internal_to_expanded[int_id][t - 1]
                exp_curr = internal_to_expanded[int_id][t]
                if (exp_prev not in vertex_constraints and exp_curr not in vertex_constraints and (exp_prev, exp_curr) not in edge_constraints):
                    G_expanded.add_edge(exp_prev, exp_curr, distance=1, type_edge="wait")


    # add move edges
    for src_orig, dst_orig, _ in G.edges(data=True):
        src_int = orig_to_internal[src_orig]
        dst_int = orig_to_internal[dst_orig]
        for t in range(T - 1):
            u_t  = internal_to_expanded[src_int][t]
            v_t1 = internal_to_expanded[dst_int][t + 1]
            if (u_t not in vertex_constraints and v_t1 not in vertex_constraints and (u_t, v_t1) not in edge_constraints):
                G_expanded.add_edge(u_t, v_t1, distance=1, type_edge="move")

    return G_expanded
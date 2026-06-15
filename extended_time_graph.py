import networkx as nx
from graph_utils.time_graph_builder import build_id_remapping, build_teg_mappings, create_edges_constraints, build_expanded_graph

class TimeExpandedGraph:
    """
    Time-expanded NetworkGraph and provides utilities to:
    - map between original and expanded node ids
    - apply and update vertex/edge constraints
    - rebuild or incrementally update the expanded graph

    Structure of the time-expanded graph:
        Each node original_id is assigned to a sequential numbers from 0 to Number of nodes - 1. "internal nodes".
        These nodes provide a way to expande the graph:
        An unique expanded integer id is created: expanded_id = internal_id * T + t

    Two types of edges:
        - Wait edges:  (node, t) -> (node, t+1)  — agent waits in place
        - Move edges:  (u, t) -> (v, t+1) — agent moves from u to v

    Constraint handling:
        Vertex constraints forbid specific expanded nodes (and their incident edges).
        Edge constraints forbid specific (src, dst) expanded edge pairs.

    """

    def __init__(self, G, T, vertex_constraints=None, edge_constraints=None):
        """
        Builds the time-expanded graph with the given constraints.
        Construction order:
        1. Build id mapping: arbitrary IDs  - indexes 0..N-1 node id consecutiveness. orig_to_idx and idx_to_orig
        2. Build id mapping id original and expanded one (old_id_to_new) and swap_pairs via build_teg_mappings. This must happen before constraint expansion, since swap_pairs is needed to resolve edge constraint inverses.
        3. Expand edge constraints to include swap-pair inverses via create_edges_constraints.
        4. Build the actual expanded graph via time_expansion_graph_with_constr.

        Args:
            G: original directed NetworkGraph
            T: number of timesteps
            vertex_constraints: set of expanded node ids to forbid (default: empty)
            edge_constraints: set of (src, dst) expanded edge pairs to forbid (default: empty). Swap-pair inverses are added automatically.
        """
        self.G_original = G
        self.T = T
        self.vertex_constraints = vertex_constraints if vertex_constraints is not None else set()


        # build the remapping: arbitrary orig ids <-> internal consecutive ids
        self.orig_to_internal, self.internal_to_orig = build_id_remapping(G)

        # build node id mapping and swap pairs (no graph construction yet)
        self.internal_to_expanded, self.swap_pairs = build_teg_mappings(G, T, self.orig_to_internal)

        # expand edge constraints — add swap-pair inverses
        self.edge_constraints = create_edges_constraints(self.swap_pairs, edge_constraints)

        # costruzione del TEG
        self.G_expanded = build_expanded_graph(G, T, self.orig_to_internal, self.internal_to_expanded, self.vertex_constraints, self.edge_constraints)



    def add_vertex_constraint(self, *nodes: int) -> None:
        """
        Records one or more vertex constraints in the internal constraint set.
        Does not modify G_expanded — call update_teg_with_adding_constraints
        to apply changes to the graph.

        Args:
            nodes: one or more expanded node ids to forbid
        """
        for node in nodes:
            self.vertex_constraints.add(node)

    def add_edge_constraint(self, *edges: tuple) -> None:
        """
        Records one or more edge constraints in the internal constraint set.
        Does not modify G_expanded — call update_teg_with_adding_constraints
        to apply changes to the graph.
        Note: swap-pair inverses are not added here automatically.

        Args:
            edges: one or more (src, dst) expanded edge pairs to forbid
        """
        for edge in edges:
            self.edge_constraints.add(edge)


    def add_constraints_from_path(self, path: list) -> None:
        """
        Extracts and records vertex and edge constraints from a planned path.
        All nodes and edges in the path are forbidden for subsequent agents.
        Used by Prioritized Planner to block the path of an already-planned
        agent before planning the next one.
        Note: path must be in expanded node ids.
        Note: does not modify G_expanded — call update_teg_with_adding_constraints
        to apply the constraints incrementally.
        Note: swap-pair inverses are not added here automatically.

        Args:
            path: list of expanded node ids representing the planned path
        """
        self.add_vertex_constraint(*path)
        for i in range(len(path) - 1):
            self.add_edge_constraint((path[i], path[i + 1]))

    def rebuild_with_constraints(self, vertex_constraints=None, edge_constraints=None) -> None:
        """
        Replaces the current constraints and rebuilds G_expanded from scratch.

        Args:
            vertex_constraints: new set of forbidden expanded node ids (replaces current)
            edge_constraints:   new set of forbidden (src, dst) pairs (replaces current).
                                Swap-pair inverses are added automatically.
        """
        self.vertex_constraints = vertex_constraints if vertex_constraints is not None else set()
        self.edge_constraints = create_edges_constraints(self.swap_pairs, edge_constraints or set())
        self.G_expanded = build_expanded_graph(self.G_original, self.T, self.orig_to_internal, self.internal_to_expanded,self.vertex_constraints, self.edge_constraints)


    def update_teg_with_adding_constraints(self, new_vertex_constr, new_edge_constr) -> None:
        """
        Incrementally updates G_expanded by adding new constraints.
        Instead of rebuilding the entire graph, removes the newly forbidden
        nodes and edges from the current structure.
        Swap-pair inverses are added automatically to new_edge_constr.

        Args:
            new_vertex_constr: set of expanded node ids to forbid and remove
            new_edge_constr:   set of (src, dst) expanded edge pairs to forbid and remove.
                               Swap-pair inverses are added automatically.
        """
        # expand new edge constraints with swap-pair inverses
        new_edge_constr_final = create_edges_constraints(self.swap_pairs, new_edge_constr)
        # record new constraints in internal sets
        self.add_vertex_constraint(*new_vertex_constr)
        self.add_edge_constraint(*new_edge_constr_final)
        # apply constraints by removing nodes/edges from the current graph
        self.G_expanded = self._remove_nodes_edges_from_graph(self.G_expanded, new_vertex_constr, new_edge_constr_final)


    def _remove_nodes_edges_from_graph(self, G, new_vertex_constr, new_edge_constr):
        """
        Returns a new nx.DiGraph with the specified nodes and edges removed.
        An edge is removed if either endpoint is forbidden or the edge itself
        is in new_edge_constr (which already includes swap-pair inverses).

        Args:
            G:               current expanded graph
            new_vertex_constr: expanded node ids to remove
            new_edge_constr:   (src, dst) expanded edge pairs to remove
        Returns:
            new nx.DiGraph with constraints applied
        """
        G_new = nx.DiGraph()

        # copy nodes that are not forbidden
        for id_node, attrs in G.nodes(data=True):
            if id_node not in new_vertex_constr:
                G_new.add_node(id_node, **attrs)

        # copy edges whose endpoints are not forbidden and the edge itself is not forbidden
        for src, dst, attrs in G.edges(data=True):
            if (src not in new_vertex_constr
                    and dst not in new_vertex_constr
                    and (src, dst) not in new_edge_constr):
                G_new.add_edge(src, dst, **attrs)

        return G_new

    def get_expanded_id_from_original_id(self, original_id: int, t: int) -> int:
        int_id = self.orig_to_internal[original_id]
        return self.internal_to_expanded[int_id][t]

    def get_original_id_from_expanded(self, expanded_id: int) -> int:
        internal_id = expanded_id // self.T
        return self.internal_to_orig[internal_id]

    @staticmethod
    def get_expanded_id_from_internal(internal_id: int, t: int, T: int) -> int:
        return internal_id * T + t

    @staticmethod
    def get_internal_id_from_expanded(expanded_id: int, T: int) -> int:
        return expanded_id // T

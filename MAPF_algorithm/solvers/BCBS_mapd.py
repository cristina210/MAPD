import heapq
from typing import Optional
from instance.Network_graph import NetworkGraph
from fleet_utils.fleet import Fleet
from extended_time_graph import TimeExpandedGraph
from graph_utils.time_graph_builder import build_id_remapping
from shortest_path_algorithm.A_star import a_star_with_focal_search, a_star
from MAPF_algorithm.heuristicsBCBS import HEURISTICS_HIGH_L
from MAPF_algorithm.heuristicsBCBS import HEURISTICS_LOW_L
import copy

class BCBSNode:
    """
    Represents a node in the BCBS constraint tree (CT).
    Each node stores:
    - per-agent vertex and edge constraints accumulated from the root down to this node
    - a solution: one path (in expanded node ids) per active agent
    - the sum-of-costs of the solution
    - g_c: conflict heuristic used by the high-level focal search to prefer
      nodes whose solutions have fewer inter-agent conflicts
    g_c is initialized to 0 and must be set explicitly after solution is
    assigned (see BCBSSolver.plan), because solution is empty at __init__ time.
    The tree is explored via focal search: OPEN ordered by cost, FOCAL ordered by g_c.
    """

    def __init__(self, vertex_constraints=None, edge_constraints=None, solution=None, parent=None, cost_for_each_agent=None):
        """
        Args:
            vertex_constraints: dict {agent_id: set of forbidden expanded node ids}
            edge_constraints:   dict {agent_id: set of forbidden (src, dst) expanded edge pairs}
            solution:           dict {agent_id: list of expanded node ids} — inizialized empty at creation
            parent:             parent BCBSNode in the constraint tree, None for root
        """

        self.vertex_constraints = vertex_constraints if vertex_constraints is not None else {}
        self.edge_constraints = edge_constraints if edge_constraints is not None else {}
        self.solution = solution if solution is not None else {}
        self.parent = parent
        self.cost_for_each_agent = cost_for_each_agent if cost_for_each_agent is not None else {}
        self.cost = self._compute_cost()
        self.depth = 0 if parent is None else parent.depth + 1
        self.g_c = 0    # to update after creation (depends on solution)


    def _compute_cost(self) -> float:   # to do inserire anche makespan?
        """
        Sum-of-costs: sum of costs of paths of single agent
        """
        return sum(self.cost_for_each_agent.values())

    def __lt__(self, other: "BCBSNode") -> bool:
        # required by heapq when two nodes have equal cost in a tuple comparison
        return self.cost < other.cost
    
    def compute_how_many_fixed_agent_in_each_node(self, id_agent = None):
        """
        Counts how many agents pass through each expanded node in the current solution.
        Used both to build the congestion_dict for the low-level planner and to
        compute g_c for the high-level focal search.
        Indeed, it is a metrics that helps in computing approximatly the number of vertex constraints violated
        Args:
            id_agent: if provided, excludes this agent from the count.
                      Used when replanning agent i: counts congestion from all other agents (used for focal search in the low level).
                      If None, counts all agents (used for g_c computation).
        Returns:
            dict {expanded_node_id: number of agents passing through it}
        """
        dict_number_agent_in_node = {}
        for agent_id, path in self.solution.items():  # self.solution, non solution
            if (id_agent is not None) and  (agent_id == id_agent):
                continue
            for node in path:
                dict_number_agent_in_node[node] = dict_number_agent_in_node.get(node, 0) + 1
        return dict_number_agent_in_node
    
    def update_constr_in_node_for_agent(self, vertex_constr, edge_constr, agent_id):
        """
        Initialises this node's constraints by copying the parent's constraints
        and adding one new constraint for the specified agent.
        Called immediately after child node creation in high level CBS.
        Note: edge constraints here are expressed as expanded node id pairs (src, dst).
        The TEG builder handles swap-pair symmetry automatically when the TEG is constructed.
        Args:
            vertex_constr: expanded node id to forbid for agent_id, or None
            edge_constr:   (src, dst) expanded edge pair to forbid for agent_id, or None
            agent_id:      agent to whom the new constraint applies
        """
        # copy parent constraints and add the new one for this child
        new_vertex_constr_diz = {k: set(v) for k, v in self.parent.vertex_constraints.items()}
        new_edge_constr_diz = {k: set(v) for k, v in self.parent.edge_constraints.items()}

        # initialise sets for this agent if not already present
        if agent_id not in new_vertex_constr_diz:
            new_vertex_constr_diz[agent_id] = set()
        if agent_id not in new_edge_constr_diz:
            new_edge_constr_diz[agent_id] = set()

        if vertex_constr is not None:
            new_vertex_constr_diz[agent_id].add(vertex_constr)

        if edge_constr is not None:
            new_edge_constr_diz[agent_id].add(edge_constr)
        self.vertex_constraints = new_vertex_constr_diz
        self.edge_constraints = new_edge_constr_diz

class BCBSSolver_mapd:
    """
    Bounded Suboptimal CBS (BCBS) solver using focal search at both levels.
    Based on: Sharon et al., "Conflict-Based Search for Optimal Multi-Agent Path Finding"
    and the BCBS extension described in the same literature.
    Taking into consideration:
    - Starting time can be different between agents
    - There are external constraints to take into consideration

    Two-level structure:
    HIGH LEVEL — constraint tree (CT) with focal search:
      OPEN:  all CT nodes, ordered by sum-of-costs.
      FOCAL: subset of OPEN with cost <= w_h * cost_min,
             ordered by g_c (conflict heuristic).
      The node with lowest g_c in focal is expanded, favouring solutions with
      fewer conflicts and expected to converge faster.

    LOW LEVEL — single-agent planning with focal A*:
      For each agent, builds a TEG with that agent's constraints and
      runs a_star_with_focal_search with factor w_l, guided by a
      congestion_dict built from the other agents' paths.

    Parameters:
      w_l: low-level suboptimality factor  (w_l=1 -> optimal low-level)
      w_h: high-level suboptimality factor (w_h=1 -> optimal high-level)

    Special cases:
      BCBS(1, 1) — equivalent to CBS in solution cost
      BCBS(w, w) — BCBS(w_H, w_L) with equal factors

    Guarantees:
      - Conflict-free: yes
      - Complete: yes
      - Cost bound: sum-of-costs <= w_h * w_l * C*
    """

    def __init__(self, teg:TimeExpandedGraph, w_l=1, w_h=1,conflict_heuristic_high_l="h3", conflict_heuristic_low_l="h3", diz_constr_vertex: dict = None, diz_constr_edge: dict = None):
        """
        Args:
            G_original: original spatial graph (not time-expanded)
            T: time horizon = length of the time-expanded graph
            w_l: low-level suboptimality factor  (default 1: optimal)
            w_h: high-level suboptimality factor (default 1: optimal)
            conflict_heuristic: heuristic function for approximate the goodness of the solution in terms of feasibility (no conflicts)
            diz_constr_vertex_obstacle: rapresent external obstacle in terms of vertex constraint (expanded) for each agent
            diz_constr_edge_obstacle: rapresent external obstacle in terms of edge constraint (expanded) for each agent

        """
        self.teg = teg   # "vuoto" conflitti gestiti internamente
        self.max_expanded_nodes = 10000
        self.expanded_nodes = 0
        self.w_l = w_l
        self.w_h = w_h
        self.conflict_heuristic_L_level = conflict_heuristic_low_l   
        self.conflict_heuristic_H_level = conflict_heuristic_high_l 
        self.diz_constr_vertex_obstacle = diz_constr_vertex or {}
        self.diz_constr_edges_obstacle  = diz_constr_edge   or {}


    def plan(self, diz_start_and_goal:dict, diz_time_start:dict) -> dict:
        """
        Finds a conflict-free bounded-suboptimal plan for all agents using BCBS.
        Args:  diz_start_and_goal: {agent_id: (start_node, goal_node)}
               diz_time_start: {agent_id: start_time}
        Returns: (solution, True) solution is {agent_id: expanded-node path} if a conflict-free plan is found. ({}, False) if no feasible solution is found or the expansion limit
                is exceeded.
        """
        # root node        
        teg_root = self.teg
        internal_to_orig = teg_root.internal_to_orig

        root = BCBSNode()
        diz_costs_for_agent = {}
        root_solution = {}

        # create root solution: considering external constraints 
        for agent_id in diz_start_and_goal.keys():
            start_node = diz_start_and_goal[agent_id][0]
            goal_node = diz_start_and_goal[agent_id][1]

            start_t = diz_time_start[agent_id]

            root.vertex_constraints[agent_id] = set()
            root.edge_constraints[agent_id]   = set()

            # add constraint to the agent
            if agent_id in self.diz_constr_vertex_obstacle:
                for constr in self.diz_constr_vertex_obstacle[agent_id]:
                    root.vertex_constraints[agent_id].add(constr)

            if agent_id in self.diz_constr_edges_obstacle:
                for constr in self.diz_constr_edges_obstacle[agent_id]:
                    root.edge_constraints[agent_id].add(constr)

            congestion_dict = {}  # empty at root — no other paths yet

            # compute solution with A* with focal search
            first_path, cost = self._plan_single_agent(start_node, start_t, goal_node,vertex_constraints=root.vertex_constraints[agent_id],edge_constraints=root.edge_constraints[agent_id],congestion_dict=congestion_dict)
            if first_path is None:
                return ({}, False)

            # update solutions
            root_solution[agent_id] = first_path
            diz_costs_for_agent[agent_id] = cost

        root.solution = root_solution
        root.cost_for_each_agent = diz_costs_for_agent
        root.cost = root._compute_cost()
        root.g_c = 0

        # OPEN list: all unexpanded CT nodes
        # implemented as a list [si potrebbe ottimizzare]
        list_open = [root]

        while list_open:

            # f_min: lowest cost among all nodes in OPEN
            # used to define the focal threshold
            f_min = min(node.cost for node in list_open)

            # build FOCAL (list with nodes in increasing order of g_c)
            focal = self._get_focal_list(list_open, f_min)

            # expand the node with lowest g_c from FOCAL
            node = focal.pop(0)
            list_open.remove(node)

            self.expanded_nodes = self.expanded_nodes + 1

            if self.expanded_nodes > self.max_expanded_nodes:
                result = ({}, False)
                return result

            # detect the first conflict in the current solution (si potrebbe cambiare quale conflitto pescare)
            conflict = self._detect_conflict(node.solution, diz_start_and_goal, diz_time_start, teg_root)

            if conflict is None:
                # no conflicts: solution is valid and within the cost bound
                return (node.solution, True)

            agent_tuple, vertex_constr, edge_constr = conflict

            # split: one child per agent involved in the conflict
            # each child inherits parent constraints plus one new constraint
            for agent_id in agent_tuple:

                # create child node
                child = BCBSNode(parent=node)
                child.update_constr_in_node_for_agent(vertex_constr, edge_constr, agent_id)    

                start_t = diz_time_start[agent_id]
                start_node = diz_start_and_goal[agent_id][0]
                goal_node = diz_start_and_goal[agent_id][1]
                
                # congestion_dict: how many other agents pass through each expanded node (apart from agent_id)
                # used by the low-level focal search to prefer less congested paths
                congestion_dict = node.compute_how_many_fixed_agent_in_each_node(agent_id)
                new_path, cost_sol = self._plan_single_agent(start_node, start_t, goal_node, vertex_constraints=child.vertex_constraints[agent_id], edge_constraints=child.edge_constraints[agent_id], congestion_dict=congestion_dict)

                # if no feasible path exists under the new constraints, skip this child
                if new_path is None:
                    continue

                # update child solution: copy parent solution and replace replanned agent
                new_solution = copy.deepcopy(node.solution)
                new_solution[agent_id] = new_path

                # update costs 
                new_costs_diz = copy.deepcopy(node.cost_for_each_agent)
                new_costs_diz[agent_id] = cost_sol

                child.solution = new_solution
                child.cost_for_each_agent = new_costs_diz

                # recompute cost
                child.cost = child._compute_cost()

                # generate cost (classic and heuristic) of solution in child node
                child.g_c = HEURISTICS_HIGH_L[self.conflict_heuristic_H_level](child.solution, self.teg.T, internal_to_orig)

                list_open.append(child)

        # heap exhausted
        result = ({}, False)
        return result

    def _get_focal_list(self, listt: list, f_min:float) -> list:
        """
        Builds the high-level FOCAL list from OPEN.
        Selects all CT nodes with cost <= w_h * f_min (within the suboptimality bound),
        then sorts them by (g_c, cost) so the least conflicting node is expanded first.

        Args:
            listt:  current OPEN list of BCBSNodes
            f_min:  minimum cost among all nodes in OPEN
        Returns:
            sorted list of BCBSNodes in FOCAL, ordered by (g_c, cost)
        """
        threshold = f_min * self.w_h
        focal_list = []
        for CBSnode in listt:
            if CBSnode.cost <= threshold:
                focal_list.append(CBSnode)
        focal_list.sort(key=lambda x: (x.g_c, x.cost))
        return focal_list


    def _plan_single_agent(self, start_node, start_t, goal_node, vertex_constraints: set, edge_constraints: set, congestion_dict: dict) -> tuple[Optional[list], Optional[float]]:
        """  
        Computes a path for a single agent under the specified constraints.
        A constrained Time Expanded Graph (TEG) is built for the agent and focal A* is executed with suboptimality factor w_l.
        The congestion dictionary contains occupancy information generated from the other agents' paths and is used by the focal heuristic to
        prefer less congested solutions when multiple paths have similarcost.

        Args: 
        start_node: Original graph start vertex.
        start_t: Departure timestep.
        goal_node: Original graph goal vertex.
        vertex_constraints: Forbidden expanded vertices for the agent.
        edge_constraints: Forbidden expanded transitions for the agent.
        congestion_dict: {expanded_node_id: number_of_other_agents}

        Returns:
        (path, cost) path is a list of expanded-node ids. cost is the sum of traversed edge costs.
        (None, None) if no feasible path exists.
        """
        teg = TimeExpandedGraph(self.teg.G_original, self.teg.T, vertex_constraints, edge_constraints)

        start_exp = teg.get_expanded_id_from_original_id( start_node, t=start_t)

        # A* with focal search
        path = a_star_with_focal_search(teg.G_expanded, start_exp, goal_node, congestion_dict = congestion_dict, conflict_heuristic = self.conflict_heuristic_L_level, extended=True, w = self.w_l )
        if path is None:
            return None, None

        # sum of costs as objective function
        cost_sol = 0
        for u, v in zip(path[:-1], path[1:]):
            cost_sol += teg.G_expanded[u][v]["distance"]

        return path, cost_sol


    def _detect_conflict(self, solution: dict, diz_start_and_goal: dict, diz_time_start: dict, teg: TimeExpandedGraph) -> Optional[tuple]:
        """
        Detects the first conflict between any pair of agents.

        Agents may have different start times; therefore conflicts are
        checked only during the time interval in which both agents are
        simultaneously active.

        Conflict types:
        - Vertex conflict: two agents occupy the same original vertex at the same timestep.
        - Edge conflict (swap): two agents traverse the same edge in opposite directions during the same timestep.

        Special handling of start positions:
        If one of the agents is still located at its initial vertex, the generated constraint is assigned only to the other agent.
        This prevents forbidding an agent from occupying its own start location at its departure time. This because initial nodes can be assigned outside the MAPF
        and in a MAPD solution the node start can't be changed (is the one taken from the previous planning)
        Args:  solution:  {agent_id: expanded-node path}
        diz_start_and_goal:  Agent start/goal information (original).
        diz_time_start: Agent release times.  
        """

        agent_ids = list(diz_start_and_goal.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                ai = agent_ids[i]
                aj = agent_ids[j]

                if ai not in solution or aj not in solution:  
                    continue
                
                path_i = solution[ai]
                path_j = solution[aj]

                # starting timestep (resto della divisione, vale per costruizione del tef)
                start_t_i = path_i[0] % teg.T
                start_t_j = path_j[0] % teg.T

                # timestep where both agent are active
                t_begin = max(start_t_i, start_t_j)
                t_end = min(start_t_i + len(path_i), start_t_j + len(path_j)) - 1

                if t_begin > t_end:
                    continue

                for t_abs in range(t_begin, t_end):
                    idx_i = t_abs - start_t_i
                    idx_j = t_abs - start_t_j

                    orig_i_t = teg.get_original_id_from_expanded(path_i[idx_i])
                    orig_j_t = teg.get_original_id_from_expanded(path_j[idx_j])
                    orig_i_t1 = teg.get_original_id_from_expanded(path_i[idx_i + 1])
                    orig_j_t1 = teg.get_original_id_from_expanded(path_j[idx_j + 1])

                    # vertex conflict
                    if orig_i_t == orig_j_t:
                        # se è il nodo di start di aj, il vincolo va ad ai
                        # se è il nodo di start di ai, il vincolo va ad aj 
                        # ( il vincolo va all'agente che non è al suo start)
                        if idx_j == 0:
                            # aj è al suo start, il vincolo va solo ad ai
                            return (ai,), path_i[idx_i], None
                        elif idx_i == 0:
                            # ai è al suo start, il vincolo va solo ad aj
                            return (aj,), path_j[idx_j], None
                        else:
                            return (ai, aj), path_i[idx_i], None

                    # edge conflict (swap)
                    if orig_i_t == orig_j_t1 and orig_j_t == orig_i_t1:
                        return (ai, aj), None, (path_i[idx_i], path_i[idx_i + 1])

                # check vertex conflict at last timestep
                last_idx_i = t_end - start_t_i
                last_idx_j = t_end - start_t_j
                orig_i_last = teg.get_original_id_from_expanded(path_i[last_idx_i])
                orig_j_last = teg.get_original_id_from_expanded(path_j[last_idx_j])
                if orig_i_last == orig_j_last:
                    if last_idx_j == 0:
                        return (ai,), path_i[last_idx_i], None
                    elif last_idx_i == 0:
                        return (aj,), path_j[last_idx_j], None
                    else:
                        return (ai, aj), path_i[last_idx_i], None

        return None

    
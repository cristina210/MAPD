from Simulator import Simulator
from extended_time_graph import TimeExpandedGraph
from MAPF_algorithm.solvers.CBS_mapd import CBSSolver_mapd
from MAPF_algorithm.solvers.BCBS_mapd import BCBSSolver_mapd
from MAPF_algorithm.solvers.PP_mapd import PrioritizedPlanner_mapd
import networkx as nx
import copy

from Simulator import Simulator
from extended_time_graph import TimeExpandedGraph
from MAPF_algorithm.solvers.CBS_mapd import CBSSolver_mapd
from MAPF_algorithm.solvers.BCBS_mapd import BCBSSolver_mapd
from MAPF_algorithm.solvers.PP_mapd import PrioritizedPlanner_mapd
import networkx as nx
import copy


'''
MAPD (multi agent picking and delivery overview):
Each agent has multiple waypoints organised in couples (picking node, delivery node) that need to be visited sequentially.
This is particularly relevant if considering also the assignement decision problem together with the routing problem which should take into
consideration this characteristic. Here, we consider only the routing part with sequence of nodes to be reached by agents already fixed in a higher level.
Therefore the MAPD become a planning mechanism that avoid conflicts and let each agent visit each waypoints. So it can be seen as a sort of sequence of MAPF:
start node --- MAPF ---> first waypoint  --- MAPF ---> second waypoint --- ... ---> last waypoint.
There are multiple perspective for implementing a MAPD, these are the choice:

From: State of the Art on: Multi-Agent Pickup and Delivery

Different methods to decompose the problem and to define the planning strategy have been proposed:
• Solving the MAPD problem by knowing all the tasks in advance and working in an offline setting where the
solution is computed once and does not change during the execution of the MAPD instance.
• Decomposing the MAPD problem into a sequence of MAPF instances where all paths are replanned at every
time step to take into account the new tasks.
• Decomposing the MAPD problem into a sequence of MAPF instances where the path replanning is performed
only for some of the agents at each time step, for example only for the agents that have reached their current
goal location or in the most extreme scenario for one agent at the time 

So far this implementation reflect the last situation: 
this implementation is a sequential/real time (missions with waypoints arrives during time) and partial replanning MAPD (only the agent with new missions are planned and the planning starts from the last one).
To do so a sort of "windowed"/rolling-horizon aspect is introduced: MAPF solvers are called multiple times and when new MAPF paths are built only the part of this paths up to the first waypoint event is taken as final.
A new MAPF starts from this first time.
Indeed, only the agents listed in `missions` (the ones with a new task to execute) are (re)planned by compute_paths(). All other agents are NOT touched: their already
committed _progress is treated as a fixed trajectory and injected as external vertex/edge constraints (_extract_constr) for the agents being planned.
This class should be called with plan() as soon as new missions arrive and works on the preexisting situation. 

In particular:

Within the subset of agents being planned, the problem is decomposed into a SEQUENCE of MAPF instances (not a single offline solve, not a full per-timestep
replan of everyone):
  - At each iteration of the main while loop, a fresh Time-Expanded Graph (TEG) is built with a horizon T estimated from the agents' current shortest-path
    distances plus a congestion margin (_makespan_shortest_path).
  - The MAPF solver (PP/CBS/BCBS) is run ONCE on this TEG for ALL agents still being planned, jointly producing conflict-free paths toward their current
    intermediate waypoint, subject to:
      (a) constraints from agents not in `missions` (their full future trajectory, already in _progress, is fixed), and
      (b) constraints from agents in `missions` that already have a partially committed prefix in _progress - built by a previous MAPD for example - (only that prefix is fixed; the part
          from their own shifted start time onward is the solver MAPF's own decision).
  - Asynchronous starts are handled via a per-agent shift: t=0 in the TEG corresponds to the global time t_min, and each agent's own start is
    shifted relative to t_min (diz_time_start_mapd_shifted_for_agent). This is choosen for avoid waste of memory.
  - From the jointly-planned solution, only a prefix of each agent's new segment is committed to _progress — up to t_min_local, the earliest TEG
    time at which ANY of the planned agents reaches its current intermediate waypoint (steps_committed). This is the "windowed"/rolling-horizon aspect:
    the full joint plan is recomputed, but only the part up to the first waypoint event is taken as final.
  - After committing, the loop updates each agent's (start, goal) pair —  advancing to the next waypoint for agents that arrived, or just updating
    the start position for agents still mid-segment — and re-extracts constraints for the next iteration. The loop repeats until every agent in
    `missions` has visited all its waypoints (diz_flag all False).

In summary: MAPD decomposed as a sequence of joint MAPF sub-problems over a sliding/rolling time window, restricted to a subset of agents (those with new
missions), with all other agents' committed trajectories treated as moving obstacles.
'''

# ANCORA DA VEDERE:
# 1) come vincoli considerare anche gli amr fermi in un nodo (quindi non presenti in progress ma solo in amr_position) -> mettere eventualmente in extract_constr
# importante nel caso in cui a un amr fermo a un certo punto venga assegnata una missione ma altri amr hanno già pianificato di passare per il suo nodo start (ovvero la sua posizione)
# a quel punto penso si blocchi: non posso imporre dei vincoli sullo start (in extract evito) e anche in CBS evito. Questo perchè lo start lo vedo come non pianificabile
# sia perchè fa parte di progress quindi della pianificazione passata sia perchè lo start può essere forzato esternamente
# 2) Controllare il caso in cui il primo waypoint è uguale allo start (penso basta mettere un controllo iniziale di path insieme a quando faccio il controllo che la lista waypoint sia almeno lunga 2 e fare continue)
# -> in realtà sembra che il MAPF lo gestisca in automatico con percorso pari a 0.

class MAPD_solver:
    """
    Initializes the MAPD solver.

    Args:
        sim: Simulator instance containing graph, agents positions and progress.
        solver_name:  Name of the MAPF solver to use ("PP", "CBS", "BCBS").
        solver_kwargs: Additional parameters passed to the selected solver.
    """

    SOLVER_REGISTRY = {
        "PP": PrioritizedPlanner_mapd,
        "CBS": CBSSolver_mapd,
        "BCBS": BCBSSolver_mapd,
    }

    def __init__(self, sim: Simulator, solver_name: str = "CBS", solver_kwargs: dict = None):
        self.sim = sim

        if solver_name not in self.SOLVER_REGISTRY:
            raise ValueError(
                f"solver_name should be in {list(self.SOLVER_REGISTRY.keys())}, "
                f"received: {solver_name}"
            )

        self.solver_class = self.SOLVER_REGISTRY[solver_name]
        self.solver_kwargs = solver_kwargs if solver_kwargs is not None else {}

    def compute_paths(self, missions):
        """
        Computes paths for all agents until all missions are completed.

        Args:
            missions: List of tuples: (agent_id, waypoint_list). waypoint_list contains
            the sequence of goals that the agent has to visit. The first node in
            waypoint_list is the current position of the agent (start node of this MAPD call).
        Returns:
            True if all paths are successfully computed, False if no collision-free solution exists.
        """
        ### Initialization
        diz_waypoints_for_agent = {}
        diz_flag = {}
        diz_time_start_mapd_for_agent = {}
        diz_couple_next_start_goal_node = {}
        diz_waypoints_to_visit = {}
        t_starts = []

        for tupla in missions:
            agent_id = tupla[0]
            if len(tupla[1]) <= 1:
                # only the starting node, nothing to plan
                diz_flag[agent_id] = False
                continue

            diz_waypoints_for_agent[agent_id] = tupla[1]
            diz_flag[agent_id] = True

            if len(self.sim._progress.get(agent_id, [])) > 0:
                # agent already has a scheduled path: new segment starts after the last scheduled node
                diz_time_start_mapd_for_agent[agent_id] = len(self.sim._progress[agent_id])
                t_starts.append(len(self.sim._progress[agent_id]))
                start_node = self.sim._progress[agent_id][-1]
            else:
                # first MAPD call, or agent that just finished its previous plan
                self.sim._progress[agent_id] = []
                diz_time_start_mapd_for_agent[agent_id] = 0
                t_starts.append(0)
                start_node = self.sim.amr_positions[agent_id]

            goal_node = diz_waypoints_for_agent[agent_id][1]
            diz_couple_next_start_goal_node[agent_id] = (start_node, goal_node)
            diz_waypoints_to_visit[agent_id] = 1

        ### Shift start times
        # t_min = earliest absolute start time among all active agents
        # t=0 in the TEG corresponds to absolute time t_min
        t_min = min(t_starts)
        diz_time_start_mapd_shifted_for_agent = {}
        for agent_id in diz_flag:
            if not diz_flag[agent_id]:
                continue
            diz_time_start_mapd_shifted_for_agent[agent_id] = diz_time_start_mapd_for_agent[agent_id] - t_min

        ### Main loop: replan every time some agent reaches its next waypoint
        flag_end = False
        while not flag_end:

            # time horizon for this iteration's TEG
            T_makespan = self._makespan_shortest_path(diz_couple_next_start_goal_node, max(t_starts)) + 10

            teg = TimeExpandedGraph(G=self.sim.graph, T=T_makespan)

            # external constraints from agents already scheduled 
            edge_constr, vertex_constr = _extract_constr(
                self.sim, t_min, diz_time_start_mapd_shifted_for_agent,
                teg.internal_to_expanded, teg.orig_to_internal
            )

            # run MAPF for all agents that still need a path
            result = self.solver_class(
                teg=teg,
                diz_constr_vertex=copy.deepcopy(vertex_constr),
                diz_constr_edge=copy.deepcopy(edge_constr),
                **self.solver_kwargs
            ).plan(diz_couple_next_start_goal_node, diz_time_start_mapd_shifted_for_agent)

            paths, success = result
            if not success:
                return False

            # convert expanded ids to original node ids
            paths_orig = {}
            for agent_id, path_exp in paths.items():
                paths_orig[agent_id] = [teg.internal_to_orig[teg.get_internal_id_from_expanded(n, teg.T)]  for n in path_exp ]

            # len_paths[agent_id] = number of steps of the NEW segment (path length minus start node)
            len_paths = {}
            for agent_id in paths_orig:
                len_paths[agent_id] = len(paths_orig[agent_id]) - 1

            # arrival_teg_time[agent_id] = TEG time at which agent_id reaches its waypoint
            # = its own shifted start time + the length of its new segment
            arrival_teg_time = {}
            for agent_id in paths_orig:
                arrival_teg_time[agent_id] = diz_time_start_mapd_shifted_for_agent[agent_id] + len_paths[agent_id]  # time (relative to t_min) when agent arrive in the next waypoint

            # t_min_local = earliest TEG time at which some agent reaches its waypoint
            t_min_local = min(arrival_teg_time.values())

            # Now, we consider as path only the path of all the agent from their start to t_min_local
            # So the steps taken in the path are equal to:
            # min(t_min_local - shifted start, len(path)) if shifted start < t_min_local
            # otherwise it means that the agent didn't start the new path before the other agent (the one with t_min) arrives at its next waypoiny
            # 0 if shifted start > t_min_local
            # steps = max(t_min_local - start, 0)

            # steps_committed[agent_id] = how many new steps of agent_id's segment
            steps_committed = {}
            for agent_id in paths_orig:
                shifted_start = diz_time_start_mapd_shifted_for_agent[agent_id]
                # steps = min(len_paths[agent_id], t_min_local - shifted_start)  dovrebbe non servire per come è definito t_min_local
                # steps_committed[agent_id] = max(0, steps)
                steps_committed[agent_id] = max(0, t_min_local - shifted_start)

            # save the committed steps into _progress
            for agent_id, path_orig in paths_orig.items():
                n = steps_committed[agent_id]
                for node_orig in path_orig[1: n + 1]:
                    self.sim._progress[agent_id].append(node_orig)

            # advance the global time
            t_min = t_min + t_min_local

            # update waypoint pointer and next goal for each active agent
            for agent_id in list(diz_waypoints_for_agent.keys()):
                if not diz_flag[agent_id]:
                    continue
                if steps_committed[agent_id] == len_paths[agent_id]:
                    # the full new segment was committed: agent reached its waypoint
                    if diz_waypoints_to_visit[agent_id] < len(diz_waypoints_for_agent[agent_id]) - 1:
                        diz_waypoints_to_visit[agent_id] += 1
                        new_start_node = paths_orig[agent_id][-1]
                        new_end_node = diz_waypoints_for_agent[agent_id][diz_waypoints_to_visit[agent_id]]
                        diz_couple_next_start_goal_node[agent_id] = (new_start_node, new_end_node)
                    else:
                        # agent visited all its waypoints
                        diz_flag[agent_id] = False
                        diz_couple_next_start_goal_node.pop(agent_id)
                        diz_time_start_mapd_shifted_for_agent.pop(agent_id)
                else:
                    # segment not finished yet: same goal, new start = last committed position
                    new_start_node = paths_orig[agent_id][steps_committed[agent_id]]
                    diz_couple_next_start_goal_node[agent_id] = (new_start_node,diz_couple_next_start_goal_node[agent_id][1])

            # recompute shifted start time for the next iteration
            # new_shifted_start =
            #  old_shifted_start  (old shift time relatively to t_min)
            # + steps_committed  (how many steps committed so how many nodes added to progress in relative term of t_min)
            # - t_min_local  ( new time is shifted from t_min to t_min + t_min_local)
            for agent_id in diz_time_start_mapd_shifted_for_agent:
                old_shifted_start = diz_time_start_mapd_shifted_for_agent[agent_id]
                diz_time_start_mapd_shifted_for_agent[agent_id] = old_shifted_start + steps_committed[agent_id] - t_min_local

            t_starts = list(diz_time_start_mapd_shifted_for_agent.values())

            # stop when all agents have visited all their waypoints
            flag_end = all(not v for v in diz_flag.values())

        return True

    def _makespan_shortest_path(self, diz_couple_next_start_goal_node, t_max):
        """
        Estimates the time horizon needed for the Time Expanded Graph.

        Args:
            diz_couple_next_start_goal_node: Dictionary {agent_id: (current_position, next_goal)}, original ids.
            t_max: Maximum shifted start time among agents.

        Returns:
            Estimated makespan used as T horizon.
        """
        max_sp = 0
        for agent_id in diz_couple_next_start_goal_node:
            sp_len = nx.shortest_path_length(
                self.sim.graph,
                diz_couple_next_start_goal_node[agent_id][0],
                diz_couple_next_start_goal_node[agent_id][1],
            )
            max_sp = max(max_sp, sp_len)

        delta = len(self.sim.graph.nodes) - self.sim.N_AMR
        congestion = round(len(self.sim.graph.edges) / delta) if delta > 0 else self.sim.N_AMR
        return max(1, max_sp + congestion + t_max)


def _extract_constr(sim, t_min, diz_time_start_x_agent, internal_to_expanded, orig_to_internal):
    """
    Extracts vertex and edge constraints caused by already scheduled agents.

    Two kinds of "already scheduled" agents are considered:
    - agents not in diz_time_start_x_agent: not being planned in this MAPD, their whole _progress is fixed and acts as a constraint for everyone.
    - agents in  diz_time_start_x_agent (also being planned this iteration):
      only the portion of their _progress before their own shifted start time is fixed and acts as a constraint for the others (their progress has already path scheduled). From their shifted
      start time onward, their position is given by their new path, which is part of the solver's own solution and handled by its conflict detection.

    Args:
        sim: Simulator containing current agent trajectories.
        t_min: Current global simulation timestep (corresponds to t=0 in the TEG).
        diz_time_start_x_agent: Dictionary {agent_id: shifted start time in current TEG}.
        internal_to_expanded: Mapping internal node -> time expanded node.
        orig_to_internal: Mapping original graph node -> internal node.

    Returns:
        edge_constr: Dictionary of forbidden transitions per agent.
        vertex_constr: Dictionary of forbidden nodes per agent.
    """
    edge_constr = {}
    vertex_constr = {}

    for agent_id in diz_time_start_x_agent:
        vertex_constr[agent_id] = set()
        edge_constr[agent_id] = set()

    for agent_id_to_plan in diz_time_start_x_agent:

        t_start_this_agent = t_min + diz_time_start_x_agent[agent_id_to_plan]

        for other_agent_id in sim._progress:

            if other_agent_id == agent_id_to_plan:
                # an agent cannot be an obstacle for itself
                continue

            other_is_planned = other_agent_id in diz_time_start_x_agent  # check if is involved in the planning
            if other_is_planned:
                shifted_start_other = diz_time_start_x_agent[other_agent_id]

            for t_abs in range(len(sim._progress[other_agent_id])):  
                t_assoluto = t_abs + 1
                # positions before agent_id_to_plan's own start are irrelevant for it
                if t_assoluto < t_start_this_agent:
                    continue
                t_teg = t_assoluto - t_min  # t_min is time 0 in the teg

                # t_teg == 0 is now: the current configuration is fixed
                if t_teg == 0:
                    continue

                # time outside the one considered in the teg
                if t_teg >= len(internal_to_expanded[0]):
                    continue

                # not considering segment of path of agent which are in missions so needs to be rescheduled but the time considered is after their start
                # we consider constraints only the previous path already planned (in a previous MAPD) that now is already in _progress
                if other_is_planned and t_teg >= shifted_start_other:
                    # from here on, other_agent_id follows its NEW path,
                    # already handled by the solver's conflict detection
                    continue

                # vertex constraint: other_agent_id occupies this node at this TEG time
                node_orig = sim._progress[other_agent_id][t_abs]
                int_id = orig_to_internal[node_orig]
                node_exp = internal_to_expanded[int_id][t_teg]
                vertex_constr[agent_id_to_plan].add(node_exp)

                # edge constraint: forbid moving into node_exp from its predecessor
                if t_abs > 0:
                    prev_orig = sim._progress[other_agent_id][t_abs - 1]
                else:
                    prev_orig = sim.amr_positions[other_agent_id]

                prev_int = orig_to_internal[prev_orig]
                node_exp_prev = internal_to_expanded[prev_int][t_teg - 1]
                edge_constr[agent_id_to_plan].add((node_exp_prev, node_exp))

    return edge_constr, vertex_constr
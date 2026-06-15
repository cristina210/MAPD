from instance.Network_graph import NetworkGraph
from fleet_utils.fleet import Fleet
from extended_time_graph import TimeExpandedGraph
from shortest_path_algorithm.A_star import a_star



class PrioritizedPlanner_mapd:
    """
    Prioritized Planning MAPF is a MAPF solver working on a time-expanded graph.
    Agents are planned one by one in a specific order (in this implementation the order is the in fleet.agents)
    Each planned path is converted in constraints related to nodes or edges for the following agents guaranteeing the absence of conflicts.


    Note: quality of solution depends on the processing order. 
    Nota: completeness and optimality are not guaranteed.
    """

    def __init__(self, teg, diz_constr_vertex, diz_constr_edge):
        """
        Args:
            teg: Extended time graph (empty)
            diz_constr_vertex: rapresent external obstacle in terms of vertex constraint (expanded) for each agent
            diz_constr_edge: rapresent external obstacle in terms of edge constraint (expanded) for each agent
        """
        self.vertex_constraints = diz_constr_vertex
        self.edge_constraints = diz_constr_edge
        self.teg = teg


    def plan(self, diz_start_and_goal: dict, diz_time_start: dict) -> tuple:
        ''' 
        Planning paths for agent wih PP taking into consideration:
        - Starting time can be different between agents
        - There are external constraints to take into consideration
        Args:
        diz_start_and_goal {agent_id: (original start_node, original goal_node)},
        diz_time_start {agent_id: start_t} -> (dict {agent_id: path}, bool)
        '''
        result = {}
        list_agent = list(diz_start_and_goal.keys())  # lista di agent_id

        # --- pre-blocco: ogni agente non può passare per il nodo di start
        # di un altro agente esattamente al suo start_t, indipendentemente
        # dall'ordine di pianificazione ---
        for agent_id in list_agent:
            for other_agent_id in list_agent:
                if other_agent_id == agent_id:
                    continue
                other_start_node = diz_start_and_goal[other_agent_id][0]
                other_start_t = diz_time_start[other_agent_id]
                other_start_exp = self.teg.get_expanded_id_from_original_id(other_start_node, t=other_start_t)
                self.vertex_constraints[agent_id].add(other_start_exp)

        for idx, agent_id in enumerate(list_agent):
            start_node = diz_start_and_goal[agent_id][0]
            goal_node  = diz_start_and_goal[agent_id][1]
            start_t    = diz_time_start[agent_id]

            teg = TimeExpandedGraph(self.teg.G_original, self.teg.T,  self.vertex_constraints[agent_id], self.edge_constraints[agent_id] )
            start_exp = teg.get_expanded_id_from_original_id(start_node, t=start_t)


            path = a_star(teg.G_expanded, start_exp, goal_node, extended=True)

            if path is None:
                return ({}, False)

            result[agent_id] = path

            nodes_constr = set(path)
            edges_constr = {(path[j], path[j + 1]) for j in range(len(path) - 1)}

            for future_agent_id in list_agent[idx + 1:]:
                self.vertex_constraints[future_agent_id].update(nodes_constr)
                self.edge_constraints[future_agent_id].update(edges_constr)

        return (result, True)
        
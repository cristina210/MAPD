import logging
import networkx as nx
import matplotlib.pyplot as plt
from instance.Network_graph import NetworkGraph


class Simulator():
    def __init__(self, graph: NetworkGraph, initial_positions):
        self.graph = graph
        self.N_AMR = len(initial_positions)
        self.amr_positions = list(initial_positions)
        self.initial_positions = initial_positions
        self._progress: dict[int, list] = {}  # 0-based amr_id -> remaining nodes to visit
        self.step_num: int = 0
        logging.info("Step %d — AMR positions: %s", self.step_num, self.amr_positions)

    def reset(self):
        self.amr_positions = list(self.initial_positions)

    def step(self, paths: dict = None) -> tuple[int, list[bool]]:
        """Execute one simulation step.

        Returns (step_num, working_status) where working_status[i] is True if
        AMR i still has nodes to visit, False if it has completed its path.

        paths: optional {amr_id: list_of_nodes}. When provided, the new path is
        appended to any already-queued nodes for that AMR (the first node of the
        new path is skipped because it is the AMR's current position).
        """
        # if there are path
        if paths:
            # concatenate them
            for amr_id, path in paths.items():
                remaining = list(path[1:])  # path[0] is the current position
                if amr_id in self._progress:
                    self._progress[amr_id].extend(remaining)
                else:
                    self._progress[amr_id] = remaining
        # In anycase, run the step
        for amr_id in list(self._progress):
            remaining = self._progress[amr_id]
            self.amr_positions[amr_id] = remaining.pop(0)
            if not remaining:
                del self._progress[amr_id]
        # update the time
        self.step_num += 1
        logging.info("Step %d — AMR positions: %s", self.step_num, self.amr_positions)
        # Check for conflicts
        positions = [p for p in self.amr_positions if p is not None]
        if len(positions) != len(set(positions)):
            #self._conflict_management()
            pass
        # working_status[i] = True if AMR i still has pending nodes
        working_status = [i in self._progress for i in range(self.N_AMR)]
        return self.step_num, working_status

    def time_to_end_work(self, amr_id: int) -> float:
        """Estimate the time to end all the associated missions of amr_id (0-based)."""
        return len(self._progress[amr_id])

    def _conflict_management(self):
        logging.info("   Conflict detected")
        from collections import defaultdict

        position_to_amrs: dict[int, list[int]] = defaultdict(list)
        for amr_id, pos in enumerate(self.amr_positions):
            if pos is not None:
                position_to_amrs[pos].append(amr_id)

        for conflict_node, amr_ids in position_to_amrs.items():
            if len(amr_ids) <= 1:
                continue

            # Keep the AMR with the lowest id; reroute all others
            to_reroute = sorted(amr_ids)[1:]
            logging.info("   Conflict at node %s — keeping AMR %d, rerouting %s",
                         conflict_node, sorted(amr_ids)[0], to_reroute)

            # Graph without the blocked node so the rerouted AMRs cannot pass through it
            temp_G = self.graph.copy()
            temp_G.remove_node(conflict_node)

            for amr_id in to_reroute:
                if amr_id not in self._progress:
                    continue  # AMR already done, nothing to reroute

                target = self._progress[amr_id][-1]

                # Try every successor of conflict_node as the first hop
                best_path: list | None = None
                best_cost = float("inf")
                for nb in self.graph.successors(conflict_node):
                    if nb not in temp_G:
                        continue
                    try:
                        sub = nx.shortest_path(temp_G, nb, target, weight="distance")
                        cost = self.graph[conflict_node][nb]["distance"] + sum(
                            temp_G[sub[i]][sub[i + 1]]["distance"]
                            for i in range(len(sub) - 1)
                        )
                        if cost < best_cost:
                            best_cost = cost
                            best_path = sub  # [nb, ..., target]
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

                if best_path is not None:
                    self._progress[amr_id] = best_path
                    logging.info("   AMR %d rerouted via %s (cost %.2f)", amr_id, best_path, best_cost)
                else:
                    logging.warning("   AMR %d: no alternative path from %s to %s",
                                    amr_id, conflict_node, target)

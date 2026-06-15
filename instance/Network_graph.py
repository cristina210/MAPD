from enum import Enum
import networkx as nx


class NodeType(str, Enum):
    CHARGING_STATION = "charging_station"
    ACTION = "action"
    PARKING = "parking"
    INTERSECTION = "intersection"


class NetworkGraph(nx.DiGraph):
    """Directed graph that tracks semantic node types (charging station, picking, parking, intersection).

    The constructor mirrors nx.DiGraph so any existing graph data or keyword
    attributes can be forwarded directly — keeping the class open to future
    data sources beyond YAML.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    def nodes_by_type(self, node_type) -> list:
        """Return all node ids whose ``node_type`` attribute matches *node_type*."""
        type_val = node_type.value if isinstance(node_type, NodeType) else node_type
        return [n for n, d in self.nodes(data=True) if d.get("node_type") == type_val]

    def nearest_charging_station(self, node, weight: str | None = None):
        """Return the id of the charging station closest to *node* by shortest path.

        Parameters
        ----------
        node:
            A node id that exists in the graph.
        weight:
            Edge attribute to use as cost (e.g. ``"distance"``).
            ``None`` means hop count.

        Returns ``None`` when no charging station is reachable.
        """
        stations = self.nodes_by_type(NodeType.CHARGING_STATION)
        if not stations:
            return None

        best_node = None
        best_dist = float("inf")
        for cs in stations:
            try:
                dist = nx.shortest_path_length(self, node, cs, weight=weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if dist < best_dist:
                best_dist = dist
                best_node = cs
        return best_node

from fleet_utils.fleet import Fleet
from instance.Network_graph import NetworkGraph, NodeType

def make_grid_graph(rows: int, cols: int, step: float = 1.0) -> NetworkGraph:
    """
    Builds a rows for cols grid graph where each node holds (x, y) coordinates
    and every cell is connected to its right and upper neighbours via
    bidirectional edges.

    Args:
        rows:  number of rows in the grid
        cols:  number of columns in the grid
        step:  distance between adjacent nodes (default 1.0, standard in MAPF)

    Returns:
        G: the constructed NetworkGraph
    """
    G = NetworkGraph()

    # Node creation
    # Each node gets a linear id  nid = r*cols + c  and spatial attributes x, y
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            attrs = {"x": float(c * step), "y": float(r * step)}
            G.add_node(nid, **attrs)

    # Edge creation 
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c

            # Horizontal edge (only if the next column exists)
            if c + 1 < cols:
                right = r * cols + (c + 1)
                G.add_edge(nid, right, distance=step)
                G.add_edge(right, nid, distance=step)

            # Vertical edge (only if the next row exists)
            if r + 1 < rows:
                up = (r + 1) * cols + c
                G.add_edge(nid, up, distance=step)
                G.add_edge(up, nid, distance=step)

    return G




import json
import math
import xml.etree.ElementTree as ET
import yaml


def _infer_node_type(name: str, attrs: dict) -> NodeType:
    if attrs.get("is_charger"):
        return NodeType.CHARGING_STATION
    if attrs.get("is_parking"):
        return NodeType.PARKING
    if name:
        return NodeType.ACTION
    return NodeType.INTERSECTION


def load_map_graph_yaml(yaml_path: str, level: str = "L1") -> NetworkGraph:
    """Read a toy_map.yaml file and return a NetworkGraph.

    Nodes carry attributes: x, y, name, node_type, plus any extra vertex attrs.
    Edges carry: distance (Euclidean).
    Bidirectional lanes produce two directed edges.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    level_data = data["levels"][level]
    vertices = level_data["vertices"]
    lanes = level_data["lanes"]

    G = NetworkGraph()

    for idx, vertex in enumerate(vertices):
        x, y, _, name = vertex[0], vertex[1], vertex[2], vertex[3]
        raw_attrs = vertex[4] if len(vertex) > 4 else {}
        node_attrs = {"x": x, "y": y, "name": name}
        for k, v in raw_attrs.items():
            node_attrs[k] = v[1] if isinstance(v, list) and len(v) == 2 else v
        node_attrs["node_type"] = _infer_node_type(name, node_attrs).value
        G.add_node(idx, **node_attrs)

    for lane in lanes:
        src, dst, lane_attrs = lane[0], lane[1], lane[2]
        bidirectional = lane_attrs.get("bidirectional", [4, False])
        is_bidi = bidirectional[1] if isinstance(bidirectional, list) else bidirectional

        x1, y1 = G.nodes[src]["x"], G.nodes[src]["y"]
        x2, y2 = G.nodes[dst]["x"], G.nodes[dst]["y"]
        dist = math.hypot(x2 - x1, y2 - y1)

        G.add_edge(src, dst, distance=dist)
        if is_bidi:
            G.add_edge(dst, src, distance=dist)

    return G


def _infer_node_type_xml(is_chargeable: bool, road_property: int) -> NodeType:
    if is_chargeable:
        return NodeType.CHARGING_STATION
    if road_property == 0:
        return NodeType.PICKING
    return NodeType.INTERSECTION


def load_map_graph_xml(xml_path: str) -> NetworkGraph:
    """Read a MapCfg XML file and return a NetworkGraph.

    Each <PointInfo> becomes a node (id, x, y, node_type).
    Each <NeighbInfo> inside a <PointInfo> becomes a directed edge;
    when <Rever> is 1 a reverse edge is also added.
    Edges carry: distance.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    G = NetworkGraph()

    for point in root.findall("PointInfo"):
        node_id = int(point.findtext("id"))
        x = float(point.findtext("xpos"))
        y = float(point.findtext("ypos"))
        value_el = point.find("value")
        is_chargeable = value_el is not None and value_el.get("isChargeable", "0") == "1"
        road_property = int(point.findtext("RoadProperty") or 1)
        node_type = _infer_node_type_xml(is_chargeable, road_property)
        charge_rate_text = point.findtext("ChargeRate")
        charge_rate = float(charge_rate_text) if charge_rate_text is not None else None
        G.add_node(node_id, x=x, y=y, node_type=node_type.value, charge_rate=charge_rate)

    for point in root.findall("PointInfo"):
        src = int(point.findtext("id"))
        for nb in point.findall("NeighbInfo"):
            dst = int(nb.findtext("id"))
            distance = float(nb.findtext("distance") or 0.0)
            reverse = nb.findtext("Rever") == "1"
            G.add_edge(src, dst, distance=distance)
            if reverse:
                G.add_edge(dst, src, distance=distance)

    return G


def load_map_graph_json(json_path: str) -> NetworkGraph:
    """Read a graph JSON file (list of node dicts) and return a NetworkGraph.

    Each dict has: id, x, y, node_type, neighbors (list of neighbor IDs).
    Charging station nodes also carry charge_rate.
    Edge distances are computed as Euclidean distance between node positions.
    """
    with open(json_path, "r") as f:
        nodes = json.load(f)

    G = NetworkGraph()

    for node in nodes:
        if node["node_type"] == 'charging_station':
            node_type = NodeType.CHARGING_STATION
        elif node["node_type"] == 'parking':
            node_type = NodeType.PARKING
        elif node["node_type"] == 'intersection':
            node_type = NodeType.INTERSECTION
        else:
            node_type = NodeType.ACTION
        attrs = {"x": node["x"], "y": node["y"], "node_type": node_type.value}
        if node_type is NodeType.CHARGING_STATION:
            attrs["charge_rate"] = node.get("charge_rate")
        G.add_node(node["id"], **attrs)

    for node in nodes:
        src = node["id"]
        x1, y1 = G.nodes[src]["x"], G.nodes[src]["y"]
        for dst in node.get("neighbors", []):
            x2, y2 = G.nodes[dst]["x"], G.nodes[dst]["y"]
            dist = math.hypot(x2 - x1, y2 - y1)
            G.add_edge(src, dst, distance=dist)

    return G
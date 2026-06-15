import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import networkx as nx
from instance.Network_graph import NetworkGraph


def plot_graph(G: NetworkGraph) -> None:
    """
    Renders the graph using matplotlib/networkx:
    - nodes drawn in steel-blue with a name or id label
    - directed edges in grey with a slight curve so both directions
      of a bidirectional pair remain visible
    - edge labels showing speed_limit or speed when available
    """
    # Map each node to its (x, -y) position so that y increases upward
    # in the coordinate system but downward on screen (image convention)
    pos = {node: (data["x"], -data["y"]) for node, data in G.nodes(data=True)}

    # Use "name" attribute as label when present, otherwise fall back to the numeric id
    labels = {
        node: data.get("name") or str(node)
        for node, data in G.nodes(data=True)
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw nodes, labels, and edges separately for fine-grained control
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color="steelblue")
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_color="white")
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=True, arrowsize=15,
        edge_color="gray",
        connectionstyle="arc3,rad=0.1"  # slight arc to distinguish the two directions
    )

    # Edge labels: speed_limit takes priority over speed; empty string if neither is present
    edge_labels = {
        (u, v): str(d.get("speed_limit", d.get("speed", "")))
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)

    ax.set_title("MAPF Grid Graph")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def print_expanded_graph(G_expanded: NetworkGraph) -> None:
    """
    Prints the time-expanded graph to the terminal.
    Time-expanded graphs are common in MAPF formulations over a finite
    time horizon: each node represents a (original_node, timestep) pair,
    and edges are split into two categories:
      • "wait" – the agent stays in the same node at the next timestep
      • "move" – the agent moves to an adjacent node at the next timestep
    """
    print("Expanded graph")

    # --- Print nodes ---
    print("\nNodes:")
    for nid, attrs in G_expanded.nodes(data=True):
        # Show the expanded id, the original graph id, and the timestep
        print(f"  {nid}: (orig={attrs['original_id']}, t={attrs['t']})")

    # Print WAIT edges
    print("\nWAIT edges:")
    for src, dst, edge_attrs in G_expanded.edges(data=True):
        if edge_attrs.get("type_edge") == "wait":
            src_attrs = G_expanded.nodes[src]
            dst_attrs = G_expanded.nodes[dst]
            print(
                f"  ({src_attrs['original_id']},t={src_attrs['t']},id={src}) → "
                f"({dst_attrs['original_id']},t={dst_attrs['t']},id={dst})"
            )

    # Print MOVE edges 
    print("\nMOVE edges:")
    for src, dst, edge_attrs in G_expanded.edges(data=True):
        if edge_attrs.get("type_edge") == "move":
            src_attrs = G_expanded.nodes[src]
            dst_attrs = G_expanded.nodes[dst]
            print(
                f"  ({src_attrs['original_id']},t={src_attrs['t']},id={src}) → "
                f"({dst_attrs['original_id']},t={dst_attrs['t']},id={dst})"
            )

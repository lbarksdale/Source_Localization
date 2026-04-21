from igraph import Graph
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig

# Creates an Erdos-Renyi graph with a specified number of vertices
# Labels vertex 1 (source) with vtype=0, vertex 2 (observer) with vtype=1
def create_er_graph(num_vertices):
    p = 1.2 * np.log(num_vertices) / num_vertices
    er_graph = Graph.Erdos_Renyi(num_vertices, p)
    while not Graph.is_connected(er_graph):
        er_graph = Graph.Erdos_Renyi(num_vertices, p)
    vertex_types = np.zeros(num_vertices)
    er_graph.vs["vtype"] = vertex_types.tolist()
    er_graph.vs[1]["vtype"] = -1
    er_graph.vs[2]["vtype"] = 1
    return er_graph

def plot_graph(g, save=False, show_vtype=False):
    # Set the graph layout to something fairly nice for plotting
    Graph.layout_fruchterman_reingold(g)

    # Plot in matplotlib
    # Note that attributes can be set globally (e.g., vertex_size), or set individually using arrays (e.g., vertex_color)
    fig, ax = plt.subplots(figsize=(5, 5))
    if show_vtype:
        ig.plot(
            g,
            target=ax,
            vertex_size=10,
            vertex_frame_width=2.0,
            vertex_frame_color="white",
            edge_label=g.es["weight"],
            # vertex_label=g.vs["vertex_num"],
            vertex_color=["salmon" if vtype == -1 else "green" if vtype == 1 else "steelblue" for vtype in g.vs["vtype"]],
        )
    else:
        ig.plot(
            g,
            target=ax,
            vertex_size=10,
            vertex_frame_width=2.0,
            vertex_frame_color="white",
            edge_label=g.es["weight"]
        )

    if save:
        plt.savefig("sampleGraph.png")
    plt.show()

def get_random_spanning_tree(g, root=0):
    total_vertices = g.vcount()
    if total_vertices == 0:
        raise ValueError("Cannot build a spanning tree from an empty graph.")
    if not 0 <= root < total_vertices:
        raise ValueError(f"Root vertex {root} is out of range for a graph with {total_vertices} vertices.")
    if not g.is_connected():
        raise ValueError("A spanning tree can only be built from a connected graph.")

    spanning_tree = Graph(n=total_vertices, directed=g.is_directed())

    for attribute_name in g.vs.attributes():
        spanning_tree.vs[attribute_name] = list(g.vs[attribute_name])

    current_vertex = root
    visited_vertices = {root}

    while len(visited_vertices) < total_vertices:
        incident_vertices = g.neighbors(current_vertex)
        if not incident_vertices:
            raise ValueError(f"Vertex {current_vertex} has no neighbors in a graph marked connected.")

        random_vertex = int(np.random.choice(incident_vertices))
        if random_vertex not in visited_vertices:
            edge_id = g.get_eid(current_vertex, random_vertex, directed=g.is_directed(), error=False)
            edge_attributes = {}
            if edge_id != -1:
                edge_attributes = {
                    attribute_name: g.es[edge_id][attribute_name]
                    for attribute_name in g.es.attributes()
                }
            spanning_tree.add_edge(current_vertex, random_vertex, **edge_attributes)
            visited_vertices.add(random_vertex)

        current_vertex = random_vertex

    return spanning_tree

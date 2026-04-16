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

def plot_graph(g, save=False):
    # Set the graph layout to something fairly nice for plotting
    Graph.layout_fruchterman_reingold(g)

    # Plot in matplotlib
    # Note that attributes can be set globally (e.g., vertex_size), or set individually using arrays (e.g., vertex_color)
    fig, ax = plt.subplots(figsize=(5, 5))
    ig.plot(
        g,
        target=ax,
        vertex_size=10,
        vertex_frame_width=2.0,
        vertex_frame_color="white",
        vertex_color=["salmon" if vtype == -1 else "green" if vtype == 1 else "steelblue" for vtype in g.vs["vtype"]],
    )
    if save:
        plt.savefig("sampleGraph.png")
    plt.show()
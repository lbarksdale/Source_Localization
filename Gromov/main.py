import random

from igraph import Graph

from Graph_Basics import create_er_graph, plot_graph


def create_one_cycle_graph(num_vertices):
    gr = create_er_graph(num_vertices)
    g1 = Graph.spanning_tree(gr)
    all_edges = Graph.get_edgelist(gr)
    tree_edges = Graph.get_edgelist(g1)
    random_edge = random.choice(list(all_edges))
    while random_edge in tree_edges:
        random_edge = random.choice(list(all_edges))

    Graph.add_edge(g1, random_edge[0], random_edge[1])

    return g1

if __name__ == "__main__":
    g = create_one_cycle_graph(30)
    plot_graph(g)
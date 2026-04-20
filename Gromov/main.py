import random

import numpy as np
from igraph import Graph

from Graph_Basics import create_er_graph, plot_graph, get_random_spanning_tree
from Gromov.Gromov_Operations import get_gromov_matrix, g_convex_combination, get_distance


def create_one_cycle_graph(n_vertices):
    gr = create_er_graph(n_vertices)
    g1 = Graph.spanning_tree(gr)
    all_edges = Graph.get_edgelist(gr)
    tree_edges = Graph.get_edgelist(g1)
    random_edge = random.choice(list(all_edges))
    while random_edge in tree_edges:
        random_edge = random.choice(list(all_edges))

    Graph.add_edge(g1, random_edge[0], random_edge[1])

    return g1

if __name__ == "__main__":
    # Import diamond graph
    filename = "C:\\Users\\Levi Barksdale\\PycharmProjects\\ManuelResearch\\diamond.txt"
    g = Graph.Read_Adjacency(filename, attribute="weight", mode="undirected")
    # Add labels for the source and the observer
    num_vertices = g.vcount()
    vertex_types = np.zeros(num_vertices)
    g.vs["vtype"] = vertex_types.tolist()
    g.vs[0]["vtype"] = -1
    g.vs[1]["vtype"] = 1

    vertex_numbers = np.arange(num_vertices)
    g.vs["vertex_num"] = vertex_numbers.tolist()

    # Create a spanning tree and find Gromov matrix
    tree_root = 0
    plot_graph(g, show_vtype=True)

    random_spanning_tree = get_random_spanning_tree(g, tree_root)
    plot_graph(random_spanning_tree, show_vtype=True)
    gromov_1 = get_gromov_matrix(random_spanning_tree, tree_root)

    second_spanning_tree = get_random_spanning_tree(g, tree_root)
    plot_graph(second_spanning_tree, show_vtype=True)

    gromov_2 = get_gromov_matrix(second_spanning_tree, tree_root)
    print("Matrix 1:\n", gromov_1, "\nMatrix 2:\n", gromov_2, "\nGromov Combination:")

    combined_matrix = g_convex_combination(gromov_1, gromov_2, 0.5)

    print("G-Convex Combination:\n", combined_matrix)
    print("Time to infect: ", get_distance(combined_matrix, 0, 1))

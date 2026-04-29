import random

import matplotlib.pyplot as plt
import numpy as np
from igraph import Graph

from Graph_Basics import create_er_graph, plot_graph
from Graph_Compression.mc_graph_compression import get_infection_time_via_compression
from Gromov.Gromov_Operations import get_distance, \
    reconstruct_tree_from_gromov, get_multiconvex_combination, test_graph


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


def test_many_graphs():
    num_vertices = 20
    g = create_er_graph(num_vertices)

    vertex_numbers = np.arange(num_vertices)
    g.vs["vertex_num"] = vertex_numbers.tolist()

    # Create a spanning tree and find Gromov matrix
    plot_graph(g, show_vtype=True)

    combined_matrix = get_multiconvex_combination(g, 24)

    print("G-Convex Combination:\n", combined_matrix)
    print("Expected time to infect: ", get_distance(combined_matrix, 0, 1))

    reconstructed_graph = reconstruct_tree_from_gromov(combined_matrix)
    plot_graph(reconstructed_graph, show_vtype=True)

    # Now compare infection times on the original graph versus on the average Gromov spanning tree

    # First: Sample lots with one graph!
    num_samples = 10000
    test_graph(g, num_samples=num_samples)

    # Second: Compare across many graphs!
    quick_sample_size = 1000
    num_graphs = 100
    many_full_graph_infection_times = np.zeros(num_graphs)
    many_gromov_infection_times = np.zeros(num_graphs)
    for i in range(num_graphs):
        # Create a new graph for each trial
        g = create_er_graph(num_vertices)
        combined_matrix = get_multiconvex_combination(g, 24)
        reconstructed_graph = reconstruct_tree_from_gromov(combined_matrix)

        temp_full_infection_times = np.zeros(quick_sample_size)
        temp_gromov_infection_times = np.zeros(quick_sample_size)
        for j in range(quick_sample_size):
            temp_full_infection_times[j] = get_infection_time_via_compression(g, 1)
            temp_gromov_infection_times[j] = get_infection_time_via_compression(reconstructed_graph, 1)
        many_full_graph_infection_times[i] = np.average(temp_full_infection_times)
        many_gromov_infection_times[i] = np.average(temp_gromov_infection_times)

    plt.hist(many_full_graph_infection_times, bins=50, density=True, alpha=0.5, label="Full graph infection times")
    plt.hist(many_gromov_infection_times, bins=50, density=True, alpha=0.5,
             label="Gromov spanning tree infection times")
    plt.title("Comparison of many graphs")
    plt.xlabel("Infection time")
    plt.ylabel("Probability density")
    plt.legend()
    plt.show()

    plt.plot(many_full_graph_infection_times / many_gromov_infection_times)
    plt.title("Ratio of full graph infection times to Gromov spanning tree infection times")
    plt.xlabel("Graph number")
    plt.ylabel("Ratio of infection times")
    plt.show()


def test_diamond_graph():
    filename = "C:\\Users\\Levi Barksdale\\PycharmProjects\\ManuelResearch\\diamond.txt"
    g = Graph.Read_Adjacency(filename, attribute="weight", mode="undirected")
    # Add labels for the source and the observer
    num_vertices = g.vcount()
    vertex_types = np.zeros(num_vertices)
    g.vs["vtype"] = vertex_types.tolist()
    g.vs[0]["vtype"] = -1
    g.vs[1]["vtype"] = 1

    plot_graph(g, show_vtype=True)

    test_graph(g)


def test_big_graph(num_trees=4, num_vertices=50):
    g = create_er_graph(num_vertices)

    num_samples = 1000
    test_graph(g, num_trees, num_samples)


if __name__ == "__main__":
    test_diamond_graph()
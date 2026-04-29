import numpy as np
from igraph import Graph
from collections import deque

from Graph_Basics import create_er_graph, plot_graph


# Input: Graph with weighted edges
# Output: List of frequencies of edge uses
def get_edge_importance_via_walk(g, start_vertex=0, num_steps=10000):
    if not 0 <= start_vertex < g.vcount():
        raise ValueError(f"start_vertex {start_vertex} is out of range for a graph with {g.vcount()} vertices.")

    has_weight_attr = "weight" in g.es.attributes()

    # Initialize a dictionary to keep track of edge frequencies
    edge_frequencies = {}

    # Initialize a stack to keep track of backtracking
    stack = deque()
    stack.append(start_vertex)
    cur_vertex = start_vertex
    edge_index = 0

    for cur_step in range(num_steps):
        incident_edge_ids = g.incident(cur_vertex)
        if not incident_edge_ids:
            break

        incident_edges = g.es[incident_edge_ids]
        # Select an edge based on its weight relative to other edges
        edge_weights = incident_edges["weight"] if has_weight_attr else [1.0] * len(incident_edges)
        total_weight = sum(edge_weights)
        if total_weight <= 0:
            raise ValueError("Incident edge weights must sum to a positive value.")

        selection = np.random.rand() * total_weight
        cumulative_weight = 0
        for edge, edge_weight in zip(incident_edges, edge_weights):
            cumulative_weight += edge_weight
            if cumulative_weight >= selection:
                cur_vertex_pair = edge.tuple
                cur_vertex = cur_vertex_pair[1] if cur_vertex_pair[0] == cur_vertex else cur_vertex_pair[0]
                edge_index = edge.index
                break

        # Once an edge is selected, see if a cycle has occurred or just backtracking
        if cur_vertex == stack[-1]:
            stack.pop()
        else:
            stack.append(cur_vertex)
            edge_frequencies[edge_index] = edge_frequencies.get(edge_index, 0) + 1


    return edge_frequencies


if __name__ == "__main__":
    g = create_er_graph(9)
    plot_graph(g, show_vtype=True)
    print(get_edge_importance_via_walk(g, 0, 100))

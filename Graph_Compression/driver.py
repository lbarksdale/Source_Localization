# Creates a random graph where all edge delays are iid exp(1)
# Iteratively contracts edges incident to source until observer is the same
# as source vertex to simulate infection times
import random

import numpy as np

from Graph_Basics import create_er_graph, plot_graph


def sample_exponential(lambda_):
    return -np.log(1 - np.random.rand()) / lambda_


# Run a single iteration of time, i.e., contract one edge in the graph
def iterate_graph(gr):
    # First, detect edges incident to source
    incident_vertices = gr.neighbors(1)
    num_edges = len(incident_vertices)
    next_vertex_infected_time = sample_exponential(lambda_=num_edges)
    next_vertex_infected = random.choice(incident_vertices)

    # For each incident edge, sample an exponential time
    # Only keep track of which one had the smallest infection time
    # smallest_time = np.inf
    # vertex_with_smallest_time = None
    # # print(incident_vertices)
    # for v in incident_vertices:
    #     cur_time = sample_exponential(lambda_=1)
    #     if cur_time < smallest_time:
    #         smallest_time = cur_time
    #         vertex_with_smallest_time = v

    # Contract the edge between source and vertex with the smallest time
    mapping = list(range(gr.vcount()))
    mapping[next_vertex_infected] = 1
    # Combine the vertex attributes (notably vtype) by multiplying
    # When an observer is absorbed into the source, the source vtype iterates
    # Once all observers are absorbed, the source vtype will equal the number of observers
    gr.contract_vertices(mapping, combine_attrs="sum")

    # Remove self-loops but not duplicate edges. Duplicate edges are important to keep
    # as they encode information about transmission likelihood
    gr.simplify(multiple=False, loops=True)

    # Return the infection time of the vertex with the smallest time
    # print("Smallest time: ", smallest_time)
    return next_vertex_infected_time


if __name__ == "__main__":
    g = create_er_graph(100000)
    infection_time = 0

    # plot_graph(g)

    # Run while source and observer are different vertices
    num_observers = 1
    while g.vs[1]["vtype"] < num_observers - 1:
        infection_time += iterate_graph(g)
    print("Infection time: ", infection_time)
    

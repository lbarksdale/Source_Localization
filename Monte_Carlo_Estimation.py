# Test file to try out using Devlin Costello's source localization package
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from graphExportToJSON import simulate_edge_delays, create_graph_weights
from igraph import Graph
from tree_source_localization.Tree import Tree


# Method to get the estimated source using Devlin's library using the expected spanning tree
# This method is poorly implemented and should not be used
def getSourceEstimationExpected(obs, s):
    expectedTreeFilepath = "expected.json"
    expectedTree = Tree(expectedTreeFilepath, obs)
    expectedTree.simulate()
    expectedTree.simulate_infection(s)
    print("Expected tree observer infection times: " + str(expectedTree.infection_times))
    return expectedTree.localize()


# Method to get the estimated source using Devlin's library
# Specifically uses a simulated tree
# This method is poorly implemented and should not be used
def getSourceEstimationSimulated(obs, s):
    simulatedTreeFilepath = "simulated.json"
    simulatedTree = Tree(simulatedTreeFilepath, obs)
    simulatedTree.simulate()
    simulatedTree.simulate_infection(s)
    # print("Simulated tree observer infection times: " + str(simulatedTree.infection_times))
    return simulatedTree.localize()


# This method is poorly optimized and should be avoided
def getInfectionTime(filename, obs, s):
    # Import whole graph via adjacency matrix
    g = Graph.Read_Adjacency(filename, attribute="weight")
    simulate_edge_delays(g)
    return Graph.distances(g, source=int(s), target=int(obs[0]), weights=g.es["simulated_weight"])


# Gets a single infection time of a specified observer node when the graph is passed in as input
# Called as helper method in simulate_infections()
def get_infection_time(g, obs, s):
    simulate_edge_delays(g)
    return Graph.distances(g, source=s, target=obs, weights=g.es["simulated_weight"])


# Gets a vector of infection times from specified graph
def simulate_infections(g, obs, s, num_simulations):
    infection_times = np.zeros(num_simulations)
    for i in range(num_simulations):
        infection_times[i] = np.linalg.norm(get_infection_time(g, obs, s))

    return infection_times


# Simulates infection time on many graphs of the same size and plots a comparison of the expected infection time versus
# a single simulated infection time per graph
def simulate_many_graphs(num_simulations, num_nodes, p=1.2):
    full_graph_times = np.zeros(num_simulations)
    span_tree_times = np.zeros(num_simulations)

    for j in range(num_simulations):
        g = Graph.Erdos_Renyi(num_nodes, p*np.log(num_nodes) / num_nodes)

        while not Graph.is_connected(g) or Graph.are_adjacent(g, 1,2):
            g = Graph.Erdos_Renyi(num_nodes, p*np.log(num_nodes) / num_nodes)

        create_graph_weights(g)

        # Label the source with 1 and the observer with -1
        vertex_colors = np.zeros(num_nodes)
        g.vs["vtype"] = vertex_colors.tolist()
        g.vs[1]["vtype"] = 1
        g.vs[2]["vtype"] = -1

        # Simulate edge delays
        simulate_edge_delays(g)

        # Identify the most likely (in expectation) spanning tree
        # Note that there are two weights on the edges - the expectation "weight", and a sample "simulated_weight"
        span_tree = Graph.spanning_tree(g, weights=g.es["weight"])
        short_path = g.get_shortest_paths(1, to=2, weights=g.es["weight"], output="vpath")

        short_path_graph = g.subgraph(short_path[0])

        span_tree_times[j] = np.linalg.norm(sum(short_path_graph.es["simulated_weight"]))

        # Identify infection times
        full_graph_times[j] = np.linalg.norm(Graph.distances(g, source=1, target=2, weights=g.es["simulated_weight"]))
        # span_tree_times[j] = np.linalg.norm(Graph.distances(span_tree, source=1, target=2, weights=span_tree.es["simulated_weight"]))

    plot_comparison(full_graph_times, span_tree_times)


# Plots two histograms (simulated infection and spanning tree expected time) on the same plot for comparison
def plot_comparison(full_times, expected_times):
    plt.hist(full_times, bins=50, density=True, label="Full graph observer infection time", alpha=0.4)
    plt.hist(expected_times, bins=50, range=(0, 0.8*expected_times.max()), density=True, label="Expected spanning tree observer infection time", alpha=0.4)
    plt.title("Erdos-Renyi Spanning tree vs full graph infection times")
    plt.legend()
    plt.xlabel("Infection time")
    plt.ylabel("Probability density")

    plt.savefig("comparisonHistogram.png")
    plt.show()


# Plots a vector of infection times as a histogram
def plot_histogram(infection_times, highlight_time=-1, save_file_name="infection_histogram.png", title="Erdos_Renyi infection times"):
    plt.hist(infection_times, bins=50, range=(0,20), density=True, alpha=0.4)
    plt.title(title)
    plt.xlabel("Infection time")
    plt.ylabel("Probability density")

    if highlight_time != -1:
        plt.axvline(highlight_time, color="green")
        mean_val = np.average(infection_times)
        plt.axvline(mean_val, color="red")

    plt.savefig(save_file_name)
    plt.show()


# Simulates graphs of many sizes and plots the ratio of the simulated infection time (over many trials) to the expected
# spanning tree infection time
# Uses 50 graphs per size, with 50 simulations of infection time per graph, so it takes a while to run
def compare_graph_size_infection_times(num_simulations, p=2):
    normalized_infection_times = np.zeros(num_simulations)
    graph_sizes = np.linspace(50, 9*num_simulations+50, num_simulations)
    for i in range(num_simulations):
        cur_size = int(graph_sizes[i])
        num_graphs_per_size = 50
        cur_normalized_infection_times = np.zeros(num_graphs_per_size)

        for k in range(num_graphs_per_size):
            # Create graph of correct size
            g = Graph.Erdos_Renyi(cur_size, p * np.log(cur_size) / cur_size)
            # g = Graph.GRG(cur_size, 1/np.log(cur_size))
            create_graph_weights(g)
            expected_infection_time = np.linalg.norm(Graph.distances(g, source=1, target=2, weights=g.es["weight"]))

            # Check if graph is connected and has sufficiently far away enough observer / source to be useful
            # Note: You may be introducing some bias here with the "sufficiently far away" condition!
            while not g.is_connected() or expected_infection_time < 10:
                g = Graph.Erdos_Renyi(cur_size, p * np.log(cur_size) / cur_size)
                # g = Graph.GRG(cur_size, 1/np.log(cur_size))
                create_graph_weights(g)
                expected_infection_time = np.linalg.norm(Graph.distances(g, source=1, target=2, weights=g.es["weight"]))

            print("Graph created successfully! ", cur_size)

            # Simulate infection numerous times on graph to get estimated mean
            num_trials = 50
            simulated_times = np.zeros(num_trials)
            for j in range(num_trials):
                simulated_times[j] = np.linalg.norm(get_infection_time(g, 1, 2))
            true_infection_time = np.mean(simulated_times)

            cur_normalized_infection_times[k] = true_infection_time / expected_infection_time
        normalized_infection_times[i] = np.mean(cur_normalized_infection_times)


    plt.plot(graph_sizes, normalized_infection_times)
    plt.title("Graph size vs normalized average infection time")
    plt.xlabel("Graph size")
    plt.ylabel("Normalized average infection time")
    plt.savefig("forest_fire_graph_size_vs_infection_time.png")
    plt.show()
    return normalized_infection_times


# Simulates graphs of the same size but with varying connection probabilities, then plots the ratio of the simulated
# infection time (over many trials) to the expected spanning tree infection time
def compare_graph_connectivity_infection_times(num_simulations, graph_size):
    normalized_infection_times = np.zeros(num_simulations)
    connection_threshold = np.log(graph_size) / graph_size
    p = np.linspace(1.2, 10, num_simulations)

    for i in range(num_simulations):
        num_graphs_per_p = 50
        cur_normalized_infection_times = np.zeros(num_graphs_per_p)

        for j in range(num_graphs_per_p):
            g = Graph.Erdos_Renyi(graph_size, p[i] * connection_threshold)
            create_graph_weights(g)
            expected_infection_time = np.linalg.norm(Graph.distances(g, source=1, target=2, weights=g.es["weight"]))

            # Check that graph is connected and we have a decently far observer from source
            while not g.is_connected() or expected_infection_time < 1:
                g = Graph.Erdos_Renyi(graph_size, p[i] * connection_threshold)
                create_graph_weights(g)
                expected_infection_time = np.linalg.norm(Graph.distances(g, source=1, target=2, weights=g.es["weight"]))

            # Simulate infection time in numerous trials to estimate mean
            num_trials = 50
            simulated_times = np.zeros(num_trials)
            for k in range(num_trials):
                simulated_times[k] = np.linalg.norm(get_infection_time(g, 1, 2))
            true_infection_time = np.mean(simulated_times)

            cur_normalized_infection_times[j] = true_infection_time / expected_infection_time
        normalized_infection_times[i] = np.mean(cur_normalized_infection_times)

    plt.plot(p, normalized_infection_times)
    plt.title("Edge connection likelihood vs normalized average infection time")
    plt.xlabel("Multiple of connection threshold")
    plt.ylabel("Normalized average infection time")
    plt.savefig("connection_likelihood_vs_infection_time.png")
    plt.show()
    return normalized_infection_times


if __name__ == "__main__":
    # simulate_many_graphs(10000, 100)

    # compare_graph_size_infection_times(num_simulations=20)

    # compare_graph_connectivity_infection_times(num_simulations=100, graph_size=100)

    ##########################################################################
    # Code to create Monte Carlo simulation comparison for a graph given in filename
    ##########################################################################
    filename = "unit_delays.txt"
    g = Graph.Read_Adjacency(filename, attribute="weight", mode="undirected")
    n_vertices = g.vcount()
    vertexColors = np.zeros(n_vertices)
    g.vs["vtype"] = vertexColors.tolist()
    g.vs[0]["vtype"] = 1
    g.vs[1]["vtype"] = -1
    # g.vs[3]["vtype"] = -1
    # g.vs[4]["vtype"] = -1
    # g.vs["obs_number"] = vertexColors.tolist()
    # g.vs[2]["obs_number"] = 1
    # g.vs[3]["obs_number"] = 2
    # g.vs[4]["obs_number"] = 3

    n = 10000
    source = 1
    obs = 0

    obs_1_times = simulate_infections(g, 2, source, n)
    # obs_2_times = simulate_infections(g, 3, source, n)
    # obs_3_times = simulate_infections(g, 4, source, n)

    observer_value = get_infection_time(g, 1, 2)[0][0]

    plot_histogram(obs_1_times, observer_value, "Observer_1_histogram.png", "Source Candidate 1 Infection Times")
    # plot_histogram(obs_2_times, observer_value, "Observer_2_histogram.png", "Source Candidate 2 Infection Times")
    # plot_histogram(obs_3_times, observer_value, "Observer_3_histogram.png", "Source Candidate 3 Infection Times")

    # print("Observer infection time: ", observer_value)
    mean_infection_time = np.abs(np.mean(obs_1_times))
    print("Mean observer infection time:", mean_infection_time)
    # print("Source 2:", np.abs(np.mean(obs_2_times)))
    # print("Source 3:", np.abs(np.mean(obs_3_times)))

    # Center and scale the mean infection time
    expected_infection_time = Graph.distances(g, source=source, target=obs, weights=g.es["weight"])[0][0]

    # Let's test over a lot of possible numbers of paths
    possible_num_paths = np.linspace(1, 101, 100)



    # Plot in matplotlib
    # Note that attributes can be set globally (e.g., vertex_size), or set individually using arrays (e.g., vertex_color)
    fig, ax = plt.subplots(figsize=(5, 5))
    ig.plot(
        g,
        target=ax,
        vertex_size=[20 if vtype==1 or vtype==-1 else 10 for vtype in g.vs["vtype"]],
        # vertex_label=[vertex["obs_number"] if vertex["vtype"]==-1 else "o" if vertex["vtype"]==1 else "" for vertex in g.vs],
        vertex_frame_width=2.0,
        vertex_frame_color="white",
        vertex_color=["salmon" if vtype == -1 else "green" if vtype == 1 else "steelblue" for vtype in g.vs["vtype"]],
        # edge_label=g.es["weight"]
    )
    # plt.savefig("sampleGraph.png")
    plt.show()
    ################################################################################


    # filePath = "expectedBIG.json"  # "data/Pinto.json"
    # # observers: String list of observer nodes
    # observers = ["2"]
    # source = "1"
    #
    # print("Expected-time graph source: " + getSourceEstimationExpected(observers, source))
    #
    # # Simulated-time graph source isn't great, I think you need to tweak the distributions of the simulated graph edges
    # # Perhaps to normal with a small variance? Perhaps it's completely unnecessary?
    #
    # # print("Simulated-time graph source: " + getSourceEstimationSimulated(observers, source))
    #
    # print("Actual infection time: " + str(getInfectionTime("playgraphBIG.txt", observers, source)))
    #
    # # Simulate a bunch of infection times
    # numSimulations = 10000
    #
    # infectionTimes = np.zeros(numSimulations)
    # expectedInfectionTimes = np.zeros(numSimulations)
    #
    # for i in range(numSimulations):
    #     infectionTimes[i] = np.linalg.norm(getInfectionTime("playgraphBIG.txt", observers, source))
    #     expectedInfectionTimes[i] = np.linalg.norm(getInfectionTime("expectedSpanTree.txt", observers, source))
    #     # print("Simulation " + str(i) + " infection time: " + str(getTrueInfectionTime("playgraph.txt", observers, source)))
    #
    # plot_comparison(infectionTimes, expectedInfectionTimes)





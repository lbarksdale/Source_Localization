# Random graph generation via dropping points on the unit square
# Points are connected if they are close to each other
# This is the Geometric Random Graph model

import igraph as ig
from igraph import Graph
import random
import matplotlib.pyplot as plt

from Graph_Basics import plot_graph


# Creates array of (index, x,y) points uniformly on unit square
def create_points(num_vertices):
    points = []
    for j in range(num_vertices):
        cur_point = [j, random.random(), random.random()]
        points.append(cur_point)
    return points

def create_graph(points, d):
    g = ig.Graph()
    num_vertices = len(points)
    Graph.add_vertices(g, num_vertices)
    for j in range(num_vertices):
        idx1, x1, y1 = points[j]
        for k in range(j+1, num_vertices):
            idx2, x2, y2 = points[k]
            dx = x1 - x2
            dy = y1 - y2

            if dx * dx + dy * dy < d:
                Graph.add_edge(g, idx1, idx2)
    return g


if __name__ == "__main__":
    points = create_points(30)
    g = create_graph(points, 0.05)
    plot_graph(g)

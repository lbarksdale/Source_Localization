# File contains several functions describing basic operations with Gromov matrices
# Relies on the igraph framework
import numpy as np


def get_gromov_product(graph, v1, v2, root):
    d1 = graph.distances(source=v1, target=root)[0][0]
    d2 = graph.distances(source=v2, target=root)[0][0]
    d3 = graph.distances(source=v1, target=v2)[0][0]
    return (1/2)*(d1 + d2 - d3)

def get_gromov_matrix(graph, root):
    matrix_size = graph.vcount() - 1
    matrix = np.zeros((matrix_size, matrix_size))

    # Problem: If root is not the final vertex in the graph, the indexing is getting messed up

    # Fill the upper triangle of the matrix with the Gromov products
    for row_vertex in range(0, matrix_size):
        for col_vertex in range(row_vertex + 1, matrix_size):
            # Correct indexing and calculate Gromov product
            fixed_row_vertex = row_vertex if row_vertex < root else row_vertex + 1
            fixed_col_vertex = col_vertex if col_vertex < root else col_vertex + 1

            matrix[row_vertex, col_vertex] = get_gromov_product(graph, fixed_row_vertex, fixed_col_vertex, root)

    # Make matrix symmetric
    matrix = matrix + matrix.T

    # Diagonal entries
    for i in range(matrix_size):
        # Correct indexing
        target_vertex = i if i < root else i + 1
        matrix[i, i] = graph.distances(source=root, target=target_vertex)[0][0]

    return matrix
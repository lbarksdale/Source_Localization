# File contains several functions describing basic operations with Gromov matrices
# Relies on the igraph framework
import igraph as ig
from igraph import Graph
import numpy as np


def get_gromov_product(graph, v1, v2, root):
    d1 = graph.distances(source=v1, target=root)[0][0]
    d2 = graph.distances(source=v2, target=root)[0][0]
    d3 = graph.distances(source=v1, target=v2)[0][0]
    return (1 / 2) * (d1 + d2 - d3)


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


def g_convex_combination(m1, m2, alpha):
    """
    Compute the G-convex combination of two Gromov matrices m1, m2
    with weight alpha in [0,1].

    This follows the paper's procedure:
    1) Form convex combination M_alpha
    2) Extract upper-triangular entries and sort them (non-increasing)
    3) Iteratively enforce the three-point condition using the ordering
    4) Write the corrected values back into a symmetric matrix
    """

    m1 = np.array(m1, dtype=float)
    m2 = np.array(m2, dtype=float)

    # --- Step 1: convex combination ---
    m = alpha * m1 + (1 - alpha) * m2
    n = m.shape[0]

    # --- Step 2: build U(M): sorted upper-triangular entries ---
    # We store entries as (value, i, j)
    entries = []
    for i in range(n):
        for j in range(i + 1, n):
            entries.append([m[i, j], i, j])

    # Sort in non-increasing order
    entries.sort(key=lambda x: -x[0])

    # --- Helper: check if three entries correspond to the same triple ---
    def is_same_triangle(e1, e2, e3):
        """
        Check if entries e1=(i, j), e2=(i, k), e3=(j, k) for some i,j,k.
        """
        indices = {tuple(sorted((e1[1], e1[2]))),
                   tuple(sorted((e2[1], e2[2]))),
                   tuple(sorted((e3[1], e3[2])))}

        # A triangle must involve exactly 3 distinct nodes and 3 edges
        nodes = set()
        for (a, b) in indices:
            nodes.add(a)
            nodes.add(b)

        return len(nodes) == 3 and len(indices) == 3

    # --- Step 3: enforce the three-point condition ---
    # Number of iterations per combination: n(n-2)/2
    max_t = n * (n - 2) // 2

    for t in range(max_t):
        # Re-scan this pivot until every triangle involving it satisfies
        # the three-point condition under the current ordering.
        while True:
            xt = entries[t][0]
            violation_fixed = False

            # Look for pairs (u, v) with u < t < v
            for u in range(t):
                for v in range(t + 1, len(entries)):

                    if is_same_triangle(entries[u], entries[t], entries[v]):
                        # entries[v] is the smallest in the triple (due to sorting)

                        # If violation: smallest < middle (xt)
                        if entries[v][0] < xt:
                            # --- Core correction step ---
                            # Raise the smallest entry to match xt
                            entries[v][0] = xt

                            # --- Reinsert to maintain sorted order ---
                            # Remove and reinsert at the correct position
                            val, i, j = entries[v]
                            del entries[v]

                            # Find new position (after all >= xt)
                            insert_pos = t + 1
                            while (insert_pos < len(entries) and
                                   entries[insert_pos][0] >= val):
                                insert_pos += 1

                            entries.insert(insert_pos, [val, i, j])
                            violation_fixed = True
                            break
                if violation_fixed:
                    break

            if not violation_fixed:
                break

    # --- Step 4: reconstruct symmetric matrix ---
    m_out = np.array(m)

    # Write corrected upper-triangular entries back
    for val, i, j in entries:
        m_out[i, j] = val
        m_out[j, i] = val

    return m_out


def get_distance(m, v1, v2):
    return m[v1, v1] + m[v2, v2] - 2*m[v1, v2]


def reconstruct_tree_from_gromov(m):
    """
    Reconstruct a weighted tree from a valid Gromov matrix M
    using igraph.

    Returns:
        g: igraph.Graph (tree)
    """

    m = np.array(m, dtype=float)
    n = m.shape[0]

    g = ig.Graph()

    # Add root node
    Graph.add_vertex(g, name="s")
    root = 0

    # Keep track of vertex indices
    node_index = {"s": root}
    next_internal_id = 0

    def add_vertex(name):
        Graph.add_vertex(g, name=name)
        node_index[name] = g.vcount() - 1
        return node_index[name]

    def add_edge(u, v, w):
        Graph.add_edge(g, u, v)
        g.es[-1]["weight"] = w

    # Add v0
    v0 = add_vertex("v0")
    add_edge(root, v0, m[0, 0])

    # --- helper: find the path from root to a node ---
    def get_path(u, v):
        return g.get_shortest_paths(u, to=v, weights=None, output="vpath")[0]

    # --- helper: find or create a point at distance d along the path ---
    def find_or_create_point(input_path, d):
        """
        Given a path (list of vertex indices), find the point at distance d from the root.
        If inside an edge, split it.
        """
        dist = 0.0

        for i in range(len(input_path) - 1):
            u = input_path[i]
            v = input_path[i + 1]

            eid = g.get_eid(u, v)
            w = g.es[eid]["weight"]

            if dist + w == d:
                return v
            elif dist < d < dist + w:
                # Split edge
                nonlocal next_internal_id
                new_name = f"p{next_internal_id}"
                next_internal_id += 1
                new_node = add_vertex(new_name)

                # Remove old edge
                g.delete_edges(eid)

                # Add split edges
                add_edge(u, new_node, d - dist)
                add_edge(new_node, v, dist + w - d)

                return new_node

            dist += w

        return input_path[-1]  # fallback (exact match at leaf)

    # --- main loop ---
    for k in range(1, n):
        vk_name = f"v{k}"
        vk = add_vertex(vk_name)

        # Find anchor j maximizing M(j,k)
        j = max(range(k), key=lambda i: m[i, k])
        vj = node_index[f"v{j}"]

        # Path from root to vj
        path = get_path(root, vj)

        # Find attachment point
        attach_dist = m[j, k]
        p = find_or_create_point(path, attach_dist)

        # Attach vk
        edge_length = m[k, k] - attach_dist
        add_edge(p, vk, edge_length)

    # Keep plotting helpers compatible with the rest of the project.
    g.vs["vtype"] = [0] * g.vcount()

    # Contract any edges of length 0
    zero_weight_edges = [
        edge.tuple for edge in g.es
        if np.isclose(edge["weight"], 0.0)
    ]
    if zero_weight_edges:
        zero_graph = ig.Graph(n=g.vcount())
        Graph.add_edges(zero_graph, zero_weight_edges)
        membership = Graph.connected_components(zero_graph).membership

        g.contract_vertices(membership, combine_attrs="first")
        Graph.simplify(g, multiple=True, loops=True, combine_edges="first")

    # Color the source and the observers
    g.vs[0]["vtype"] = -1
    g.vs[1]["vtype"] = 1

    return g

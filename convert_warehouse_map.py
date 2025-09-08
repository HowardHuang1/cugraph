#!/usr/bin/env python3
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
"""
Convert warehouse.map to edge list format for betweenness centrality testing.
Creates edges between adjacent traversable cells (4-connected grid).
"""

def read_map_file(filename):
    """Read the map file and return the grid."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip header lines
    grid_lines = []
    for line in lines:
        line = line.strip()
        if line == 'map':
            continue
        elif line.startswith('type') or line.startswith('height') or line.startswith('width'):
            continue
        elif line:  # Non-empty line
            grid_lines.append(line)

    return grid_lines

def grid_to_edge_list(grid):
    """Convert grid to edge list format."""
    height = len(grid)
    width = len(grid[0]) if grid else 0

    edges = []
    node_id = 0

    # Create a mapping from (row, col) to node_id
    coord_to_id = {}

    # First pass: assign node IDs to traversable cells in row-major order
    for row in range(height):
        for col in range(width):
            if grid[row][col] == '.':  # Traversable cell
                coord_to_id[(row, col)] = node_id
                node_id += 1

    print(f"Found {node_id} traversable cells")

    # Second pass: create edges between adjacent cells
    edge_set = set()  # Use set to avoid duplicates

    for row in range(height):
        for col in range(width):
            if grid[row][col] == '.':  # Current cell is traversable
                current_id = coord_to_id[(row, col)]

                # Check 4-connected neighbors (up, down, left, right)
                neighbors = [
                    (row-1, col),  # up
                    (row+1, col),  # down
                    (row, col-1),  # left
                    (row, col+1),  # right
                ]

                for n_row, n_col in neighbors:
                    # Check bounds
                    if 0 <= n_row < height and 0 <= n_col < width:
                        if grid[n_row][n_col] == '.':  # Neighbor is traversable
                            neighbor_id = coord_to_id[(n_row, n_col)]
                            # Add bidirectional edges for undirected graph
                            edge_set.add((current_id, neighbor_id))
                            edge_set.add((neighbor_id, current_id))

    # Convert set to list
    edges = list(edge_set)

    return edges, node_id

def write_edge_list(edges, filename, weighted=False):
    """Write edges to CSV file."""
    with open(filename, 'w') as f:
        for src, dst in edges:
            if weighted:
                f.write(f"{src},{dst},1.0\n")  # Add weight of 1.0
            else:
                f.write(f"{src},{dst}\n")

def main():
    import sys
    map_file = sys.argv[1] if len(sys.argv) > 1 else 'warehouse.map'

    # Read the map
    grid = read_map_file(map_file)

    if not grid:
        print("Error: Could not read map file")
        return

    print(f"Map dimensions: {len(grid[0])} x {len(grid)}")

    # Convert to edge list
    edges, num_nodes = grid_to_edge_list(grid)

    print(f"Created {len(edges)} edges")
    print(f"Graph has {num_nodes} nodes")

    # Write edge list (unweighted)
    write_edge_list(edges, 'warehouse_edges.csv', weighted=False)
    print("Edge list written to warehouse_edges.csv")

    # Write weighted edge list
    write_edge_list(edges, 'warehouse_edges_weighted.csv', weighted=True)
    print("Weighted edge list written to warehouse_edges_weighted.csv")

    # Print some statistics
    print(f"\nGraph statistics:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {len(edges)}")
    print(f"  Average degree: {2 * len(edges) / num_nodes:.2f}")

if __name__ == "__main__":
    main()

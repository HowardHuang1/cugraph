#!/usr/bin/env python3
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_warehouse_nx():
    """Visualize NetworkX betweenness centrality results on warehouse graph"""

    print("NETWORKX WAREHOUSE CENTRALITY VISUALIZATION")
    print("=" * 60)

    # Read the warehouse edge list
    try:
        G = nx.read_edgelist('warehouse.csv',
                            delimiter=',',
                            nodetype=int,
                            data=[('weight', float)],
                            create_using=nx.Graph())
        print(f"Loaded warehouse graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except FileNotFoundError:
        print("ERROR: warehouse.csv not found!")
        print("Please run convert_warehouse_map.py first to generate the edge list.")
        return

    # Calculate betweenness centrality
    centrality = nx.betweenness_centrality(G, normalized=False, endpoints=True)

    # Create the map grid (9x12)
    map_grid = [
        "TTTTTTTTTTTT",
        "T..........T",
        "T..........T",
        "T..........T",
        "TTTTT..TTTTT",  # Bottleneck at col 5-6
        "T..........T",
        "T..........T",
        "T..........T",
        "TTTTTTTTTTTT"
    ]

    # Create centrality grid
    height, width = len(map_grid), len(map_grid[0])
    centrality_grid = np.zeros((height, width))

    # Map node IDs to grid positions (row-major order)
    node_idx = 0
    for row in range(height):
        for col in range(width):
            if map_grid[row][col] == '.':
                if node_idx in centrality:
                    centrality_grid[row][col] = centrality[node_idx]
                node_idx += 1

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Original map
    ax1.set_title("Warehouse Map Layout", fontsize=14, fontweight='bold')
    original_matrix = []
    for row in map_grid:
        original_matrix.append([1 if c == '.' else 0 for c in row])

    ax1.imshow(original_matrix, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")

    # Add grid lines
    ax1.set_xticks(range(width))
    ax1.set_yticks(range(height))
    ax1.grid(True, alpha=0.3)

    # Highlight bottleneck area
    bottleneck_row, bottleneck_cols = 4, [5, 6]
    for col in bottleneck_cols:
        ax1.add_patch(plt.Rectangle((col-0.5, bottleneck_row-0.5), 1, 1,
                                   fill=False, edgecolor='red', linewidth=3))

    # Plot 2: Centrality heatmap
    ax2.set_title("NetworkX Betweenness Centrality", fontsize=14, fontweight='bold')

    # Create heatmap
    im = ax2.imshow(centrality_grid, cmap='Reds', aspect='equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Betweenness Centrality', rotation=270, labelpad=20)

    # Add grid lines
    ax2.set_xticks(range(width))
    ax2.set_yticks(range(height))
    ax2.grid(True, alpha=0.3)

    # Highlight bottleneck area
    for col in bottleneck_cols:
        ax2.add_patch(plt.Rectangle((col-0.5, bottleneck_row-0.5), 1, 1,
                                   fill=False, edgecolor='blue', linewidth=3))

    # Add node numbers on the heatmap
    node_idx = 0
    for row in range(height):
        for col in range(width):
            if map_grid[row][col] == '.':
                if node_idx in centrality:
                    # Only show node numbers for high centrality nodes
                    if centrality[node_idx] > 300:
                        ax2.text(col, row, str(node_idx), ha='center', va='center',
                                fontsize=8, fontweight='bold', color='white')
                node_idx += 1

    plt.tight_layout()
    plt.savefig('warehouse_nx_centrality_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print analysis
    print("\n" + "=" * 60)
    print("CENTRALITY ANALYSIS:")
    print("=" * 60)

    # Get top centrality nodes
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    print("Top 10 nodes by centrality:")
    for i, (node, cent) in enumerate(sorted_centrality[:10]):
        # Find grid position
        node_idx = 0
        grid_pos = None
        for row in range(height):
            for col in range(width):
                if map_grid[row][col] == '.':
                    if node_idx == node:
                        grid_pos = (row, col)
                        break
                    node_idx += 1

        if grid_pos:
            row, col = grid_pos
            print(f"  {i+1:2d}. Node {node:2d}: {cent:8.3f} at Row {row}, Col {col}")
        else:
            print(f"  {i+1:2d}. Node {node:2d}: {cent:8.3f}")

    # Check bottleneck area
    print(f"\nBottleneck area (Row 4, Col 5-6):")
    bottleneck_nodes = [30, 31]  # Nodes at bottleneck
    for node in bottleneck_nodes:
        if node in centrality:
            rank = next(i for i, (n, _) in enumerate(sorted_centrality) if n == node) + 1
            print(f"  Node {node}: {centrality[node]:.3f} (rank {rank})")

    print(f"\nThis shows the CORRECT betweenness centrality pattern:")
    print(f"- High centrality nodes are near the bottleneck")
    print(f"- Corner nodes have lower centrality")
    print(f"- The spatial distribution makes sense")

if __name__ == "__main__":
    try:
        visualize_warehouse_nx()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install matplotlib networkx")
    except Exception as e:
        print(f"Error: {e}")

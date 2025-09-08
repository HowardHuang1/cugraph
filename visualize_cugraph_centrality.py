#!/usr/bin/env python3
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import os
os.environ['NX_CUGRAPH_AUTOCONFIG'] = 'True'

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def create_centrality_visualization(centrality_dict):
    """Create a color-coded visualization of centrality on the map"""

    # Use the same hardcoded map grid as the NetworkX script
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
    centrality_array = np.zeros((height, width))

    # Map node IDs to grid positions (row-major order) - same as NetworkX script
    node_idx = 0
    for row in range(height):
        for col in range(width):
            if map_grid[row][col] == '.':
                if node_idx in centrality_dict:
                    centrality_array[row][col] = centrality_dict[node_idx]
                node_idx += 1

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Original map
    original_matrix = []
    for row in map_grid:
        original_matrix.append([1 if c == '.' else 0 for c in row])

    ax1.imshow(original_matrix, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax1.set_title('Original Map (Black=Walls, White=Paths)')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')

    # Add grid lines
    ax1.set_xticks(range(width))
    ax1.set_yticks(range(height))
    ax1.grid(True, alpha=0.3)

    # Highlight bottleneck area
    bottleneck_row, bottleneck_cols = 4, [5, 6]
    for col in bottleneck_cols:
        ax1.add_patch(plt.Rectangle((col-0.5, bottleneck_row-0.5), 1, 1,
                                   fill=False, edgecolor='red', linewidth=3))

    # Plot 2: Centrality visualization
    im = ax2.imshow(centrality_array, cmap='Reds', aspect='equal')
    ax2.set_title('cuGraph Betweenness Centrality (Red=High, Blue=Low)')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')

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

    # Add node numbers on the heatmap for high centrality nodes
    node_idx = 0
    for row in range(height):
        for col in range(width):
            if map_grid[row][col] == '.':
                if node_idx in centrality_dict and centrality_dict[node_idx] > 300:
                    ax2.text(col, row, str(node_idx), ha='center', va='center',
                            fontsize=8, fontweight='bold', color='white')
                node_idx += 1

    plt.tight_layout()
    return fig

def get_cugraph_centrality():
    """Get centrality values using cuGraph via nx-cugraph"""

    # Read the warehouse edge list
    G = nx.read_edgelist('warehouse.csv',
                        delimiter=',',
                        nodetype=int,
                        data=[('weight', float)],
                        create_using=nx.Graph())

    print(f"Loaded warehouse graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Calculate betweenness centrality using cuGraph (via nx-cugraph)
    centrality = nx.betweenness_centrality(G, normalized=False, endpoints=True)

    return centrality

def main():
    print("CUGRAPH CENTRALITY VISUALIZATION")
    print("=" * 40)

    # Get centrality values from cuGraph
    try:
        centrality = get_cugraph_centrality()
        print(f"Got centrality for {len(centrality)} nodes")
        print(f"Max centrality: {max(centrality.values()):.3f}")
        print(f"Min centrality: {min(centrality.values()):.3f}")

        # Show top 10 nodes
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 10 nodes by centrality:")
        for i, (node_id, cent) in enumerate(sorted_centrality[:10]):
            print(f"  {i+1:2d}. Node {node_id:2d}: {cent:8.3f}")

    except Exception as e:
        print(f"Error getting centrality values: {e}")
        return

    # Create visualization
    try:
        fig = create_centrality_visualization(centrality)

        # Save the plot
        output_file = 'cugraph_centrality_visualization.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as: {output_file}")

        # Show the plot
        plt.show()

    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import os
os.environ['NX_CUGRAPH_AUTOCONFIG'] = 'True'

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def create_centrality_visualization(centrality_dict, map_type="warehouse"):
    """Create a color-coded visualization of centrality on the map"""

    # Define different map grids for different graph sizes
    map_grids = {
        "small": [
            "TTTTTTTTTTTT",
            "T..........T",
            "T..........T",
            "T..........T",
            "TTTTT..TTTTT",  # Single bottleneck at col 5-6
            "T..........T",
            "T..........T",
            "T..........T",
            "TTTTTTTTTTTT"
        ],
        "medium": [
            "TTTTTTTTTTTTTTTTTT",
            "T..................T",
            "T..................T",
            "T..................T",
            "TTTTT........TTTTTTT",  # First bottleneck
            "T..................T",
            "T..................T",
            "T..................T",
            "T..................T",
            "TTTTT........TTTTTTT",  # Second bottleneck
            "T..................T",
            "T..................T",
            "T..................T",
            "TTTTTTTTTTTTTTTTTT"
        ],
        "large": [
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTT",
            "T..........................T",
            "T..........................T",
            "T..........................T",
            "T..........................T",
            "TTTTT................TTTTTTT",  # First bottleneck
            "T..........................T",
            "T..........................T",
            "T..........................T",
            "T..........................T",
            "TTTTT................TTTTTTT",  # Second bottleneck
            "T..........................T",
            "T..........................T",
            "T..........................T",
            "T..........................T",
            "TTTTT................TTTTTTT",  # Third bottleneck
            "T..........................T",
            "T..........................T",
            "T..........................T",
            "T..........................T",
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTT"
        ]
    }

    # Select the appropriate map grid
    map_grid = map_grids.get(map_type, map_grids["small"])

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

    # Highlight bottleneck areas based on map type
    if map_type == "small":
        bottleneck_row, bottleneck_cols = 4, [5, 6]
        for col in bottleneck_cols:
            ax1.add_patch(plt.Rectangle((col-0.5, bottleneck_row-0.5), 1, 1,
                                       fill=False, edgecolor='red', linewidth=3))
    elif map_type == "medium":
        # Two bottlenecks in medium map
        bottlenecks = [(4, [5, 6]), (9, [5, 6])]
        for row, cols in bottlenecks:
            for col in cols:
                ax1.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1,
                                           fill=False, edgecolor='red', linewidth=3))
    elif map_type == "large":
        # Three bottlenecks in large map
        bottlenecks = [(5, [5, 6]), (10, [5, 6]), (15, [5, 6])]
        for row, cols in bottlenecks:
            for col in cols:
                ax1.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1,
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

    # Highlight bottleneck areas based on map type
    if map_type == "small":
        bottleneck_row, bottleneck_cols = 4, [5, 6]
        for col in bottleneck_cols:
            ax2.add_patch(plt.Rectangle((col-0.5, bottleneck_row-0.5), 1, 1,
                                       fill=False, edgecolor='blue', linewidth=3))
    elif map_type == "medium":
        # Two bottlenecks in medium map
        bottlenecks = [(4, [5, 6]), (9, [5, 6])]
        for row, cols in bottlenecks:
            for col in cols:
                ax2.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1,
                                           fill=False, edgecolor='blue', linewidth=3))
    elif map_type == "large":
        # Three bottlenecks in large map
        bottlenecks = [(5, [5, 6]), (10, [5, 6]), (15, [5, 6])]
        for row, cols in bottlenecks:
            for col in cols:
                ax2.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1,
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

def get_cugraph_centrality(csv_file="warehouse.csv"):
    """Get centrality values using cuGraph via nx-cugraph"""

    # Read the edge list
    G = nx.read_edgelist(csv_file,
                        delimiter=',',
                        nodetype=int,
                        data=[('weight', float)],
                        create_using=nx.Graph())

    print(f"Loaded graph from {csv_file}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Calculate betweenness centrality using cuGraph (via nx-cugraph)
    centrality = nx.betweenness_centrality(G, normalized=False, endpoints=True)

    return centrality

def main():
    import sys

    # Parse command line arguments - require both map type and CSV file
    if len(sys.argv) < 3:
        print("ERROR: Missing required arguments!")
        print("Usage: python3 visualize_cugraph_centrality.py <map_type> <csv_file>")
        print("Map types: small, medium, large")
        print("Example: python3 visualize_cugraph_centrality.py small warehouse.csv")
        return

    map_type = sys.argv[1]
    csv_file = sys.argv[2]

    # Validate map type
    valid_types = ["small", "medium", "large"]
    if map_type not in valid_types:
        print(f"ERROR: Invalid map type '{map_type}'!")
        print(f"Valid types: {', '.join(valid_types)}")
        return

    print(f"CUGRAPH CENTRALITY VISUALIZATION - {map_type.upper()}")
    print("=" * 50)
    print(f"Map type: {map_type}")
    print(f"CSV file: {csv_file}")
    print()

    # Get centrality values from cuGraph
    try:
        centrality = get_cugraph_centrality(csv_file)
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
        fig = create_centrality_visualization(centrality, map_type)

        # Create robotics_maps directory if it doesn't exist
        import os
        os.makedirs('robotics_maps', exist_ok=True)

        # Save the plot
        output_file = f'robotics_maps/cugraph_centrality_{map_type}.png'
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

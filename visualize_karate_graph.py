#!/usr/bin/env python3
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import os
os.environ['NX_CUGRAPH_AUTOCONFIG'] = 'True'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx

def visualize_karate_graph():
    """Visualize betweenness centrality on the karate graph"""

    # Load the karate graph
    G = nx.karate_club_graph()

    print("KARATE GRAPH BETWEENNESS CENTRALITY ANALYSIS")
    print("=" * 50)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print()

    # Calculate betweenness centrality using cuGraph (via nx-cugraph)
    centrality = nx.betweenness_centrality(G, normalized=False, endpoints=True)

    # Get the top nodes by centrality
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    print("Top 10 nodes by betweenness centrality:")
    for i, (node, cent) in enumerate(sorted_centrality[:10]):
        print(f"{i+1:2d}. Node {node:2d}: {cent:.4f}")

    print()
    print("Expected high centrality nodes:")
    print("- Node 0: Club president (Mr. Hi)")
    print("- Node 33: Club instructor (John A.)")
    print("- Node 1: Close to president")
    print("- Node 2: Close to president")
    print("- Node 3: Close to president")
    print("- Node 32: Close to instructor")
    print("- Node 31: Close to instructor")
    print("- Node 30: Close to instructor")

    # Check if our expected high centrality nodes are actually high
    expected_high = [0, 33, 1, 2, 3, 32, 31, 30]
    print(f"\nCentrality values for expected high centrality nodes:")
    for node in expected_high:
        if node in centrality:
            print(f"Node {node:2d}: {centrality[node]:.4f}")

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Plot 1: Network layout with centrality coloring
    plt.subplot(1, 3, 1)
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility

    # Color nodes by centrality
    node_colors = [centrality[node] for node in G.nodes()]

    nx.draw(G, pos,
            node_color=node_colors,
            node_size=300,
            cmap='Reds',
            with_labels=True,
            font_size=8,
            font_weight='bold')

    plt.title("Karate Graph - Betweenness Centrality")

    # Create colorbar properly
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Centrality')

    # Plot 2: Centrality bar chart
    plt.subplot(1, 3, 2)
    nodes = list(centrality.keys())
    values = list(centrality.values())

    bars = plt.bar(nodes, values, color='red', alpha=0.7)
    plt.xlabel('Node ID')
    plt.ylabel('Betweenness Centrality')
    plt.title('Centrality Values by Node')
    plt.xticks(rotation=45)

    # Highlight expected high centrality nodes
    for node in expected_high:
        if node in nodes:
            idx = nodes.index(node)
            bars[idx].set_color('blue')
            bars[idx].set_alpha(1.0)

    # Plot 3: Centrality distribution
    plt.subplot(1, 3, 3)
    plt.hist(values, bins=15, color='red', alpha=0.7, edgecolor='black')
    plt.xlabel('Centrality Value')
    plt.ylabel('Frequency')
    plt.title('Centrality Distribution')

    # Add statistics
    mean_cent = np.mean(values)
    std_cent = np.std(values)
    plt.axvline(mean_cent, color='blue', linestyle='--', label=f'Mean: {mean_cent:.3f}')
    plt.axvline(mean_cent + std_cent, color='green', linestyle='--', label=f'Mean+Std: {mean_cent + std_cent:.3f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('cugraph_karate.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Analyze the results
    print("\n" + "=" * 50)
    print("ANALYSIS:")
    print("=" * 50)

    # Check if the algorithm makes sense
    top_5_nodes = [node for node, _ in sorted_centrality[:5]]
    print(f"Top 5 centrality nodes: {top_5_nodes}")

    # Check if expected nodes are in top 10
    top_10_nodes = [node for node, _ in sorted_centrality[:10]]
    expected_in_top_10 = [node for node in expected_high if node in top_10_nodes]
    print(f"Expected high centrality nodes in top 10: {expected_in_top_10}")

    # Check if the results make sense
    if 0 in top_5_nodes and 33 in top_5_nodes:
        print("✓ GOOD: Both club leaders (0, 33) are in top 5")
    else:
        print("✗ BAD: Club leaders not in top 5")

    if len(expected_in_top_10) >= 6:
        print("✓ GOOD: Most expected high centrality nodes are in top 10")
    else:
        print("✗ BAD: Few expected high centrality nodes in top 10")

    # Check centrality distribution
    max_cent = max(values)
    min_cent = min(values)
    print(f"Centrality range: {min_cent:.4f} to {max_cent:.4f}")

    if max_cent > 0.1:  # Reasonable range for karate graph
        print("✓ GOOD: Centrality values are in reasonable range")
    else:
        print("✗ BAD: Centrality values seem too low")

    print(f"\nThis shows how betweenness centrality should work on a known graph.")
    print(f"If the C++ algorithm produces similar results, it's working correctly.")
    print(f"If it produces different results, there's a bug in the C++ implementation.")

if __name__ == "__main__":
    try:
        visualize_karate_graph()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install matplotlib networkx")
    except Exception as e:
        print(f"Error: {e}")

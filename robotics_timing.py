#!/usr/bin/env python3
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import os
import sys
import time
import numpy as np
import subprocess
import tempfile
import json

# Set environment variable for cuGraph support
os.environ['NX_CUGRAPH_AUTOCONFIG'] = 'True'

import networkx as nx

def benchmark_networkx_centrality(G, num_runs=5):
    """Benchmark pure NetworkX betweenness centrality using subprocess"""
    # Write graph to temporary CSV first
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            f.write(f"{u},{v},{weight}\n")
        temp_csv = f.name

    # Create a temporary script for pure NetworkX
    script_content = f'''
import networkx as nx
import time
import sys
import json

# Check if cuGraph is being used
print("  Checking cuGraph availability in subprocess...")
try:
    import cugraph
    print(f"  ❌ cuGraph is available in subprocess: {{cugraph.__version__}}")
except ImportError:
    print("  ✓ cuGraph not available in subprocess - using pure NetworkX")

# Load graph from edge list
G = nx.read_edgelist("{temp_csv}",
                    delimiter=',',
                    nodetype=int,
                    data=[('weight', float)],
                    create_using=nx.Graph())

times = []
for i in range({num_runs}):
    start_time = time.time()
    centrality = nx.betweenness_centrality(G, normalized=False, endpoints=True)
    end_time = time.time()
    times.append(end_time - start_time)
    print(f"  NetworkX run {{i+1}}: {{times[-1]:.4f}}s")

# Print timing results
print(f"  Mean time: {{sum(times)/len(times):.4f}}s ± {{(max(times)-min(times))/2:.4f}}s")
print(f"  Min time: {{min(times):.4f}}s")
print(f"  Max time: {{max(times):.4f}}s")
'''

    # Write and run script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        # Run the script without cuGraph environment
        env = os.environ.copy()
        env.pop('NX_CUGRAPH_AUTOCONFIG', None)

        # Also disable any other cuGraph-related environment variables
        env.pop('CUDA_VISIBLE_DEVICES', None)

        result = subprocess.run([sys.executable, temp_script],
                              capture_output=True, text=True, env=env, timeout=300)

        if result.returncode != 0:
            print(f"  Error running pure NetworkX: {result.stderr}")
            return None

        # Parse timing from output
        lines = result.stdout.strip().split('\n')
        times = []
        for line in lines:
            if 'NetworkX run' in line and ':' in line:
                # Extract just the time part
                time_part = line.split(': ')[1].split('s')[0]
                times.append(float(time_part))

        if not times:
            print("  Failed to parse timing results")
            return None

        print(result.stdout)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'centrality': None  # We don't need centrality for timing
        }

    finally:
        # Clean up temp files
        os.unlink(temp_csv)
        os.unlink(temp_script)

def benchmark_nx_cugraph_centrality(G, num_runs=5):
    """Benchmark nx-cugraph betweenness centrality"""
    # Environment variable already set at module level

    # Test if cuGraph is actually available
    try:
        import cugraph
        print(f"  ✓ cuGraph module available: {cugraph.__version__}")
    except ImportError:
        print("  ❌ cuGraph module not available - falling back to NetworkX")

    # Check if nx-cugraph is working
    try:
        # Try to use cugraph directly
        import cugraph
        cugraph_G = cugraph.from_networkx(G)
        print("  ✓ Successfully created cuGraph graph from NetworkX")
    except Exception as e:
        print(f"  ❌ Failed to create cuGraph graph: {e}")

    times = []
    for i in range(num_runs):
        start_time = time.time()
        centrality = nx.betweenness_centrality(G, normalized=False, endpoints=True)
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  nx-cugraph run {i+1}: {times[-1]:.4f}s")

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'centrality': centrality
    }

def benchmark_direct_cugraph_centrality(G, num_runs=5):
    """Benchmark direct cuGraph betweenness centrality"""
    try:
        import cugraph
        print(f"  ✓ Using direct cuGraph: {cugraph.__version__}")
    except ImportError:
        print("  ❌ cuGraph not available")
        return {
            'mean_time': float('inf'),
            'std_time': 0,
            'min_time': float('inf'),
            'max_time': float('inf'),
            'centrality': {}
        }

    times = []
    for i in range(num_runs):
        start_time = time.time()

        # Convert NetworkX graph to cuGraph
        cugraph_G = cugraph.from_networkx(G)

        # Run cuGraph betweenness centrality
        centrality_result = cugraph.betweenness_centrality(cugraph_G, normalized=False)

        # Convert back to dictionary format
        centrality = dict(zip(centrality_result['vertex'], centrality_result['betweenness_centrality']))

        end_time = time.time()
        times.append(end_time - start_time)
        print(f"  Direct cuGraph run {i+1}: {times[-1]:.4f}s")

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'centrality': centrality
    }

def benchmark_local_cugraph_centrality(csv_file, G, num_runs=5):
    """Benchmark local cuGraph implementation via C++ test"""
    import subprocess
    import tempfile
    import os

    # Use all nodes as sources for fair comparison
    num_nodes = G.number_of_nodes()

    times = []
    centralities = []

    for i in range(num_runs):
        print(f"  Local cuGraph run {i+1}: ", end="", flush=True)
        start_time = time.time()

        # Run the C++ test
        try:
            result = subprocess.run([
                './cpp/build/tests/centrality/test_betweenness_centrality',
                '--input_file', csv_file,
                '--num_seeds', str(num_nodes)
            ], capture_output=True, text=True, timeout=300)

            end_time = time.time()
            run_time = end_time - start_time
            times.append(run_time)
            print(f"{run_time:.4f}s")

            if result.returncode == 0:
                # Parse centrality values from output
                lines = result.stdout.split('\n')
                centrality_values = []
                for line in lines:
                    if 'centrality[' in line and '=' in line:
                        try:
                            value = float(line.split('=')[1].strip())
                            centrality_values.append(value)
                        except:
                            pass
                centralities.append(centrality_values)
            else:
                print(f"    Error: {result.stderr}")
                centralities.append([])

        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            times.append(float('inf'))
            centralities.append([])
        except Exception as e:
            print(f"ERROR: {e}")
            times.append(float('inf'))
            centralities.append([])

    # Filter out failed runs
    valid_times = [t for t in times if t != float('inf')]
    valid_centralities = [c for c in centralities if c]

    if not valid_times:
        return {
            'mean_time': float('inf'),
            'std_time': 0,
            'min_time': float('inf'),
            'max_time': float('inf'),
            'centrality': {}
        }

    return {
        'mean_time': np.mean(valid_times),
        'std_time': np.std(valid_times),
        'min_time': np.min(valid_times),
        'max_time': np.max(valid_times),
        'centrality': valid_centralities[0] if valid_centralities else {}
    }

def compare_centrality_results(nx_result, nx_cugraph_result, direct_cugraph_result, local_cugraph_result):
    """Compare centrality results between implementations"""
    print("\n" + "="*60)
    print("CENTRALITY RESULTS COMPARISON")
    print("="*60)

    if nx_result['centrality'] and nx_cugraph_result['centrality']:
        nx_values = list(nx_result['centrality'].values())
        nx_cugraph_values = list(nx_cugraph_result['centrality'].values())

        if len(nx_values) == len(nx_cugraph_values):
            diff = np.abs(np.array(nx_values) - np.array(nx_cugraph_values))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"NetworkX vs nx-cugraph:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Results match: {'Yes' if max_diff < 1e-6 else 'No'}")
        else:
            print(f"NetworkX vs nx-cugraph: Different number of nodes ({len(nx_values)} vs {len(nx_cugraph_values)})")

    if local_cugraph_result['centrality']:
        print(f"Local cuGraph: {len(local_cugraph_result['centrality'])} centrality values")
        if len(local_cugraph_result['centrality']) > 0:
            print(f"  Sample values: {local_cugraph_result['centrality'][:5]}")
    else:
        print("Local cuGraph: No centrality results available")

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 benchmark_centrality.py <csv_file> [num_runs]")
        print("Example: python3 benchmark_centrality.py Maps/warehouse_small.csv 3")
        return

    csv_file = sys.argv[1]
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"BENCHMARKING CENTRALITY ALGORITHMS")
    print(f"CSV file: {csv_file}")
    print(f"Number of runs: {num_runs}")
    print("="*60)

    # Load graph
    print("Loading graph...")
    G = nx.read_edgelist(csv_file,
                        delimiter=',',
                        nodetype=int,
                        data=[('weight', float)],
                        create_using=nx.Graph())

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()

    # Benchmark NetworkX
    print("1. BENCHMARKING PURE NETWORKX")
    print("-" * 40)
    nx_result = benchmark_networkx_centrality(G, num_runs)
    if nx_result is None:
        print("  Failed to run pure NetworkX implementation")
        return
    print(f"  Mean time: {nx_result['mean_time']:.4f}s ± {nx_result['std_time']:.4f}s")
    print(f"  Min time: {nx_result['min_time']:.4f}s")
    print(f"  Max time: {nx_result['max_time']:.4f}s")
    print()

    # Benchmark nx-cugraph
    print("2. BENCHMARKING NX-CUGRAPH")
    print("-" * 40)
    nx_cugraph_result = benchmark_nx_cugraph_centrality(G, num_runs)
    print(f"  Mean time: {nx_cugraph_result['mean_time']:.4f}s ± {nx_cugraph_result['std_time']:.4f}s")
    print(f"  Min time: {nx_cugraph_result['min_time']:.4f}s")
    print(f"  Max time: {nx_cugraph_result['max_time']:.4f}s")
    print()

    # Benchmark direct cuGraph
    print("3. BENCHMARKING DIRECT CUGRAPH")
    print("-" * 40)
    direct_cugraph_result = benchmark_direct_cugraph_centrality(G, num_runs)
    if direct_cugraph_result['mean_time'] != float('inf'):
        print(f"  Mean time: {direct_cugraph_result['mean_time']:.4f}s ± {direct_cugraph_result['std_time']:.4f}s")
        print(f"  Min time: {direct_cugraph_result['min_time']:.4f}s")
        print(f"  Max time: {direct_cugraph_result['max_time']:.4f}s")
    else:
        print("  Failed to run direct cuGraph implementation")
    print()

    # Benchmark local cuGraph
    print("4. BENCHMARKING LOCAL CUGRAPH")
    print("-" * 40)
    local_cugraph_result = benchmark_local_cugraph_centrality(csv_file, G, num_runs)
    if local_cugraph_result['mean_time'] != float('inf'):
        print(f"  Mean time: {local_cugraph_result['mean_time']:.4f}s ± {local_cugraph_result['std_time']:.4f}s")
        print(f"  Min time: {local_cugraph_result['min_time']:.4f}s")
        print(f"  Max time: {local_cugraph_result['max_time']:.4f}s")
    else:
        print("  Failed to run local cuGraph implementation")
    print()

    # Compare results
    compare_centrality_results(nx_result, nx_cugraph_result, direct_cugraph_result, local_cugraph_result)

    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    if local_cugraph_result['mean_time'] != float('inf'):
        speedup_nx = nx_result['mean_time'] / local_cugraph_result['mean_time']
        speedup_nx_cugraph = nx_cugraph_result['mean_time'] / local_cugraph_result['mean_time']

        print(f"Local cuGraph vs NetworkX: {speedup_nx:.2f}x speedup")
        print(f"Local cuGraph vs nx-cugraph: {speedup_nx_cugraph:.2f}x speedup")
        print(f"nx-cugraph vs NetworkX: {nx_result['mean_time'] / nx_cugraph_result['mean_time']:.2f}x speedup")
    else:
        print(f"nx-cugraph vs NetworkX: {nx_result['mean_time'] / nx_cugraph_result['mean_time']:.2f}x speedup")
        print("Local cuGraph: Not available")

if __name__ == "__main__":
    main()

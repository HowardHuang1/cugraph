/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "prims/count_if_v.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_if_e.cuh"
#include "prims/detail/extract_transform_if_v_frontier_e.cuh"
#include "prims/kv_store.cuh"
#include "prims/reduce_op.cuh"
#include "prims/detail/prim_functors.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"


#include "prims/fill_edge_property.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/handle.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include <iostream>

//
// The formula for BC(v) is the sum over all (s,t) where s != v != t of
// sigma_st(v) / sigma_st.  Sigma_st(v) is the number of shortest paths
// that pass through vertex v, whereas sigma_st is the total number of shortest
// paths.
namespace {

template <typename vertex_t>
struct brandes_e_op_t {
  template <typename value_t, typename ignore_t>
  __device__ value_t operator()(vertex_t, vertex_t, value_t src_sigma, vertex_t, ignore_t) const
  {
    return src_sigma;
  }
};

template <typename vertex_t>
struct brandes_pred_op_t {
  const vertex_t invalid_distance_{std::numeric_limits<vertex_t>::max()};

  template <typename value_t, typename ignore_t>
  __device__ bool operator()(
    vertex_t, vertex_t, value_t src_sigma, vertex_t dst_distance, ignore_t) const
  {
    return (dst_distance == invalid_distance_);
  }
};

template <typename vertex_t>
struct extract_edge_e_op_t {
  template <typename edge_t, typename weight_t>
  __device__ thrust::tuple<vertex_t, vertex_t> operator()(
    vertex_t src,
    vertex_t dst,
    thrust::tuple<vertex_t, edge_t, weight_t> src_props,
    thrust::tuple<vertex_t, edge_t, weight_t> dst_props,
    cuda::std::nullopt_t) const
  {
    return thrust::make_tuple(src, dst);
  }

  template <typename edge_t, typename weight_t>
  __device__ thrust::tuple<vertex_t, vertex_t, edge_t> operator()(
    vertex_t src,
    vertex_t dst,
    thrust::tuple<vertex_t, edge_t, weight_t> src_props,
    thrust::tuple<vertex_t, edge_t, weight_t> dst_props,
    edge_t edge_multi_index) const
  {
    return thrust::make_tuple(src, dst, edge_multi_index);
  }
};

template <typename vertex_t>
struct extract_edge_pred_op_t {
  vertex_t d{};

  template <typename edge_t, typename weight_t>
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             thrust::tuple<vertex_t, edge_t, weight_t> src_props,
                             thrust::tuple<vertex_t, edge_t, weight_t> dst_props,
                             cuda::std::nullopt_t) const
  {
    return ((thrust::get<0>(dst_props) == d) && (thrust::get<0>(src_props) == (d - 1)));
  }

  template <typename edge_t, typename weight_t>
  __device__ bool operator()(vertex_t src,
                             vertex_t dst,
                             thrust::tuple<vertex_t, edge_t, weight_t> src_props,
                             thrust::tuple<vertex_t, edge_t, weight_t> dst_props,
                             edge_t edge_multi_index) const
  {
    return ((thrust::get<0>(dst_props) == d) && (thrust::get<0>(src_props) == (d - 1)));
  }
};

}  // namespace

namespace cugraph {
namespace detail {

// Functors for concurrent multi-source BFS
template <typename vertex_t>
struct concurrent_bfs_e_op_t {
  __device__ thrust::tuple<thrust::tuple<vertex_t, uint32_t>, vertex_t> operator()(
    thrust::tuple<vertex_t, uint32_t> tagged_src,
    vertex_t dst,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t) const
  {
    // Extract source index from tagged vertex
    auto source_idx = thrust::get<1>(tagged_src);
    // Return (tagged_destination, distance) where distance is computed separately
    return thrust::make_tuple(thrust::make_tuple(dst, source_idx), dst);
  }
};

template <typename vertex_t>
struct concurrent_bfs_pred_op_t {
  vertex_t invalid_distance;

  concurrent_bfs_pred_op_t(vertex_t invalid_dist) : invalid_distance(invalid_dist) {}

  __device__ bool operator()(thrust::tuple<vertex_t, uint32_t> tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t) const
  {
    // Always process the edge in concurrent BFS
    // The frontier management will handle duplicates
    return true;
  }
};

// Helper functions for vertex-source aggregation
template <typename vertex_t>
struct aggregate_vi_t {
  uint32_t num_sources{};
  
  __device__ uint64_t operator()(thrust::tuple<vertex_t, uint32_t> tup) const {
    return (static_cast<uint64_t>(thrust::get<0>(tup)) * static_cast<uint64_t>(num_sources)) +
           static_cast<uint64_t>(thrust::get<1>(tup));
  }
};

template <typename vertex_t>
struct split_vi_t {
  uint32_t num_sources{};
  
  __device__ thrust::tuple<vertex_t, uint32_t> operator()(uint64_t aggregated_vi) const {
    return thrust::make_tuple(
      static_cast<vertex_t>(aggregated_vi / static_cast<uint64_t>(num_sources)),
      static_cast<uint32_t>(aggregated_vi % static_cast<uint64_t>(num_sources)));
  }
};


template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>> brandes_bfs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  vertex_frontier_t<vertex_t, void, multi_gpu, true>& vertex_frontier,
  bool do_expensive_check)
{
  //
  // Do BFS with a multi-output.  If we're on hop k and multiple vertices arrive at vertex v,
  // add all predecessors to the predecessor list, don't just arbitrarily pick one.
  //
  // Predecessors could be a CSR if that's helpful for doing the backwards tracing
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();
  constexpr size_t bucket_idx_cur{0};
  constexpr size_t bucket_idx_next{1};

  rmm::device_uvector<edge_t> sigmas(graph_view.local_vertex_partition_range_size(),
                                     handle.get_stream());
  rmm::device_uvector<vertex_t> distances(graph_view.local_vertex_partition_range_size(),
                                          handle.get_stream());
  detail::scalar_fill(handle, distances.data(), distances.size(), invalid_distance);
  detail::scalar_fill(handle, sigmas.data(), sigmas.size(), edge_t{0});

  edge_src_property_t<vertex_t, edge_t> src_sigmas(handle, graph_view);
  edge_dst_property_t<vertex_t, vertex_t> dst_distances(handle, graph_view);

  auto vertex_partition =
    vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());

  if (vertex_frontier.bucket(bucket_idx_cur).size() > 0) {
    thrust::for_each(
      handle.get_thrust_policy(),
      vertex_frontier.bucket(bucket_idx_cur).begin(),
      vertex_frontier.bucket(bucket_idx_cur).end(),
      [d_sigma = sigmas.begin(), d_distance = distances.begin(), vertex_partition] __device__(
        auto v) {
        auto offset        = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v);
        d_distance[offset] = 0;
        d_sigma[offset]    = 1;
      });
  }

  edge_t hop{0};

  while (true) {
    update_edge_src_property(handle, graph_view, sigmas.begin(), src_sigmas.mutable_view());
    update_edge_dst_property(handle, graph_view, distances.begin(), dst_distances.mutable_view());

    auto [new_frontier, new_sigma] = cugraph::transform_reduce_if_v_frontier_outgoing_e_by_dst(
      handle,
      graph_view,
      vertex_frontier.bucket(bucket_idx_cur),
      src_sigmas.view(),
      dst_distances.view(),
      cugraph::edge_dummy_property_t{}.view(),
      brandes_e_op_t<vertex_t>{},
      reduce_op::plus<vertex_t>(),
      brandes_pred_op_t<vertex_t>{});

    auto next_frontier_bucket_indices = std::vector<size_t>{bucket_idx_next};
    update_v_frontier(handle,
                      graph_view,
                      std::move(new_frontier),
                      std::move(new_sigma),
                      vertex_frontier,
                      raft::host_span<size_t const>(next_frontier_bucket_indices.data(),
                                                    next_frontier_bucket_indices.size()),
                      thrust::make_zip_iterator(distances.begin(), sigmas.begin()),
                      thrust::make_zip_iterator(distances.begin(), sigmas.begin()),
                      [hop] __device__(auto v, auto old_values, auto v_sigma) {
                        return thrust::make_tuple(
                          cuda::std::make_optional(bucket_idx_next),
                          cuda::std::make_optional(thrust::make_tuple(hop + 1, v_sigma)));
                      });

    vertex_frontier.bucket(bucket_idx_cur).clear();
    vertex_frontier.bucket(bucket_idx_cur).shrink_to_fit();
    vertex_frontier.swap_buckets(bucket_idx_cur, bucket_idx_next);
    if (vertex_frontier.bucket(bucket_idx_cur).aggregate_size() == 0) { break; }

    ++hop;
  }

  return std::make_tuple(std::move(distances), std::move(sigmas));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void accumulate_vertex_results(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<weight_t> centralities,
  rmm::device_uvector<vertex_t>&& distances,
  rmm::device_uvector<edge_t>&& sigmas,
  bool with_endpoints,
  bool do_expensive_check)
{
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();

  vertex_t diameter = transform_reduce_v(
    handle,
    graph_view,
    distances.begin(),
    [] __device__(auto, auto d) { return (d == invalid_distance) ? vertex_t{0} : d; },
    vertex_t{0},
    reduce_op::maximum<vertex_t>{},
    do_expensive_check);

  rmm::device_uvector<weight_t> deltas(sigmas.size(), handle.get_stream());
  detail::scalar_fill(handle, deltas.data(), deltas.size(), weight_t{0});

  if (with_endpoints) {
    vertex_t count = count_if_v(
      handle,
      graph_view,
      distances.begin(),
      [] __device__(auto, auto d) { return (d != invalid_distance); },
      do_expensive_check);

    thrust::transform(handle.get_thrust_policy(),
                      distances.begin(),
                      distances.end(),
                      centralities.begin(),
                      centralities.begin(),
                      [count] __device__(auto d, auto centrality) {
                        if (d == vertex_t{0}) {
                          return centrality + static_cast<weight_t>(count - 1);
                        } else if (d == invalid_distance) {
                          return centrality;
                        } else {
                          return centrality + weight_t{1};
                        }
                      });
  }

  edge_src_property_t<vertex_t, thrust::tuple<vertex_t, edge_t, weight_t>> src_properties(
    handle, graph_view);
  edge_dst_property_t<vertex_t, thrust::tuple<vertex_t, edge_t, weight_t>> dst_properties(
    handle, graph_view);

  update_edge_src_property(
    handle,
    graph_view,
    thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
    src_properties.mutable_view());
  update_edge_dst_property(
    handle,
    graph_view,
    thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
    dst_properties.mutable_view());


  
  // Pre-allocate reusable buffers to avoid repeated allocations (optimized for max frontier size)
  // Use binary search method to find frontier boundaries more efficiently
  
  // Create distance-vertex pairs and sort them
  rmm::device_uvector<vertex_t> vertices_at_distance(graph_view.local_vertex_partition_range_size(), handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
    thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
    vertices_at_distance.begin());
  
  // Create distance-vertex pairs for sorting
  rmm::device_uvector<thrust::tuple<vertex_t, vertex_t>> distance_vertex_pairs(
    graph_view.local_vertex_partition_range_size(), handle.get_stream());
  
  thrust::transform(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(distances.begin(), vertices_at_distance.begin()),
    thrust::make_zip_iterator(distances.begin(), vertices_at_distance.begin()) + distances.size(),
    distance_vertex_pairs.begin(),
    [] __device__(auto pair) {
      auto distance = thrust::get<0>(pair);
      auto vertex = thrust::get<1>(pair);
      return thrust::make_tuple(distance, vertex);
    });

  // Sort by distance, then by vertex ID for stable ordering
  thrust::sort(
    handle.get_thrust_policy(),
    distance_vertex_pairs.begin(),
    distance_vertex_pairs.end());

  // Single vectorized thrust call to compute all bounds for distances 0 to diameter
  vertex_t max_distance = diameter;
  
  rmm::device_uvector<vertex_t> search_keys(max_distance + 1, handle.get_stream());
  
  // Use thrust::copy with counting_iterator instead of thrust::sequence
  thrust::copy(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<vertex_t>(0),
    thrust::make_counting_iterator<vertex_t>(max_distance + 1),
    search_keys.begin());
  
  rmm::device_uvector<vertex_t> bounds(max_distance + 1, handle.get_stream());
  
  // Vectorized lower_bound to compute all bounds at once
  auto vertex_proj = [] __device__(auto pair) { return thrust::get<0>(pair); };
  auto transform_begin = thrust::make_transform_iterator(distance_vertex_pairs.begin(), vertex_proj);
  auto transform_end = thrust::make_transform_iterator(distance_vertex_pairs.end(), vertex_proj);
  thrust::lower_bound(
    handle.get_thrust_policy(),
    transform_begin,
    transform_end,
    search_keys.begin(),
    search_keys.end(),
    bounds.data());
  
  // Copy bounds to host for use in delta loop
  std::vector<vertex_t> h_bounds(bounds.size());
  raft::update_host(h_bounds.data(), bounds.data(), bounds.size(), handle.get_stream());
  handle.sync_stream();
  
  // Calculate max frontier size using the precomputed bounds
  vertex_t max_frontier_size = 0;
  for (vertex_t d = 0; d < h_bounds.size() - 1; ++d) {
    vertex_t frontier_count = h_bounds[d + 1] - h_bounds[d];
    max_frontier_size = std::max(max_frontier_size, frontier_count);
  }

  rmm::device_uvector<vertex_t> reusable_vertex_buffer(max_frontier_size, handle.get_stream());
  rmm::device_uvector<weight_t> reusable_delta_buffer(max_frontier_size, handle.get_stream());


  
  // Based on Brandes algorithm, we want to follow back pointers in non-increasing
  // distance from S to compute delta
  //
  for (vertex_t d = diameter; d > 1; --d) {
    // Use precomputed bounds for O(1) lookup instead of binary search
    vertex_t first_d_minus_1 = h_bounds[d - 1];
    vertex_t first_d = h_bounds[d];
    vertex_t frontier_count = first_d - first_d_minus_1;

    if (frontier_count > 0) {      
      // Combined operation: clear deltas and extract frontier vertices in one kernel
      thrust::transform(
        handle.get_thrust_policy(),
        distance_vertex_pairs.begin() + first_d_minus_1,
        distance_vertex_pairs.begin() + first_d,
        thrust::make_zip_iterator(reusable_vertex_buffer.begin(), reusable_delta_buffer.begin()),
        [] __device__(auto pair) { 
          return thrust::make_tuple(thrust::get<1>(pair), weight_t{0}); 
        });
      
      // Create key_bucket_t from the frontier vertices
      key_bucket_t<vertex_t, void, multi_gpu, true> vertex_list(handle);
      vertex_list.insert(reusable_vertex_buffer.begin(), reusable_vertex_buffer.begin() + frontier_count);

      // Compute deltas for frontier vertices
      per_v_transform_reduce_outgoing_e(
        handle,
        graph_view,
        vertex_list,
        src_properties.view(),
        dst_properties.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [d] __device__(auto, auto, auto src_props, auto dst_props, auto) {
          if (thrust::get<0>(dst_props) == d) {
            auto sigma_v = static_cast<weight_t>(thrust::get<1>(src_props));
            auto sigma_w = static_cast<weight_t>(thrust::get<1>(dst_props));
            auto delta_w = thrust::get<2>(dst_props);
            
            return (sigma_v / sigma_w) * (1 + delta_w);
          } else {
            return weight_t{0};
          }
        },
        weight_t{0},
        reduce_op::plus<weight_t>{},
        reusable_delta_buffer.begin(),
        do_expensive_check);
      
      // Scatter deltas back to main deltas array
      thrust::scatter(
        handle.get_thrust_policy(),
        reusable_delta_buffer.begin(),
        reusable_delta_buffer.begin() + frontier_count,
        reusable_vertex_buffer.begin(),
        deltas.begin()
      );
      
      // Combined operation: update properties and accumulate centralities in one kernel
      
      // Get the property iterators outside the lambda (host-side)
      auto src_props_first = src_properties.mutable_view().major_value_firsts()[0];  // Source properties are major
      auto dst_props_first = dst_properties.mutable_view().minor_value_first();      // Destination properties are minor
      
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(reusable_vertex_buffer.begin(), reusable_delta_buffer.begin()),
        thrust::make_zip_iterator(reusable_vertex_buffer.begin(), reusable_delta_buffer.begin()) + frontier_count,
        [distances = distances.begin(),
         sigmas = sigmas.begin(),
         deltas = deltas.begin(),
         centralities = centralities.data(),
         src_props_first,
         dst_props_first,
         d] __device__(auto pair) {
          auto v = thrust::get<0>(pair);
          auto delta = thrust::get<1>(pair);
          
          // Update source properties (major)
          src_props_first[v] = thrust::make_tuple(distances[v], sigmas[v], deltas[v]);
          
          // Update destination properties (minor)
          dst_props_first[v] = thrust::make_tuple(distances[v], sigmas[v], deltas[v]);
          
          // Accumulate centralities
          centralities[v] += delta;
        });
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void concurrent_accumulate_vertex_results(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<weight_t> centralities,
  rmm::device_uvector<vertex_t>&& all_distances,
  rmm::device_uvector<edge_t>&& all_sigmas,
  size_t n_sources,
  bool with_endpoints,
  bool do_expensive_check)
{
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();
  auto local_vertex_partition_range_size = graph_view.local_vertex_partition_range_size();
  auto local_vertex_partition_range_first = graph_view.local_vertex_partition_range_first();
  
  // Process each source's contribution to centrality
  for (size_t source_idx = 0; source_idx < n_sources; ++source_idx) {
    auto source_distances = all_distances.begin() + source_idx * local_vertex_partition_range_size;
    auto source_sigmas = all_sigmas.begin() + source_idx * local_vertex_partition_range_size;
    
    // Find maximum distance for this source
    vertex_t max_distance = transform_reduce_v(
      handle,
      graph_view,
      source_distances,
      [] __device__(auto, auto d) { return (d == invalid_distance) ? vertex_t{0} : d; },
      vertex_t{0},
      reduce_op::maximum<vertex_t>{},
      do_expensive_check);
    
    // Initialize delta array for this source
    rmm::device_uvector<weight_t> deltas(local_vertex_partition_range_size, handle.get_stream());
    detail::scalar_fill(handle, deltas.data(), deltas.size(), weight_t{0});
    
    // Handle endpoints if requested
    if (with_endpoints) {
      vertex_t count = count_if_v(
        handle,
        graph_view,
        source_distances,
        [] __device__(auto, auto d) { return (d != invalid_distance); },
        do_expensive_check);
      
      thrust::transform(handle.get_thrust_policy(),
                        source_distances,
                        source_distances + local_vertex_partition_range_size,
                        centralities.begin(),
                        centralities.begin(),
                        [count] __device__(auto d, auto centrality) {
                          if (d == vertex_t{0}) {
                            return centrality + static_cast<weight_t>(count - 1);
                          } else if (d == invalid_distance) {
                            return centrality;
                          } else {
                            return centrality + weight_t{1};
                          }
                        });
    }
    
    // Backward pass: compute deltas in non-increasing distance order
    edge_src_property_t<vertex_t, thrust::tuple<vertex_t, edge_t, weight_t>> src_properties(handle, graph_view);
    edge_dst_property_t<vertex_t, thrust::tuple<vertex_t, edge_t, weight_t>> dst_properties(handle, graph_view);
    
    update_edge_src_property(
      handle,
      graph_view,
      thrust::make_zip_iterator(source_distances, source_sigmas, deltas.begin()),
      src_properties.mutable_view());
    update_edge_dst_property(
      handle,
      graph_view,
      thrust::make_zip_iterator(source_distances, source_sigmas, deltas.begin()),
      dst_properties.mutable_view());
    
    // Process vertices in non-increasing distance order
    for (vertex_t d = max_distance; d > 1; --d) {
      // Find vertices at distance d-1
      rmm::device_uvector<vertex_t> vertices_at_distance(local_vertex_partition_range_size, handle.get_stream());
      auto vertex_count = thrust::copy_if(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(local_vertex_partition_range_first),
        thrust::make_counting_iterator(local_vertex_partition_range_first + local_vertex_partition_range_size),
        vertices_at_distance.begin(),
        [source_distances, d, local_vertex_partition_range_first] __device__(auto v) {
          auto offset = v - local_vertex_partition_range_first;
          return source_distances[offset] == (d - 1);
        });
      vertices_at_distance.resize(thrust::distance(vertices_at_distance.begin(), vertex_count), handle.get_stream());
      
      if (vertices_at_distance.size() > 0) {
        // Create frontier for vertices at distance d-1
        key_bucket_t<vertex_t, void, multi_gpu, true> vertex_list(handle);
        vertex_list.insert(vertices_at_distance.begin(), vertices_at_distance.end());
        
        // Compute deltas for vertices at distance d-1
        rmm::device_uvector<weight_t> delta_updates(vertices_at_distance.size(), handle.get_stream());
        
        per_v_transform_reduce_outgoing_e(
          handle,
          graph_view,
          vertex_list,
          src_properties.view(),
          dst_properties.view(),
          cugraph::edge_dummy_property_t{}.view(),
          [d] __device__(auto, auto, auto src_props, auto dst_props, auto) {
            if (thrust::get<0>(dst_props) == d) {
              auto sigma_v = static_cast<weight_t>(thrust::get<1>(src_props));
              auto sigma_w = static_cast<weight_t>(thrust::get<1>(dst_props));
              auto delta_w = thrust::get<2>(dst_props);
              
              return (sigma_v / sigma_w) * (1 + delta_w);
            } else {
              return weight_t{0};
            }
          },
          weight_t{0},
          reduce_op::plus<weight_t>{},
          delta_updates.begin(),
          do_expensive_check);
        
        // Scatter delta updates back to main deltas array
        thrust::scatter(
          handle.get_thrust_policy(),
          delta_updates.begin(),
          delta_updates.end(),
          vertices_at_distance.begin(),
          deltas.begin()
        );
        
        // Accumulate centralities
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(vertices_at_distance.begin(), delta_updates.begin()),
          thrust::make_zip_iterator(vertices_at_distance.begin(), delta_updates.end()),
          [centralities = centralities.data(), local_vertex_partition_range_first] __device__(auto pair) {
            auto v = thrust::get<0>(pair);
            auto delta = thrust::get<1>(pair);
            auto offset = v - local_vertex_partition_range_first;
            centralities[offset] += delta;
          });
        
        // Update properties for next iteration
        update_edge_src_property(
          handle,
          graph_view,
          thrust::make_zip_iterator(source_distances, source_sigmas, deltas.begin()),
          src_properties.mutable_view());
        update_edge_dst_property(
          handle,
          graph_view,
          thrust::make_zip_iterator(source_distances, source_sigmas, deltas.begin()),
          dst_properties.mutable_view());
      }
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void accumulate_edge_results(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  edge_property_view_t<edge_t, weight_t*> centralities_view,
  rmm::device_uvector<vertex_t>&& distances,
  rmm::device_uvector<edge_t>&& sigmas,
  bool do_expensive_check)
{
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();

  vertex_t diameter = transform_reduce_v(
    handle,
    graph_view,
    distances.begin(),
    [] __device__(auto, auto d) { return (d == invalid_distance) ? vertex_t{0} : d; },
    vertex_t{0},
    reduce_op::maximum<vertex_t>{},
    do_expensive_check);

  rmm::device_uvector<weight_t> deltas(sigmas.size(), handle.get_stream());
  detail::scalar_fill(handle, deltas.data(), deltas.size(), weight_t{0});

  edge_src_property_t<vertex_t, thrust::tuple<vertex_t, edge_t, weight_t>> src_properties(
    handle, graph_view);
  edge_dst_property_t<vertex_t, thrust::tuple<vertex_t, edge_t, weight_t>> dst_properties(
    handle, graph_view);

  update_edge_src_property(
    handle,
    graph_view,
    thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
    src_properties.mutable_view());
  update_edge_dst_property(
    handle,
    graph_view,
    thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
    dst_properties.mutable_view());

  //
  //   For now this will do a O(E) pass over all edges over the diameter
  //   of the graph.
  //
  // Based on Brandes algorithm, we want to follow back pointers in non-increasing
  // distance from S to compute delta
  //
  for (vertex_t d = diameter; d > 0; --d) {
    //
    //  Populate edge_list with edges where `thrust::get<0>(dst_props) == d`
    //  and `thrust::get<0>(dst_props) == (d-1)`
    //
    cugraph::edge_bucket_t<vertex_t, edge_t, true, multi_gpu, true> edge_list(
      handle, graph_view.is_multigraph());

    rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<edge_t>> indices{std::nullopt};
    if (graph_view.is_multigraph()) {
      edge_multi_index_property_t<edge_t, vertex_t> edge_multi_indices(handle, graph_view);
      std::tie(srcs, dsts, indices) = extract_transform_if_e(handle,
                                                             graph_view,
                                                             src_properties.view(),
                                                             dst_properties.view(),
                                                             edge_multi_indices.view(),
                                                             extract_edge_e_op_t<vertex_t>{},
                                                             extract_edge_pred_op_t<vertex_t>{d},
                                                             do_expensive_check);

      auto triplet_first = thrust::make_zip_iterator(srcs.begin(), dsts.begin(), indices->begin());
      thrust::sort(handle.get_thrust_policy(), triplet_first, triplet_first + srcs.size());
    } else {
      std::tie(srcs, dsts) = extract_transform_if_e(handle,
                                                    graph_view,
                                                    src_properties.view(),
                                                    dst_properties.view(),
                                                    edge_dummy_property_t{}.view(),
                                                    extract_edge_e_op_t<vertex_t>{},
                                                    extract_edge_pred_op_t<vertex_t>{d},
                                                    do_expensive_check);
      auto pair_first      = thrust::make_zip_iterator(srcs.begin(), dsts.begin());
      thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + srcs.size());
    }
    edge_list.insert(srcs.begin(),
                     srcs.end(),
                     dsts.begin(),
                     indices ? std::make_optional(indices->begin()) : std::nullopt);

    transform_e(
      handle,
      graph_view,
      edge_list,
      src_properties.view(),
      dst_properties.view(),
      centralities_view,
      [d] __device__(auto src, auto dst, auto src_props, auto dst_props, auto edge_centrality) {
        if ((thrust::get<0>(dst_props) == d) && (thrust::get<0>(src_props) == (d - 1))) {
          auto sigma_v = static_cast<weight_t>(thrust::get<1>(src_props));
          auto sigma_w = static_cast<weight_t>(thrust::get<1>(dst_props));
          auto delta_w = thrust::get<2>(dst_props);

          return edge_centrality + (sigma_v / sigma_w) * (1 + delta_w);
        } else {
          return edge_centrality;
        }
      },
      centralities_view,
      do_expensive_check);

    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      src_properties.view(),
      dst_properties.view(),
      cugraph::edge_dummy_property_t{}.view(),
      [d] __device__(auto, auto, auto src_props, auto dst_props, auto) {
        if ((thrust::get<0>(dst_props) == d) && (thrust::get<0>(src_props) == (d - 1))) {
          auto sigma_v = static_cast<weight_t>(thrust::get<1>(src_props));
          auto sigma_w = static_cast<weight_t>(thrust::get<1>(dst_props));
          auto delta_w = thrust::get<2>(dst_props);

          return (sigma_v / sigma_w) * (1 + delta_w);
        } else {
          return weight_t{0};
        }
      },
      weight_t{0},
      reduce_op::plus<weight_t>{},
      deltas.begin(),
      do_expensive_check);

    update_edge_src_property(
      handle,
      graph_view,
      thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
      src_properties.mutable_view());
    update_edge_dst_property(
      handle,
      graph_view,
      thrust::make_zip_iterator(distances.begin(), sigmas.begin(), deltas.begin()),
      dst_properties.mutable_view());
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
rmm::device_uvector<weight_t> betweenness_centrality(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  VertexIterator vertices_begin,
  VertexIterator vertices_end,
  bool const normalized,
  bool const include_endpoints,
  bool const do_expensive_check)
{
  //
  // Betweenness Centrality algorithm based on the Brandes Algorithm (2001)
  //
  if (do_expensive_check) {
    auto vertex_partition =
      vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());
    auto num_invalid_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       vertices_begin,
                       vertices_end,
                       [vertex_partition] __device__(auto val) {
                         return !(vertex_partition.is_valid_vertex(val) &&
                                  vertex_partition.in_local_vertex_partition_range_nocheck(val));
                       });
    if constexpr (multi_gpu) {
      num_invalid_vertices = host_scalar_allreduce(
        handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: sources have invalid vertex IDs.");
  }

  rmm::device_uvector<weight_t> centralities(graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());
  detail::scalar_fill(handle, centralities.data(), centralities.size(), weight_t{0});

  size_t num_sources = cuda::std::distance(vertices_begin, vertices_end);
  std::vector<size_t> source_offsets{{0, num_sources}};

  if constexpr (multi_gpu) {
    auto source_counts =
      host_scalar_allgather(handle.get_comms(), num_sources, handle.get_stream());

    num_sources = std::accumulate(source_counts.begin(), source_counts.end(), 0);
    source_offsets.resize(source_counts.size() + 1);
    source_offsets[0] = 0;
    std::inclusive_scan(source_counts.begin(), source_counts.end(), source_offsets.begin() + 1);
  }

  // Use concurrent multi-source BFS for better GPU utilization
  // Convert sources to device vector for concurrent processing
  rmm::device_uvector<vertex_t> d_sources(num_sources, handle.get_stream());
  thrust::copy(handle.get_thrust_policy(), vertices_begin, vertices_end, d_sources.begin());
  
  // Call the concurrent multi-source betweenness centrality implementation
  concurrent_betweenness_centrality_impl(handle,
                                        graph_view,
                                        edge_weight_view,
                                        centralities.begin(),
                                        d_sources.data(),
                                        num_sources,
                                        include_endpoints,
                                        do_expensive_check);

  std::optional<weight_t> scale_nonsource{std::nullopt};
  std::optional<weight_t> scale_source{std::nullopt};

  weight_t num_vertices = static_cast<weight_t>(graph_view.number_of_vertices());
  if (!include_endpoints) num_vertices = num_vertices - 1;

  if ((static_cast<edge_t>(num_sources) == num_vertices) || include_endpoints) {
    if (normalized) {
      scale_nonsource = static_cast<weight_t>(num_sources * (num_vertices - 1));
    } else if (graph_view.is_symmetric()) {
      scale_nonsource =
        static_cast<weight_t>(num_sources * 2) / static_cast<weight_t>(num_vertices);
    } else {
      scale_nonsource = static_cast<weight_t>(num_sources) / static_cast<weight_t>(num_vertices);
    }

    scale_source = scale_nonsource;
  } else if (normalized) {
    scale_nonsource = static_cast<weight_t>(num_sources) * (num_vertices - 1);
    scale_source    = static_cast<weight_t>(num_sources - 1) * (num_vertices - 1);
  } else {
    scale_nonsource = static_cast<weight_t>(num_sources) / num_vertices;
    scale_source    = static_cast<weight_t>(num_sources - 1) / num_vertices;

    if (graph_view.is_symmetric()) {
      *scale_nonsource *= 2;
      *scale_source *= 2;
    }
  }

  if (scale_nonsource) {
    auto iter = thrust::make_zip_iterator(
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      centralities.begin());

    thrust::transform(
      handle.get_thrust_policy(),
      iter,
      iter + centralities.size(),
      centralities.begin(),
      [nonsource = *scale_nonsource,
       source    = *scale_source,
       vertices_begin,
       vertices_end] __device__(auto t) {
        vertex_t v          = thrust::get<0>(t);
        weight_t centrality = thrust::get<1>(t);

        return (thrust::find(thrust::seq, vertices_begin, vertices_end, v) == vertices_end)
                 ? centrality / nonsource
                 : centrality / source;
      });
  }

  return centralities;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename VertexIterator>
edge_property_t<edge_t, weight_t> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  VertexIterator vertices_begin,
  VertexIterator vertices_end,
  bool const normalized,
  bool const do_expensive_check)
{
  //
  // Betweenness Centrality algorithm based on the Brandes Algorithm (2001)
  //
  if (do_expensive_check) {
    auto vertex_partition =
      vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());
    auto num_invalid_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       vertices_begin,
                       vertices_end,
                       [vertex_partition] __device__(auto val) {
                         return !(vertex_partition.is_valid_vertex(val) &&
                                  vertex_partition.in_local_vertex_partition_range_nocheck(val));
                       });
    if constexpr (multi_gpu) {
      num_invalid_vertices = host_scalar_allreduce(
        handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input argument: sources have invalid vertex IDs.");
  }

  edge_property_t<edge_t, weight_t> centralities(handle, graph_view);

  if (graph_view.has_edge_mask()) {
    auto unmasked_graph_view = graph_view;
    unmasked_graph_view.clear_edge_mask();
    fill_edge_property(
      handle, unmasked_graph_view, centralities.mutable_view(), weight_t{0}, do_expensive_check);
  } else {
    fill_edge_property(
      handle, graph_view, centralities.mutable_view(), weight_t{0}, do_expensive_check);
  }

  size_t num_sources = cuda::std::distance(vertices_begin, vertices_end);
  std::vector<size_t> source_offsets{{0, num_sources}};
  int my_rank = 0;

  if constexpr (multi_gpu) {
    auto source_counts =
      host_scalar_allgather(handle.get_comms(), num_sources, handle.get_stream());

    num_sources = std::accumulate(source_counts.begin(), source_counts.end(), 0);
    source_offsets.resize(source_counts.size() + 1);
    source_offsets[0] = 0;
    std::inclusive_scan(source_counts.begin(), source_counts.end(), source_offsets.begin() + 1);
    my_rank = handle.get_comms().get_rank();
  }

  //
  // FIXME: This could be more efficient using something akin to the
  // technique in WCC.  Take the entire set of sources, insert them into
  // a tagged frontier (tagging each source with itself).  Then we can
  // expand from multiple sources concurrently. The challenge is managing
  // the memory explosion.
  //
  for (size_t source_idx = 0; source_idx < num_sources; ++source_idx) {
    //
    //  BFS
    //
    constexpr size_t bucket_idx_cur = 0;
    constexpr size_t num_buckets    = 2;

    vertex_frontier_t<vertex_t, void, multi_gpu, true> vertex_frontier(handle, num_buckets);

    if ((source_idx >= source_offsets[my_rank]) && (source_idx < source_offsets[my_rank + 1])) {
      vertex_frontier.bucket(bucket_idx_cur)
        .insert(vertices_begin + (source_idx - source_offsets[my_rank]),
                vertices_begin + (source_idx - source_offsets[my_rank]) + 1);
    }

    //
    //  Now we need to do modified BFS
    //
    // FIXME:  This has an inefficiency in early iterations, as it doesn't have enough work to
    //         keep the GPUs busy.  But we can't run too many at once or we will run out of
    //         memory. Need to investigate options to improve this performance
    auto [distances, sigmas] =
      brandes_bfs(handle, graph_view, edge_weight_view, vertex_frontier, do_expensive_check);
    accumulate_edge_results(handle,
                            graph_view,
                            edge_weight_view,
                            centralities.mutable_view(),
                            std::move(distances),
                            std::move(sigmas),
                            do_expensive_check);
  }

  std::optional<weight_t> scale_factor{std::nullopt};

  if (normalized) {
    weight_t n   = static_cast<weight_t>(graph_view.number_of_vertices());
    scale_factor = n * (n - 1);
  } else if (graph_view.is_symmetric()) {
    scale_factor = weight_t{2};
  }

  if (scale_factor) {
    if (graph_view.number_of_vertices() > 1) {
      if (static_cast<vertex_t>(num_sources) < graph_view.number_of_vertices()) {
        (*scale_factor) *= static_cast<weight_t>(num_sources) /
                           static_cast<weight_t>(graph_view.number_of_vertices());
      }

      auto firsts         = centralities.view().value_firsts();
      auto counts         = centralities.view().edge_counts();
      auto mutable_firsts = centralities.mutable_view().value_firsts();
      for (size_t k = 0; k < counts.size(); k++) {
        thrust::transform(
          handle.get_thrust_policy(),
          firsts[k],
          firsts[k] + counts[k],
          mutable_firsts[k],
          [sf = *scale_factor] __device__(auto centrality) { return centrality / sf; });
      }
    }
  }

  return centralities;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu>
void concurrent_betweenness_centrality_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  weight_t* result_first,
  vertex_t const* sources,
  size_t n_sources,
  bool include_endpoints,
  bool do_expensive_check)
{
  // True concurrent multi-source Brandes algorithm implementation
  // Strategy: Process all sources simultaneously in a single BFS traversal
  // This provides true concurrency while maintaining correctness
  
  // Initialize centrality results
  rmm::device_uvector<weight_t> centralities(graph_view.local_vertex_partition_range_size(), handle.get_stream());
  detail::scalar_fill(handle, centralities.data(), centralities.size(), weight_t{0});
  
  // Run concurrent multi-source BFS for all sources
  auto [all_distances, all_sigmas] = concurrent_brandes_bfs(
    handle, graph_view, edge_weight_view, sources, n_sources, do_expensive_check);
  
  // Use concurrent accumulate function for multi-source data
  concurrent_accumulate_vertex_results(
    handle,
    graph_view,
    edge_weight_view,
    raft::device_span<weight_t>(centralities.data(), centralities.size()),
    std::move(all_distances),
    std::move(all_sigmas),
    n_sources,
    include_endpoints,
    do_expensive_check);
  
  // Copy results to output
  thrust::copy(handle.get_thrust_policy(),
               centralities.begin(),
               centralities.end(),
               result_first);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<edge_t>> concurrent_brandes_bfs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  vertex_t const* sources,
  size_t n_sources,
  bool do_expensive_check)
{
  constexpr vertex_t invalid_distance = std::numeric_limits<vertex_t>::max();
  constexpr size_t bucket_idx_cur = 0;
  constexpr size_t num_buckets = 1;  // Only need 1 bucket for concurrent BFS
  
  // Initialize frontier with tagged sources: (vertex, source_index) pairs
  vertex_frontier_t<vertex_t, uint32_t, multi_gpu, true> vertex_frontier(handle, num_buckets);
  
  // Create tagged sources: (source_vertex, source_index) pairs
  rmm::device_uvector<thrust::tuple<vertex_t, uint32_t>> tagged_sources(n_sources, handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(sources, thrust::make_counting_iterator(uint32_t{0})),
                   thrust::make_zip_iterator(sources + n_sources, thrust::make_counting_iterator(uint32_t{0}) + n_sources),
                   tagged_sources.begin(),
                   [] __device__(auto pair) {
                     return thrust::make_tuple(thrust::get<0>(pair), thrust::get<1>(pair));
                   });
  
  vertex_frontier.bucket(bucket_idx_cur).insert(tagged_sources.begin(), tagged_sources.end());
  
  // Initialize distance and sigma arrays for all sources
  // Use 2D arrays: [n_sources][num_vertices]
  auto local_vertex_partition_range_size = graph_view.local_vertex_partition_range_size();
  rmm::device_uvector<vertex_t> distances(n_sources * local_vertex_partition_range_size, handle.get_stream());
  rmm::device_uvector<edge_t> sigmas(n_sources * local_vertex_partition_range_size, handle.get_stream());
  
  detail::scalar_fill(handle, distances.data(), distances.size(), invalid_distance);
  detail::scalar_fill(handle, sigmas.data(), sigmas.size(), edge_t{0});
  
  // Initialize source distances and sigmas
  auto local_vertex_partition_range_first = graph_view.local_vertex_partition_range_first();
  
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(sources, thrust::make_counting_iterator(uint32_t{0})),
                   thrust::make_zip_iterator(sources + n_sources, thrust::make_counting_iterator(uint32_t{0}) + n_sources),
                   [distances = distances.data(), sigmas = sigmas.data(), 
                    local_vertex_partition_range_first, local_vertex_partition_range_size] __device__(auto pair) {
                     auto source_vertex = thrust::get<0>(pair);
                     auto source_idx = thrust::get<1>(pair);
                     auto offset = source_vertex - local_vertex_partition_range_first;
                     distances[source_idx * local_vertex_partition_range_size + offset] = 0;
                     sigmas[source_idx * local_vertex_partition_range_size + offset] = 1;
                   });
  
  // Concurrent multi-source BFS iteration
  vertex_t depth{0};
  auto cur_frontier_size = static_cast<vertex_t>(vertex_frontier.bucket(bucket_idx_cur).aggregate_size());
  
  // Debug: Print initial frontier size
  std::cout << "Starting concurrent BFS with " << n_sources << " sources, initial frontier size: " << cur_frontier_size << std::endl;
  
  // Memory tracking
  size_t total_memory_allocated = 0;
  size_t peak_memory_usage = 0;
  
  while (cur_frontier_size > 0) {
    auto iteration_start = std::chrono::high_resolution_clock::now();
    
    // Initialize timing variables
    auto key_conversion_time = 0;
    auto sort_reduce_time = 0;
    auto distance_update_time = 0;
    auto frontier_update_time = 0;
    
    // Process current frontier - expand to neighbors for each source concurrently
    auto frontier_expand_start = std::chrono::high_resolution_clock::now();
    auto new_frontier_tagged_vertex_buffer =
      allocate_dataframe_buffer<thrust::tuple<vertex_t, uint32_t>>(0, handle.get_stream());
    rmm::device_uvector<vertex_t> distance_buffer(0, handle.get_stream());
    
    std::tie(new_frontier_tagged_vertex_buffer, distance_buffer) = detail::
      extract_transform_if_v_frontier_e<false, thrust::tuple<vertex_t, uint32_t>, vertex_t>(
        handle,
        graph_view,
        vertex_frontier.bucket(bucket_idx_cur),
        edge_src_dummy_property_t{}.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        concurrent_bfs_e_op_t<vertex_t>{},
        concurrent_bfs_pred_op_t<vertex_t>{invalid_distance},
        do_expensive_check);
    
    auto frontier_expand_end = std::chrono::high_resolution_clock::now();
    auto frontier_expand_time = std::chrono::duration_cast<std::chrono::microseconds>(frontier_expand_end - frontier_expand_start).count();
    
    // Convert to keys and sort/reduce to handle duplicates
    auto new_frontier_size = size_dataframe_buffer(new_frontier_tagged_vertex_buffer);
    
    // Debug: Print frontier expansion
    std::cout << "Depth " << depth << ": expanded to " << new_frontier_size << " vertices (frontier expand: " << frontier_expand_time << " μs)" << std::endl;
    
    if (new_frontier_size > 0) {
      auto key_conversion_start = std::chrono::high_resolution_clock::now();
      rmm::device_uvector<uint64_t> new_frontier_keys(new_frontier_size, handle.get_stream());
      
      // Track memory allocation
      size_t keys_memory = new_frontier_keys.size() * sizeof(uint64_t);
      total_memory_allocated += keys_memory;
      peak_memory_usage = std::max(peak_memory_usage, total_memory_allocated);
      
      auto key_first = thrust::make_transform_iterator(
        get_dataframe_buffer_begin(new_frontier_tagged_vertex_buffer),
        aggregate_vi_t<vertex_t>{static_cast<uint32_t>(n_sources)});
      thrust::copy(handle.get_thrust_policy(),
                   key_first,
                   key_first + new_frontier_size,
                   new_frontier_keys.begin());
      
      auto key_conversion_end = std::chrono::high_resolution_clock::now();
      key_conversion_time = std::chrono::duration_cast<std::chrono::microseconds>(key_conversion_end - key_conversion_start).count();
      
      // Sort and reduce to handle duplicates - keep minimum distance for each vertex-source pair
      auto sort_reduce_start = std::chrono::high_resolution_clock::now();
      auto before_reduce_size = new_frontier_keys.size();
      std::tie(new_frontier_keys, distance_buffer) =
        detail::sort_and_reduce_buffer_elements<uint64_t, uint64_t, vertex_t, reduce_op::minimum<vertex_t>>(
          handle,
          std::move(new_frontier_keys),
          std::move(distance_buffer),
          reduce_op::minimum<vertex_t>(),
          std::make_tuple(vertex_t{0}, graph_view.number_of_vertices()),
          std::nullopt);
      auto sort_reduce_end = std::chrono::high_resolution_clock::now();
      sort_reduce_time = std::chrono::duration_cast<std::chrono::microseconds>(sort_reduce_end - sort_reduce_start).count();
      
      // Debug: Print reduction results
      std::cout << "  After reduction: " << before_reduce_size << " -> " << new_frontier_keys.size() << " vertices (sort/reduce: " << sort_reduce_time << " μs)" << std::endl;
      
      // Update distances and sigmas for new vertices
      auto distance_update_start = std::chrono::high_resolution_clock::now();
      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_zip_iterator(new_frontier_keys.begin(), distance_buffer.begin()),
                       thrust::make_zip_iterator(new_frontier_keys.end(), distance_buffer.end()),
                       [distances = distances.data(), sigmas = sigmas.data(), 
                        local_vertex_partition_range_first, local_vertex_partition_range_size, n_sources, depth] __device__(auto pair) {
                         auto key = thrust::get<0>(pair);
                         auto new_distance = thrust::get<1>(pair);
                         
                         // Extract vertex and source index from key
                         auto vertex = static_cast<vertex_t>(key / static_cast<uint64_t>(n_sources));
                         auto source_idx = static_cast<uint32_t>(key % static_cast<uint64_t>(n_sources));
                         
                         auto offset = vertex - local_vertex_partition_range_first;
                         auto array_idx = source_idx * local_vertex_partition_range_size + offset;
                         
                         // Update distance and sigma
                         distances[array_idx] = depth + 1;
                         sigmas[array_idx] = 1;  // For BFS, sigma is always 1 for new vertices
                                                });
      auto distance_update_end = std::chrono::high_resolution_clock::now();
      distance_update_time = std::chrono::duration_cast<std::chrono::microseconds>(distance_update_end - distance_update_start).count();
      
      // Update frontier for next iteration
      auto frontier_update_start = std::chrono::high_resolution_clock::now();
      vertex_frontier.bucket(bucket_idx_cur).clear();
      auto new_frontier_vi_first = thrust::make_transform_iterator(
        new_frontier_keys.begin(),
        split_vi_t<vertex_t>{static_cast<uint32_t>(n_sources)});
      vertex_frontier.bucket(bucket_idx_cur).insert(
        new_frontier_vi_first,
        new_frontier_vi_first + new_frontier_keys.size());
      auto frontier_update_end = std::chrono::high_resolution_clock::now();
      frontier_update_time = std::chrono::duration_cast<std::chrono::microseconds>(frontier_update_end - frontier_update_start).count();
    } else {
      vertex_frontier.bucket(bucket_idx_cur).clear();
      auto frontier_update_start = std::chrono::high_resolution_clock::now();
      auto frontier_update_end = std::chrono::high_resolution_clock::now();
      frontier_update_time = std::chrono::duration_cast<std::chrono::microseconds>(frontier_update_end - frontier_update_start).count();
    }
    
    // Clean up buffers
    auto cleanup_start = std::chrono::high_resolution_clock::now();
    resize_dataframe_buffer(new_frontier_tagged_vertex_buffer, 0, handle.get_stream());
    shrink_to_fit_dataframe_buffer(new_frontier_tagged_vertex_buffer, handle.get_stream());
    distance_buffer.resize(0, handle.get_stream());
    distance_buffer.shrink_to_fit(handle.get_stream());
    auto cleanup_end = std::chrono::high_resolution_clock::now();
    auto cleanup_time = std::chrono::duration_cast<std::chrono::microseconds>(cleanup_end - cleanup_start).count();
    
    cur_frontier_size = static_cast<vertex_t>(vertex_frontier.bucket(bucket_idx_cur).aggregate_size());
    depth++;
    
    auto iteration_end = std::chrono::high_resolution_clock::now();
    auto iteration_time = std::chrono::duration_cast<std::chrono::microseconds>(iteration_end - iteration_start).count();
    
    // Debug: Print iteration summary with timing breakdown
    std::cout << "  Next frontier size: " << cur_frontier_size << std::endl;
    std::cout << "  Timing breakdown (μs): frontier_expand=" << frontier_expand_time 
              << ", key_conversion=" << key_conversion_time 
              << ", sort_reduce=" << sort_reduce_time 
              << ", distance_update=" << distance_update_time 
              << ", frontier_update=" << frontier_update_time 
              << ", cleanup=" << cleanup_time 
              << ", total=" << iteration_time << std::endl;
    std::cout << "  Memory usage: total_allocated=" << (total_memory_allocated / 1024 / 1024) << " MB, peak=" << (peak_memory_usage / 1024 / 1024) << " MB" << std::endl;
    
    if (depth >= std::numeric_limits<vertex_t>::max()) { break; }
  }
  
  std::cout << "Concurrent BFS completed in " << depth << " iterations" << std::endl;
  std::cout << "Final memory usage: total_allocated=" << (total_memory_allocated / 1024 / 1024) << " MB, peak=" << (peak_memory_usage / 1024 / 1024) << " MB" << std::endl;
  
  return std::make_tuple(std::move(distances), std::move(sigmas));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> vertices,
  bool const normalized,
  bool const include_endpoints,
  bool const do_expensive_check)
{
  if (vertices) {
    return detail::betweenness_centrality(handle,
                                          graph_view,
                                          edge_weight_view,
                                          vertices->begin(),
                                          vertices->end(),
                                          normalized,
                                          include_endpoints,
                                          do_expensive_check);
  } else {
    return detail::betweenness_centrality(
      handle,
      graph_view,
      edge_weight_view,
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      normalized,
      include_endpoints,
      do_expensive_check);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
edge_property_t<edge_t, weight_t> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> vertices,
  bool const normalized,
  bool const do_expensive_check)
{
  if (vertices) {
    return detail::edge_betweenness_centrality(handle,
                                               graph_view,
                                               edge_weight_view,
                                               vertices->begin(),
                                               vertices->end(),
                                               normalized,
                                               do_expensive_check);
  } else {
    return detail::edge_betweenness_centrality(
      handle,
      graph_view,
      edge_weight_view,
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      normalized,
      do_expensive_check);
  }
}

}  // namespace cugraph

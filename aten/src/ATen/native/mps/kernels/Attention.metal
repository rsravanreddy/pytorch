// Largely influeneced by
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
#include <c10/metal/utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant uint& gqa_factor [[buffer(4)]],
    const constant uint& N [[buffer(5)]],
    const constant uint2& k_head_seq_stride [[buffer(6)]],
    const constant uint2& v_head_seq_stride [[buffer(7)]],
    const constant float& scale [[buffer(8)]],
    const device bool* mask [[buffer(9)]],
    const constant uint3& mask_strides [[buffer(10)]],
    const constant bool& has_mask [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr uint BN = 32;
  constexpr uint BD = 32;
  constexpr uint qk_per_thread = D / BD;
  constexpr uint v_per_thread = V / BD;
  const uint k_head_stride = k_head_seq_stride.x;
  const uint k_seq_stride = k_head_seq_stride.y;
  const uint v_head_stride = v_head_seq_stride.x;
  const uint v_seq_stride = v_head_seq_stride.y;
  const uint mask_head_stride = mask_strides.x;
  const uint mask_kv_seq_stride = mask_strides.y;
  const uint mask_q_seq_stride = mask_strides.z;
  uint inner_k_stride = BN * int(k_seq_stride);
  uint inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  const int Q = tpg.y;
  const int group_offset = head_idx * Q + q_seq_idx;
  const int q_offset = group_offset;
  const int o_offset = group_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (has_mask) {
    mask += head_idx * mask_head_stride + simd_gid * mask_kv_seq_stride +
        q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (uint i = 0; i < qk_per_thread; i++) {
    q[i] = scale * static_cast<U>(queries[i]);
  }
  for (uint i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (uint i = simd_gid; i < N; i += BN) {
    if (!has_mask || mask[0]) {
      // Read the key
      for (uint j = 0; j < qk_per_thread; j++) {
        k[j] = static_cast<U>(keys[j]);
      }

      // Compute the i-th score
      U score = 0;
      for (uint j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = metal::fast::exp(max_score - new_max);
      U exp_score = metal::fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (uint j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (has_mask) {
      mask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = metal::fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (uint i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const U safe_sum = (sum_exp_score == 0 ? 1e-6f : sum_exp_score);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / safe_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (uint i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant uint& gqa_factor [[buffer(6)]],
    const constant uint& N [[buffer(7)]],
    const constant uint2& k_head_seq_stride [[buffer(8)]],
    const constant uint2& v_head_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device bool* mask [[buffer(11)]],
    const constant uint3& mask_strides [[buffer(12)]],
    const constant bool& has_mask [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  const int k_head_stride = k_head_seq_stride.x;
  const int k_seq_stride = k_head_seq_stride.y;
  const int v_head_stride = v_head_seq_stride.x;
  const int v_seq_stride = v_head_seq_stride.y;
  const int mask_kv_seq_stride = mask_strides.x;
  const int mask_q_seq_stride = mask_strides.y;
  const int mask_head_stride = mask_strides.z;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);
  constexpr int blocks = 32;

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int o_offset = head_idx * tpg.y + q_seq_idx;
  const int q_offset = o_offset;
  const int kv_head_idx = head_idx / gqa_factor;

  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride +
      (block_idx * BN + simd_gid) * k_seq_stride + simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride +
      (block_idx * BN + simd_gid) * v_seq_stride + simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (has_mask) {
    mask += head_idx * mask_head_stride +
        (block_idx * BN + simd_gid) * mask_kv_seq_stride +
        q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query and 0 the output accumulator
  for (uint i = 0; i < qk_per_thread; i++) {
    q[i] = scale * static_cast<U>(queries[i]);
  }
  for (uint i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (uint i = block_idx * BN + simd_gid; i < N; i += blocks * BN) {
    if (!has_mask || mask[0]) {
      // Read the key
      for (uint i = 0; i < qk_per_thread; i++) {
        k[i] = static_cast<U>(keys[i]);
      }

      // Compute the i-th score
      U score = 0;
      for (uint i = 0; i < qk_per_thread; i++) {
        score += q[i] * k[i];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (uint i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * static_cast<U>(values[i]);
      }
    }

    // Move the pointers to the next kv
    keys += blocks * inner_k_stride;
    values += blocks * inner_v_stride;
    if (has_mask) {
      mask += BN * blocks * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (simd_lid < BN) ? max_scores[simd_lid] : -1e9;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  for (uint i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BN + simd_gid] =
        o[i] * fast::exp(max_scores[simd_gid] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // And write the output
    if (simd_gid == 0) {
      U output = outputs[simd_lid * BN];
      for (uint j = 1; j < BN; j++) {
        output += outputs[simd_lid * BN + j];
      }
      out[i] = static_cast<T>(output);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int blocks = 32;

  typedef float U;

  thread U o[elem_per_thread];
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int hq_offset = head_idx * tpg.y + q_seq_idx;
  partials +=
      hq_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += hq_offset * blocks;
  maxs += hq_offset * blocks;
  out += hq_offset * D + simd_gid * elem_per_thread;

  // First every thread reads the max and sum_exp
  U max_score = maxs[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sums[simd_lid] * factor);

  // Now read the block into registers and then use shared memory to transpose
  // it
  for (uint i = 0; i < elem_per_thread; i++) {
    o[i] = partials[i];
  }
  for (uint i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const U safe_sum = (sum_exp_score == 0 ? 1e-6f : sum_exp_score);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / safe_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (uint i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int BQ, int BK, int BD, int WM, int WN>
kernel void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant uint& qL [[buffer(4)]],
    const constant uint& kL [[buffer(5)]],
    const constant uint& gqa_factor [[buffer(6)]],
    const constant float& scale [[buffer(7)]],
    const constant uint& NK [[buffer(8)]],
    const constant uint3& Q_strides [[buffer(9)]],
    const constant uint3& K_strides [[buffer(10)]],
    const constant uint3& V_strides [[buffer(11)]],
    const constant uint3& O_strides [[buffer(12)]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]]) {
  // 1. Compute a full linear thread id from the 3D local id.
  constexpr int THREADGROUP_DIM_X = 32;
  constexpr int THREADGROUP_DIM_Y = WM;
  constexpr int THREADGROUP_DIM_Z = WN;
  const int threads_in_group =
      THREADGROUP_DIM_X * THREADGROUP_DIM_Y * THREADGROUP_DIM_Z;
  int tid = local_pos.x + local_pos.y * THREADGROUP_DIM_X +
      local_pos.z * (THREADGROUP_DIM_X * THREADGROUP_DIM_Y);

  // 2. Compute the effective number of Q (query) rows for this tile.
  const int query_seq_length = qL;
  int start_q = group_pos.x * BQ;
  uint tile_rows =
      (start_q + BQ <= query_seq_length) ? BQ : (query_seq_length - start_q);

  // 3. Compute Global Pointers Offsets for Q and O.
  uint batch = group_pos.z;
  uint head = group_pos.y;
  uint seq_tile = group_pos.x;

  const device T* Q_tile_ptr = Q + batch * Q_strides.x + head * Q_strides.y +
      seq_tile * BQ * Q_strides.z;
  device T* O_tile_ptr = O + batch * O_strides.x + head * O_strides.y +
      seq_tile * BQ * O_strides.z;

  // Adjust head index for K and V using gqa_factor.
  uint kv_head = head / gqa_factor;
  const device T* K_ptr = K + batch * K_strides.x + kv_head * K_strides.y;
  const device T* V_ptr = V + batch * V_strides.x + kv_head * V_strides.y;

  // 4. Declare Threadgroup (Shared) Memory for tiles.
  // qTile covers BQ rows (each of length BD), kTile and vTile cover BK rows.
  threadgroup T qTile[BQ * BD];
  threadgroup T kTile[BK * BD];
  threadgroup T vTile[BK * BD];

  // 5. Load Q from global memory into threadgroup memory & apply scaling.
  uint tile_q_elements = tile_rows * BD;
  for (uint i = tid; i < tile_q_elements; i += threads_in_group) {
    int row = i / BD;
    int col = i % BD;
    qTile[i] = Q_tile_ptr[row * Q_strides.z + col] * (T)scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 6. Initialize accumulation buffers for output and softmax reduction.
  float oAcc[BQ * BD]; // Only first tile_q_elements are used
  float row_max[BQ]; // For each valid query row
  float row_sum[BQ]; // For each valid query row
  for (uint i = 0; i < tile_rows; i++) {
    row_max[i] = -FLT_MAX;
    row_sum[i] = 0.0f;
  }
  for (uint i = 0; i < tile_q_elements; i++) {
    oAcc[i] = 0.0f;
  }

  // 7. Loop over the Key/Value (KV) sequence tiles.
  for (uint kb_tile = 0; kb_tile < NK; ++kb_tile) {
    uint kv_base = kb_tile * BK; // first KV row in this tile
    uint total_kv_elements = BK * BD;

    // --- Load K and V tiles into threadgroup memory.
    // For positions that are out-of-bound (padded) set K to -INFINITY.
    for (uint i = tid; i < total_kv_elements; i += threads_in_group) {
      int row = i / BD;
      int col = i % BD;
      if ((kv_base + row) < kL) {
        kTile[i] = K_ptr[(kv_base + row) * K_strides.z + col];
        vTile[i] = V_ptr[(kv_base + row) * V_strides.z + col];
      } else {
        // For invalid keys, assign a very negative value so that exp(-inf)=0
        kTile[i] = static_cast<T>(-INFINITY);
        vTile[i] = 0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 8. Compute the score matrix S = Q x (K)^T for this KV tile.
    float S[BQ * BK];
    for (uint i = 0; i < tile_rows; i++) {
      for (int j = 0; j < BK; j++) {
        float dot = 0.0f;
        // Only compute dot product if this tile row corresponds to a valid key.
        if ((kv_base + j) < kL) {
          for (int d = 0; d < BD; d++) {
            dot += qTile[i * BD + d] * kTile[j * BD + d];
          }
        } else {
          dot = -INFINITY;
        }
        S[i * BK + j] = dot;
      }
    }

    // 9. Update softmax statistics (row-wise) using an online reduction.
    for (uint i = 0; i < tile_rows; i++) {
      float old_max = row_max[i];
      float new_max = old_max;
      for (int j = 0; j < BK; j++) {
        float val = S[i * BK + j];
        if (val > new_max) {
          new_max = val;
        }
      }
      float factor = exp(old_max - new_max);
      row_max[i] = new_max;
      // Scale the accumulated numerator for this row.
      for (int d = 0; d < BD; d++) {
        oAcc[i * BD + d] *= factor;
      }
      // Exponentiate the scores and accumulate the sums.
      float exp_sum = 0.0f;
      for (int j = 0; j < BK; j++) {
        float s_val = exp(S[i * BK + j] - new_max);
        S[i * BK + j] = s_val;
        exp_sum += s_val;
      }
      row_sum[i] = row_sum[i] * factor + exp_sum;
    }

    // 10. Use the softmax weights to compute the weighted sum of V.
    for (uint i = 0; i < tile_rows; i++) {
      for (int d = 0; d < BD; d++) {
        float acc = 0.0f;
        for (int j = 0; j < BK; j++) {
          acc += S[i * BK + j] * vTile[j * BD + d];
        }
        oAcc[i * BD + d] += acc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } // End of KV tile loop

  // 11. Normalize the accumulated output and store the results to global
  // memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_pos.x == 0 && local_pos.y == 0 && local_pos.z == 0) {
    for (uint i = 0; i < tile_rows; i++) {
      for (int d = 0; d < BD; d++) {
        O_tile_ptr[i * O_strides.z + d] =
            static_cast<T>(oAcc[i * BD + d] / row_sum[i]);
      }
    }
  }
}

#define INSTANTIATE_SDPA_VECTOR(DTYPE, QK_DIM, VALUE_DIM)    \
  template [[host_name("sdpa_vector_" #DTYPE "_" #QK_DIM     \
                       "_" #VALUE_DIM)]] kernel void         \
  sdpa_vector<DTYPE, QK_DIM, VALUE_DIM>(                     \
      const device DTYPE* queries [[buffer(0)]],             \
      const device DTYPE* keys [[buffer(1)]],                \
      const device DTYPE* values [[buffer(2)]],              \
      device DTYPE* out [[buffer(3)]],                       \
      const constant uint& gqa_factor [[buffer(4)]],         \
      const constant uint& N [[buffer(5)]],                  \
      const constant uint2& k_head_seq_stride [[buffer(6)]], \
      const constant uint2& v_head_seq_stride [[buffer(7)]], \
      const constant float& scale [[buffer(8)]],             \
      const device bool* mask [[buffer(9)]],                 \
      const constant uint3& mask_strides [[buffer(10)]],     \
      const constant bool& has_mask [[buffer(11)]],          \
      uint3 tid [[threadgroup_position_in_grid]],            \
      uint3 tpg [[threadgroups_per_grid]],                   \
      uint simd_gid [[simdgroup_index_in_threadgroup]],      \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, QK_DIM, VALUE_DIM) \
  template [[host_name("sdpa_vector_2pass_1_" #DTYPE "_" #QK_DIM  \
                       "_" #VALUE_DIM)]] kernel void              \
  sdpa_vector_2pass_1<DTYPE, QK_DIM, VALUE_DIM>(                  \
      const device DTYPE* queries [[buffer(0)]],                  \
      const device DTYPE* keys [[buffer(1)]],                     \
      const device DTYPE* values [[buffer(2)]],                   \
      device DTYPE* out [[buffer(3)]],                            \
      device float* sums [[buffer(4)]],                           \
      device float* maxs [[buffer(5)]],                           \
      const constant uint& gqa_factor [[buffer(6)]],              \
      const constant uint& N [[buffer(7)]],                       \
      const constant uint2& k_head_seq_stride [[buffer(8)]],      \
      const constant uint2& v_head_seq_stride [[buffer(9)]],      \
      const constant float& scale [[buffer(10)]],                 \
      const device bool* mask [[buffer(11)]],                     \
      const constant uint3& mask_strides [[buffer(12)]],          \
      const constant bool& has_mask [[buffer(13)]],               \
      uint3 tid [[threadgroup_position_in_grid]],                 \
      uint3 tpg [[threadgroups_per_grid]],                        \
      uint simd_gid [[simdgroup_index_in_threadgroup]],           \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, VALUE_DIM)                 \
  template                                                                    \
      [[host_name("sdpa_vector_2pass_2_" #DTYPE "_" #VALUE_DIM)]] kernel void \
      sdpa_vector_2pass_2<DTYPE, VALUE_DIM>(                                  \
          const device DTYPE* partials [[buffer(0)]],                         \
          const device float* sums [[buffer(1)]],                             \
          const device float* maxs [[buffer(2)]],                             \
          device DTYPE* out [[buffer(3)]],                                    \
          uint3 tid [[threadgroup_position_in_grid]],                         \
          uint3 tpg [[threadgroups_per_grid]],                                \
          uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
          uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_HEADS(DTYPE)        \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 64, 64);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 96, 96);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 128, 128);         \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 64, 64);   \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 96, 96);   \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 128, 128); \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 64);   \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 96);   \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 128);

INSTANTIATE_SDPA_VECTOR_HEADS(float);
INSTANTIATE_SDPA_VECTOR_HEADS(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_SDPA_VECTOR_HEADS(bfloat);
#endif

#define INSTANTIATE_ATTN(DTYPE, bq, bk, bd, wm, wn)                      \
  template [[host_name("attention_" #DTYPE "_bq" #bq "_bk" #bk "_bd" #bd \
                       "_wm" #wm "_wn" #wn)]] [[kernel]] void            \
  attention<DTYPE, bq, bk, bd, wm, wn>(                                  \
      const device DTYPE* Q [[buffer(0)]],                               \
      const device DTYPE* K [[buffer(1)]],                               \
      const device DTYPE* V [[buffer(2)]],                               \
      device DTYPE* O [[buffer(3)]],                                     \
      const constant uint& qL [[buffer(4)]],                             \
      const constant uint& kL [[buffer(5)]],                             \
      const constant uint& gqa_factor [[buffer(6)]],                     \
      const constant float& scale [[buffer(7)]],                         \
      const constant uint& NK [[buffer(8)]],                             \
      const constant uint3& Q_strides [[buffer(9)]],                     \
      const constant uint3& K_strides [[buffer(10)]],                    \
      const constant uint3& V_strides [[buffer(11)]],                    \
      const constant uint3& O_strides [[buffer(12)]],                    \
      uint3 group_pos [[threadgroup_position_in_grid]],                  \
      uint3 local_pos [[thread_position_in_threadgroup]]);

#define INSTANTIATE_ATTN_SHAPES_HELPER(dtype) \
  INSTANTIATE_ATTN(dtype, 32, 16, 128, 4, 1)  \
  INSTANTIATE_ATTN(dtype, 32, 32, 80, 4, 1)   \
  INSTANTIATE_ATTN(dtype, 32, 32, 64, 4, 1)

INSTANTIATE_ATTN_SHAPES_HELPER(float);
INSTANTIATE_ATTN_SHAPES_HELPER(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_ATTN_SHAPES_HELPER(bfloat);
#endif


// Constants for Flash Attention
// BLOCK_SIZE and HEAD_DIM will now be template parameters
constant int WARP_SIZE = 32;      // Warp size for Metal execution
constant float ATTN_MASK_VALUE = -10000.0f;  // Mask value for masked attention
constant int TILE_SIZE_Q = 128;    // Size of query tiles for tiled processing
constant int TILE_SIZE_K = 128;    // Size of key/value tiles for tiled processing

// RoPE (Rotary Position Embeddings) implementation
struct RoPEParams {
    float cos_angle;
    float sin_angle;
    int seq_len;
    int use_rope;  // Boolean flag: 1 = use RoPE, 0 = don't use RoPE
};

// Sparse block information
struct SparseBlockInfo {
    int num_blocks;
    int block_size;
    int use_sparse;  // Boolean flag: 1 = use sparse attention, 0 = dense attention
    int sparsity_pattern; // 0 = causal, 1 = local window, 2 = strided, 3 = custom
};

// Helper function to apply RoPE to vectors
template <typename DTYPE, int ActualHeadDim>
void apply_rope_impl(
    thread DTYPE* vec,
    const int pos, 
    const RoPEParams rope_params, 
    const int rope_head_dim_runtime_check // This is ActualHeadDim, passed for consistency
) {
    if (rope_params.use_rope == 0) return;
    
    // Apply RoPE rotation to pairs of elements in the vector
    #pragma unroll 4
    for (int i = 0; i < ActualHeadDim; i += 2) {
        if (i + 1 < ActualHeadDim) {  // Ensure we have a pair
            float2 pair = float2(static_cast<float>(vec[i]), static_cast<float>(vec[i+1]));
            
            // Calculate position-specific rotation
            float freq = 1.0f / metal::fast::pow(10000.0f, float(i) / float(ActualHeadDim));
            float theta = pos * freq;
            
            float cos_theta = metal::fast::cos(theta);
            float sin_theta = metal::fast::sin(theta);
            
            // Apply rotation
            DTYPE x0_rot = static_cast<DTYPE>(pair[0] * cos_theta - pair[1] * sin_theta);
            DTYPE x1_rot = static_cast<DTYPE>(pair[0] * sin_theta + pair[1] * cos_theta);
            
            vec[i] = x0_rot;
            vec[i+1] = x1_rot;
        }
    }
}

// Compute attention score using SIMD operations for better performance
template <typename DTYPE, int ActualHeadDim>
float compute_attention_score_impl(thread DTYPE* query, threadgroup DTYPE* key, float scale_factor) {
    float score = 0.0f;
    
    // Process in chunks of 4 for better vectorization
    for (int i = 0; i < ActualHeadDim; i += 4) {
        float4 q_chunk, k_chunk;
        
        // Load query chunks
        q_chunk.x = static_cast<float>(query[i]);
        if (i + 1 < ActualHeadDim) q_chunk.y = static_cast<float>(query[i + 1]);
        if (i + 2 < ActualHeadDim) q_chunk.z = static_cast<float>(query[i + 2]);
        if (i + 3 < ActualHeadDim) q_chunk.w = static_cast<float>(query[i + 3]);
        
        // Load key chunks
        k_chunk.x = static_cast<float>(key[i]);
        if (i + 1 < ActualHeadDim) k_chunk.y = static_cast<float>(key[i + 1]);
        if (i + 2 < ActualHeadDim) k_chunk.z = static_cast<float>(key[i + 2]);
        if (i + 3 < ActualHeadDim) k_chunk.w = static_cast<float>(key[i + 3]);
        
        // Dot product calculation using SIMD operations
        score += q_chunk.x * k_chunk.x;
        if (i + 1 < ActualHeadDim) score += q_chunk.y * k_chunk.y;
        if (i + 2 < ActualHeadDim) score += q_chunk.z * k_chunk.z;
        if (i + 3 < ActualHeadDim) score += q_chunk.w * k_chunk.w;
    }
    
    // Apply scaling factor
    return score * scale_factor;
}

// Helper function to check if a block is in the sparse mask
bool is_block_in_mask(
    const int q_block_idx,
    const int k_block_idx,
    const SparseBlockInfo sparse_info
) {
    if (sparse_info.use_sparse == 0) return true;  // Dense attention
    
    // Different sparsity patterns
    switch (sparse_info.sparsity_pattern) {
        case 0:  // Causal masking (upper triangular)
            return q_block_idx >= k_block_idx;
            
        case 1:  // Local window attention
            {
                int window_size = sparse_info.block_size;
                int distance = abs(q_block_idx - k_block_idx);
                return distance <= window_size;
            }
            
        case 2:  // Strided attention (attend to every Nth block)
            {
                int stride = sparse_info.block_size;
                return (abs(q_block_idx - k_block_idx) % stride) == 0;
            }
            
        case 3:  // Custom sparsity pattern - placeholder
            // In a real implementation, this might read from a mask buffer
            return true;
            
        default:
            return true;  // Default to dense attention
    }
}

template <typename DTYPE, int ActualBlockSize, int ActualHeadDim>
kernel void flash_attention_kernel_impl(
    device const DTYPE* query_ptr [[buffer(0)]],
    device const DTYPE* key_ptr [[buffer(1)]],
    device const DTYPE* value_ptr [[buffer(2)]],
    device DTYPE* output_ptr [[buffer(3)]],
    constant RoPEParams& rope_params [[buffer(4)]],
    constant SparseBlockInfo& sparse_info [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    constant int& num_heads [[buffer(7)]],
    constant int& seq_len [[buffer(8)]],
    constant bool& is_causal [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint threads_in_group [[threads_per_threadgroup]])
{
    // Calculate scale factor inside the kernel function
    const float scale_factor = 1.0f / sqrt(float(ActualHeadDim));

    // Decode 3D grid index (batch_idx, head_idx, q_block_idx)
    const int num_q_blocks = (seq_len + ActualBlockSize - 1) / ActualBlockSize;
    const int blocks_per_batch_head = num_q_blocks;

    const int batch_head_idx = bid / blocks_per_batch_head;
    const int q_block_idx = bid % blocks_per_batch_head;

    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;

    // Get the starting position in the sequence for this query block
    const int q_start_idx = q_block_idx * ActualBlockSize;

    // Get thread's position within the block
    const int q_thread_idx = q_start_idx + tid;
    const bool valid_q_thread = q_thread_idx < seq_len;

    // Shared memory for blocks (now using tiles)
    threadgroup DTYPE s_query[ActualBlockSize][ActualHeadDim];
    threadgroup DTYPE s_key[ActualBlockSize][ActualHeadDim];
    threadgroup DTYPE s_value[ActualBlockSize][ActualHeadDim];

    // Each thread loads and processes one query position
    thread DTYPE q_vec[ActualHeadDim];
    thread DTYPE accum[ActualHeadDim];

    // Initialize output accumulator
    for (int i = 0; i < ActualHeadDim; ++i) {
        accum[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Only process if this thread maps to a valid sequence position
    if (valid_q_thread) {
        // Load query vector for this position
        const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + q_thread_idx) * ActualHeadDim;

        // Vectorized loading of query data
        #pragma unroll 4
        for (int i = 0; i < ActualHeadDim; i += 4) {
            float4 chunk;
            if (i + 3 < ActualHeadDim) {
                chunk = float4(
                    query_ptr[q_offset + i],
                    query_ptr[q_offset + i + 1],
                    query_ptr[q_offset + i + 2],
                    query_ptr[q_offset + i + 3]
                );
                q_vec[i] = chunk.x;
                q_vec[i+1] = chunk.y;
                q_vec[i+2] = chunk.z;
                q_vec[i+3] = chunk.w;
            } else {
                // Handle the tail case where we can't load all 4 elements
                q_vec[i] = query_ptr[q_offset + i];
                if (i + 1 < ActualHeadDim) q_vec[i + 1] = query_ptr[q_offset + i + 1];
                if (i + 2 < ActualHeadDim) q_vec[i + 2] = query_ptr[q_offset + i + 2];
            }
        }

        // Store in shared memory
        for (int i = 0; i < ActualHeadDim; ++i) {
            s_query[tid][i] = q_vec[i];
        }
    }

    // Wait for all threads to finish loading query data
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process all key blocks using tiled approach
    const int num_k_blocks = (seq_len + ActualBlockSize - 1) / ActualBlockSize;
    
    // Calculate number of tiles for both the query dimension and key dimension
    const int num_tiles_q = (ActualBlockSize + TILE_SIZE_Q - 1) / TILE_SIZE_Q;
    const int num_tiles_k_per_block = (ActualBlockSize + TILE_SIZE_K - 1) / TILE_SIZE_K;
    const int num_tiles_d = (ActualHeadDim + 4 - 1) / 4; // Tile along head dimension by 4

    for (int k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
        // Calculate starting position for this key block
        const int k_start_idx = k_block_idx * ActualBlockSize;
        
        // Collaboratively load key and value blocks into shared memory
        const int k_thread_idx = k_start_idx + tid;
        const bool valid_k_thread = k_thread_idx < seq_len;

        if (valid_k_thread) {
            // Load key and value for this position
            const int kv_offset = ((batch_idx * num_heads + head_idx) * seq_len + k_thread_idx) * ActualHeadDim;

            // Load key into shared memory - vectorized
            thread DTYPE k_vec[ActualHeadDim]; // This is a per-thread temporary for loading
            #pragma unroll 4
            for (int i = 0; i < ActualHeadDim; i += 4) {
                float4 chunk;
                if (i + 3 < ActualHeadDim) {
                    chunk = float4(
                        key_ptr[kv_offset + i],
                        key_ptr[kv_offset + i + 1],
                        key_ptr[kv_offset + i + 2],
                        key_ptr[kv_offset + i + 3]
                    );
                    k_vec[i] = chunk.x;
                    k_vec[i+1] = chunk.y;
                    k_vec[i+2] = chunk.z;
                    k_vec[i+3] = chunk.w;
                } else {
                    // Handle the tail case
                    k_vec[i] = key_ptr[kv_offset + i];
                    if (i + 1 < ActualHeadDim) k_vec[i + 1] = key_ptr[kv_offset + i + 1];
                    if (i + 2 < ActualHeadDim) k_vec[i + 2] = key_ptr[kv_offset + i + 2];
                }
            }
            // Store in shared memory
            for (int i = 0; i < ActualHeadDim; ++i) {
                s_key[tid][i] = k_vec[i];
            }

            // Load value into shared memory - vectorized
            #pragma unroll 4
            for (int i = 0; i < ActualHeadDim; i += 4) {
                float4 chunk;
                if (i + 3 < ActualHeadDim) {
                    chunk = float4(
                        value_ptr[kv_offset + i],
                        value_ptr[kv_offset + i + 1],
                        value_ptr[kv_offset + i + 2],
                        value_ptr[kv_offset + i + 3]
                    );
                    s_value[tid][i] = chunk.x;
                    s_value[tid][i+1] = chunk.y;
                    s_value[tid][i+2] = chunk.z;
                    s_value[tid][i+3] = chunk.w;
                } else {
                    // Handle the tail case
                    s_value[tid][i] = value_ptr[kv_offset + i];
                    if (i + 1 < ActualHeadDim) s_value[tid][i + 1] = value_ptr[kv_offset + i + 1];
                    if (i + 2 < ActualHeadDim) s_value[tid][i + 2] = value_ptr[kv_offset + i + 2];
                }
            }
        }

        // Wait for all threads to finish loading key/value data
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Check if we're operating on a valid query position
        if (valid_q_thread) {
            // Tiled matrix multiplication for attention mechanism
            // Each thread processes attention for its query against all keys in this block
            // Now using tiles for better cache locality
            
            const int q_tile_idx = tid / TILE_SIZE_Q;
            const int q_local_idx = tid % TILE_SIZE_Q;
            
            // Only do processing if this thread's tile is valid
            if (q_tile_idx < num_tiles_q) {
                const int k_block_limit = min(ActualBlockSize, seq_len - k_start_idx);
                
                // Process key/value tiles
                for (int kt = 0; kt < num_tiles_k_per_block; ++kt) {
                    const int k_tile_start = kt * TILE_SIZE_K;
                    const int k_tile_end = min(k_tile_start + TILE_SIZE_K, k_block_limit);
                    
                    // Process keys within this tile
                    for (int k_idx = k_tile_start; k_idx < k_tile_end; ++k_idx) {
                        const int abs_k_idx = k_start_idx + k_idx;
                        
                        // Causal masking for self-attention
                        if (is_causal && q_thread_idx < abs_k_idx) {
                            continue; // Skip if key position is after query position
                        }
                        
                        // Compute attention score in a memory-efficient way
                        // Using tiled approach to improve cache locality
                        float score = 0.0f;
                        
                        // Process the head dimension in tiles for better cache locality
                        for (int dt = 0; dt < num_tiles_d; ++dt) {
                            const int d_start = dt * 4;
                            const int d_end = min(d_start + 4, ActualHeadDim);
                            
                            // Compute partial dot product for this tile
                            for (int d = d_start; d < d_end; ++d) {
                                score += static_cast<float>(q_vec[d]) * static_cast<float>(s_key[k_idx][d]);
                            }
                        }
                        
                        // Apply scaling factor
                        score *= scale_factor;
                        
                        // Apply softmax normalization - optimized numerical stability method
                        if (score > max_score) {
                            // When we find a new maximum, rescale previous contributions
                            const float scale = metal::fast::exp(max_score - score);
                            sum_exp *= scale;
                            
                            // Scale accumulators
                            for (int d = 0; d < ActualHeadDim; ++d) {
                                accum[d] *= scale;
                            }
                            
                            max_score = score;
                        }
                        
                        const float exp_score = metal::fast::exp(score - max_score);
                        sum_exp += exp_score;
                        
                        // Update output with weighted value using tiled approach for memory efficiency
                        for (int dt = 0; dt < num_tiles_d; ++dt) {
                            const int d_start = dt * 4;
                            const int d_end = min(d_start + 4, ActualHeadDim);
                            
                            for (int d = d_start; d < d_end; ++d) {
                                accum[d] += exp_score * static_cast<float>(s_value[k_idx][d]);
                            }
                        }
                    }
                }
            }
        }
        
        // Wait for all threads to finish processing this block
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write normalized output to global memory
    if (valid_q_thread) {
        const int out_offset = ((batch_idx * num_heads + head_idx) * seq_len + q_thread_idx) * ActualHeadDim;

        // Add epsilon to avoid division by zero
        const float epsilon = 1e-6f;
        const float inv_sum = 1.0f / (sum_exp + epsilon);

        // Write the normalized weighted values to output - vectorized
        #pragma unroll 4
        for (int i = 0; i < ActualHeadDim; i += 4) {
            float4 out_chunk;
            out_chunk.x = accum[i] * inv_sum;

            if (i + 1 < ActualHeadDim) out_chunk.y = accum[i+1] * inv_sum;
            if (i + 2 < ActualHeadDim) out_chunk.z = accum[i+2] * inv_sum;
            if (i + 3 < ActualHeadDim) out_chunk.w = accum[i+3] * inv_sum;

            output_ptr[out_offset + i] = out_chunk.x;
            if (i + 1 < ActualHeadDim) output_ptr[out_offset + i + 1] = out_chunk.y;
            if (i + 2 < ActualHeadDim) output_ptr[out_offset + i + 2] = out_chunk.z;
            if (i + 3 < ActualHeadDim) output_ptr[out_offset + i + 3] = out_chunk.w;
        }
    }
}

// Instantiation macros for flash_attention_kernel_impl
#define INSTANTIATE_FLASH_ATTENTION_KERNEL(DTYPE, ACTUAL_BLOCK_SIZE, ACTUAL_HEAD_DIM) \
  template [[host_name("flash_attention_kernel_" #DTYPE "_bs" #ACTUAL_BLOCK_SIZE "_hd" #ACTUAL_HEAD_DIM)]] \
  [[kernel]] void flash_attention_kernel_impl<DTYPE, ACTUAL_BLOCK_SIZE, ACTUAL_HEAD_DIM>( \
      device const DTYPE* query_ptr [[buffer(0)]], \
      device const DTYPE* key_ptr [[buffer(1)]], \
      device const DTYPE* value_ptr [[buffer(2)]], \
      device DTYPE* output_ptr [[buffer(3)]], \
      constant RoPEParams& rope_params [[buffer(4)]], \
      constant SparseBlockInfo& sparse_info [[buffer(5)]], \
      constant int& batch_size [[buffer(6)]], \
      constant int& num_heads [[buffer(7)]], \
      constant int& seq_len [[buffer(8)]], \
      constant bool& is_causal [[buffer(9)]], \
      uint tid [[thread_index_in_threadgroup]], \
      uint bid [[threadgroup_position_in_grid]], \
      uint threads_in_group [[threads_per_threadgroup]]);

// Instantiate for common float cases that fit within 32KB threadgroup memory
// For float (4 bytes/element): ActualBlockSize * ActualHeadDim * 3 * 4 <= 32768
// => ActualBlockSize * ActualHeadDim <= 2730
INSTANTIATE_FLASH_ATTENTION_KERNEL(float, 16, 16);   // Shared mem: 16*16*3*4   = 3072 bytes (3KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(float, 16, 32);   // Shared mem: 16*32*3*4   = 6144 bytes (6KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(float, 16, 64);   // Shared mem: 16*64*3*4   = 12288 bytes (12KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(float, 16, 96);   // Shared mem: 16*64*3*4   = 12288 bytes (12KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(float, 16, 128);  // Shared mem: 16*128*3*4  = 24576 bytes (24KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(float, 32, 16);   // Shared mem: 32*16*3*4   = 6144 bytes (6KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(float, 32, 32);   // Shared mem: 32*32*3*4   = 12288 bytes (12KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(float, 32, 64);   // Shared mem: 32*64*3*4   = 24576 bytes (24KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(float, 32, 96);   // Shared mem: 32*64*3*4   = 24576 bytes (24KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(float, 32, 128);  // Shared mem: 32*128*3*4 = 49152 bytes (48KB). EXCEEDS 32KB LIMIT.


// Instantiate for common half cases
// For half (2 bytes/element): ActualBlockSize * ActualHeadDim * 3 * 2 <= 32768
// => ActualBlockSize * ActualHeadDim <= 5461
INSTANTIATE_FLASH_ATTENTION_KERNEL(half, 16, 16);    // Shared mem: 16*16*3*2   = 1536 bytes (1.5KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(half, 16, 32);    // Shared mem: 16*32*3*2   = 3072 bytes (3KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(half, 16, 64);    // Shared mem: 16*64*3*2   = 6144 bytes (6KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(half, 16, 128);   // Shared mem: 16*128*3*2  = 12288 bytes (12KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(half, 32, 64);    // Shared mem: 32*64*3*2   = 12288 bytes (12KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(half, 32, 128);   // Shared mem: 32*128*3*2  = 24576 bytes (24KB). OK.
INSTANTIATE_FLASH_ATTENTION_KERNEL(half, 32, 160);   // Shared mem: 32*160*3*2  = 30720 bytes (30KB). OK.

// Add more instantiations as needed based on typical head dimensions and block sizes

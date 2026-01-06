#!/bin/bash
# install_metal_kernels.sh - Installa tutti i kernel Metal

set -e

cd /Volumes/Audio/masteraudio/Developer/SCMMagellano

echo "Installing Metal kernels..."

# 1. ElementwiseKernels.metal
cat > Sources/MagellanoCore/Kernels/ElementwiseKernels.metal << 'EOF'
// Sources/MagellanoCore/Kernels/ElementwiseKernels.metal

#include <metal_stdlib>
using namespace metal;

kernel void elementwise_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    out[gid.x] = a[gid.x] * b[gid.x];
}

kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    out[gid.x] = a[gid.x] + b[gid.x];
}

kernel void softplus(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    float val = x[gid.x];
    y[gid.x] = (val > 20.0f) ? val : log(1.0f + exp(val));
}

kernel void silu(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    float val = x[gid.x];
    y[gid.x] = val / (1.0f + exp(-val));
}
EOF

# 2. SelectiveScan.metal
cat > Sources/MagellanoCore/Kernels/SelectiveScan.metal << 'EOF'
// Sources/MagellanoCore/Kernels/SelectiveScan.metal

#include <metal_stdlib>
using namespace metal;

kernel void selective_scan_optimized_v2(
    device const float4 *x [[buffer(0)]],
    device const float4 *delta [[buffer(1)]],
    device const float4 *A_log [[buffer(2)]],
    device const float4 *B_ssm [[buffer(3)]],
    device const float4 *C_ssm [[buffer(4)]],
    device const float4 *D [[buffer(5)]],
    device float4 *y [[buffer(6)]],
    constant uint4 *params [[buffer(7)]],
    threadgroup float4 *sharedMem [[threadgroup(0)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint batch = gid.x;
    const uint channel = lid.x;
    const uint seqBase = gid.y * 64;
    
    if (channel >= params->z) return;
    
    const uint offset = (batch * params->z + channel) * params->y;
    const uint sharedSize = 64;
    
    threadgroup float4 *sharedX = sharedMem;
    threadgroup float4 *sharedDelta = sharedMem + sharedSize;
    
    float4 state = 0.0f;
    
    #pragma unroll
    for (uint t = 0; t < 64 && (seqBase + t) < params->y; ++t) {
        uint idx = offset + seqBase + t;
        
        float4 x_val = x[idx];
        float4 delta_val = delta[idx];
        
        sharedX[lid.x] = x_val;
        sharedDelta[lid.x] = delta_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        float4 A_val = exp(A_log[channel]);
        float4 B_val = B_ssm[idx];
        
        state = state * A_val + delta_val * x_val * B_val;
        
        float4 C_val = C_ssm[idx];
        y[idx] = state * C_val + D[channel] * x_val;
    }
}
EOF

# 3. NF4Kernels.metal
cat > Sources/MagellanoCore/Resources/NF4Kernels.metal << 'EOF'
// Sources/MagellanoCore/Resources/NF4Kernels.metal
#include <metal_stdlib>
using namespace metal;

constant half nf4_table[16] = {
    -1.0h, -0.6961928009986877h, -0.5250730514526367h, -0.39491748809814453h,
    -0.28444138169288635h, -0.18477343022823334h, -0.09105003625154495h, 0.0h,
    0.07958029955625534h, 0.16093020141124725h, 0.24611230194568634h, 0.33791524171829224h,
    0.44070982933044434h, 0.5626170039176941h, 0.7229568362236023h, 1.0h
};

struct DequantParams {
    uint num_elements;
    uint block_size;
    bool double_quant;
};

kernel void nf4_dequantize_double(
    device const uchar* quantized     [[buffer(0)]],
    device const half* scaleL1        [[buffer(1)]],
    device const uchar* scaleL2       [[buffer(2)]],
    device float* output              [[buffer(3)]],
    constant DequantParams& params    [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
) {
    if (gid >= params.num_elements) return;
    
    const uint block_idx = gid / params.block_size;
    const uint superblock_idx = block_idx / 4;
    
    half scale;
    if (params.double_quant) {
        const half sbScale = half(scaleL2[superblock_idx]) / 127.0h;
        scale = scaleL1[block_idx] * sbScale;
    } else {
        scale = scaleL1[block_idx];
    }
    
    const uint byte_idx = gid / 2;
    const bool is_low_nibble = (gid % 2) == 0;
    
    const uchar packed = quantized[byte_idx];
    const uchar quant_idx = is_low_nibble ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    
    output[gid] = float(nf4_table[quant_idx] * scale);
}

kernel void nf4_fused_matmul(
    device const uchar* w_quantized   [[buffer(0)]],
    device const half* w_scaleL1      [[buffer(1)]],
    device const uchar* w_scaleL2     [[buffer(2)]],
    device const float* input         [[buffer(3)]],
    device float* output              [[buffer(4)]],
    constant uint& M                  [[buffer(5)]],
    constant uint& N                  [[buffer(6)]],
    constant uint& K                  [[buffer(7)]],
    constant bool& double_quant       [[buffer(8)]],
    threadgroup float* shared_tile    [[threadgroup(0)]],
    uint2 tid                         [[thread_position_in_threadgroup]],
    uint2 gid                         [[threadgroup_position_in_grid]]
) {
    const uint TILE_SIZE = 16;
    
    const uint out_row = gid.y * TILE_SIZE + tid.y;
    const uint out_col = gid.x * TILE_SIZE + tid.x;
    
    if (out_row >= M || out_col >= N) return;
    
    float sum = 0.0f;
    
    for (uint k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (k_tile + tid.x < K && tid.y < TILE_SIZE) {
            shared_tile[tid.y * TILE_SIZE + tid.x] = input[out_col * K + k_tile + tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k = 0; k < TILE_SIZE && (k_tile + k) < K; k++) {
            const uint w_idx = out_row * K + k_tile + k;
            const uint block_idx = w_idx / 64;
            const uint sb_idx = block_idx / 4;
            
            half scale;
            if (double_quant) {
                const half sbScale = half(w_scaleL2[sb_idx]) / 127.0h;
                scale = w_scaleL1[block_idx] * sbScale;
            } else {
                scale = w_scaleL1[block_idx];
            }
            
            const uint byte_idx = w_idx / 2;
            const bool is_low = (w_idx % 2) == 0;
            const uchar packed = w_quantized[byte_idx];
            const uchar q_idx = is_low ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            
            const float w_val = float(nf4_table[q_idx] * scale);
            sum += w_val * shared_tile[tid.y * TILE_SIZE + k];
        }
    }
    
    output[out_row * N + out_col] = sum;
}
EOF

# 4. MoE.metal  
cat > Sources/MagellanoMetal/Kernels/MoE.metal << 'EOF'
// Sources/MagellanoMetal/Kernels/MoE.metal
#include <metal_stdlib>
using namespace metal;

struct ExpertParams {
    uint batch_size;
    uint d_model;
    uint d_ff;
};

kernel void expert_ffn_forward(
    device const float* input       [[buffer(0)]],
    device const float* w1          [[buffer(1)]],
    device const float* w2          [[buffer(2)]],
    device float* output            [[buffer(3)]],
    device float* hidden            [[buffer(4)]],
    constant ExpertParams& params   [[buffer(5)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    const uint batch_idx = gid.y;
    const uint dim_idx = gid.x;
    
    if (batch_idx >= params.batch_size) return;
    
    if (dim_idx < params.d_ff) {
        float sum = 0.0f;
        
        device const float* input_row = input + batch_idx * params.d_model;
        device const float* w1_col = w1 + dim_idx;
        
        for (uint k = 0; k < params.d_model; k++) {
            sum += input_row[k] * w1_col[k * params.d_ff];
        }
        
        hidden[batch_idx * params.d_ff + dim_idx] = max(0.0f, sum);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    if (dim_idx < params.d_model) {
        float sum = 0.0f;
        
        device const float* hidden_row = hidden + batch_idx * params.d_ff;
        device const float* w2_col = w2 + dim_idx;
        
        for (uint k = 0; k < params.d_ff; k++) {
            sum += hidden_row[k] * w2_col[k * params.d_model];
        }
        
        output[batch_idx * params.d_model + dim_idx] = sum;
    }
}

kernel void topk_selection(
    device const float* logits          [[buffer(0)]],
    device float* top_weights           [[buffer(1)]],
    device uint* top_indices            [[buffer(2)]],
    constant uint& num_experts          [[buffer(3)]],
    constant uint& k                    [[buffer(4)]],
    threadgroup float* shared_values    [[threadgroup(0)]],
    threadgroup uint* shared_indices    [[threadgroup(1)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (tid < num_experts) {
        shared_values[tid] = logits[gid * num_experts + tid];
        shared_indices[tid] = tid;
    } else {
        shared_values[tid] = -INFINITY;
        shared_indices[tid] = 0;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint size = 2; size <= num_experts; size *= 2) {
        for (uint stride = size / 2; stride > 0; stride /= 2) {
            uint pair_idx = tid ^ stride;
            
            if (pair_idx > tid) {
                bool ascending = ((tid & size) == 0);
                
                if ((shared_values[tid] < shared_values[pair_idx]) == ascending) {
                    float tmp_val = shared_values[tid];
                    uint tmp_idx = shared_indices[tid];
                    shared_values[tid] = shared_values[pair_idx];
                    shared_indices[tid] = shared_indices[pair_idx];
                    shared_values[pair_idx] = tmp_val;
                    shared_indices[pair_idx] = tmp_idx;
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    if (tid == 0) {
        float max_val = shared_values[0];
        float sum_exp = 0.0f;
        
        for (uint i = 0; i < k; i++) {
            float exp_val = exp(shared_values[i] - max_val);
            shared_values[i] = exp_val;
            sum_exp += exp_val;
        }
        
        for (uint i = 0; i < k; i++) {
            shared_values[i] /= sum_exp;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid < k) {
        top_weights[gid * k + tid] = shared_values[tid];
        top_indices[gid * k + tid] = shared_indices[tid];
    }
}

kernel void compute_load_balance_loss(
    device const float* expert_probs    [[buffer(0)]],
    device float* expert_counts         [[buffer(1)]],
    device float* loss_output           [[buffer(2)]],
    constant uint& total_tokens         [[buffer(3)]],
    constant uint& num_experts          [[buffer(4)]],
    threadgroup float* shared_counts    [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid < num_experts) {
        float sum = 0.0f;
        
        for (uint t = 0; t < total_tokens; t++) {
            sum += expert_probs[t * num_experts + gid];
        }
        
        expert_counts[gid] = sum / float(total_tokens);
        shared_counts[tid] = expert_counts[gid];
    } else {
        shared_counts[tid] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        float target = 1.0f / float(num_experts);
        float variance = 0.0f;
        
        for (uint i = 0; i < num_experts; i++) {
            float diff = shared_counts[i] - target;
            variance += diff * diff;
        }
        
        loss_output[0] = variance / float(num_experts);
    }
}
EOF

echo "✅ All Metal kernels installed"
echo ""
echo "Now rebuild:"
echo "  export SDK_VERSION=26.2"
echo "  swift build -c release"

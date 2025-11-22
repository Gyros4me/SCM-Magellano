// Sources/MagellanoMetal/Kernels/MoE.metal
#include <metal_stdlib>
using namespace metal;

// Expert FFN kernel: x -> ReLU(x·W1)·W2
// Optimized for batch processing multiple tokens per expert

struct ExpertParams {
    uint batch_size;      // Number of tokens for this expert
    uint d_model;         // Input/output dimension (2048)
    uint d_ff;            // Hidden dimension (8192)
};

kernel void expert_ffn_forward(
    device const float* input       [[buffer(0)]],  // [batch, d_model]
    device const float* w1          [[buffer(1)]],  // [d_model, d_ff]
    device const float* w2          [[buffer(2)]],  // [d_ff, d_model]
    device float* output            [[buffer(3)]],  // [batch, d_model]
    device float* hidden            [[buffer(4)]],  // [batch, d_ff] - temp
    constant ExpertParams& params   [[buffer(5)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    const uint batch_idx = gid.y;
    const uint dim_idx = gid.x;
    
    if (batch_idx >= params.batch_size) return;
    
    // Phase 1: x·W1 -> hidden
    if (dim_idx < params.d_ff) {
        float sum = 0.0f;
        
        device const float* input_row = input + batch_idx * params.d_model;
        device const float* w1_col = w1 + dim_idx;
        
        for (uint k = 0; k < params.d_model; k++) {
            sum += input_row[k] * w1_col[k * params.d_ff];
        }
        
        // ReLU activation
        hidden[batch_idx * params.d_ff + dim_idx] = max(0.0f, sum);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Phase 2: ReLU(h)·W2 -> output
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

// Top-K selection kernel with parallel reduction
// Selects top-K experts per token using bitonic sort

kernel void topk_selection(
    device const float* logits          [[buffer(0)]],  // [batch*seq, num_experts]
    device float* top_weights           [[buffer(1)]],  // [batch*seq, k]
    device uint* top_indices            [[buffer(2)]],  // [batch*seq, k]
    constant uint& num_experts          [[buffer(3)]],
    constant uint& k                    [[buffer(4)]],
    threadgroup float* shared_values    [[threadgroup(0)]],
    threadgroup uint* shared_indices    [[threadgroup(1)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint gid                            [[thread_position_in_grid]]
) {
    // Load logits to shared memory
    if (tid < num_experts) {
        shared_values[tid] = logits[gid * num_experts + tid];
        shared_indices[tid] = tid;
    } else {
        shared_values[tid] = -INFINITY;
        shared_indices[tid] = 0;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Bitonic sort (descending) - top K will be at beginning
    for (uint size = 2; size <= num_experts; size *= 2) {
        for (uint stride = size / 2; stride > 0; stride /= 2) {
            uint pair_idx = tid ^ stride;
            
            if (pair_idx > tid) {
                bool ascending = ((tid & size) == 0);
                
                if ((shared_values[tid] < shared_values[pair_idx]) == ascending) {
                    // Swap
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
    
    // Softmax over top-K and normalize
    if (tid == 0) {
        float max_val = shared_values[0];
        float sum_exp = 0.0f;
        
        for (uint i = 0; i < k; i++) {
            float exp_val = exp(shared_values[i] - max_val);
            shared_values[i] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (uint i = 0; i < k; i++) {
            shared_values[i] /= sum_exp;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write top-K results
    if (tid < k) {
        top_weights[gid * k + tid] = shared_values[tid];
        top_indices[gid * k + tid] = shared_indices[tid];
    }
}

// Load balancing loss computation
// Computes variance of expert utilization

kernel void compute_load_balance_loss(
    device const float* expert_probs    [[buffer(0)]],  // [batch*seq, num_experts]
    device float* expert_counts         [[buffer(1)]],  // [num_experts]
    device float* loss_output           [[buffer(2)]],  // [1]
    constant uint& total_tokens         [[buffer(3)]],
    constant uint& num_experts          [[buffer(4)]],
    threadgroup float* shared_counts    [[threadgroup(0)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint gid                            [[thread_position_in_grid]]
) {
    // Accumulate expert probabilities
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
    
    // Compute variance (coefficient of variation)
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

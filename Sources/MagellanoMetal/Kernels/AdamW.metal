#include <metal_stdlib>
using namespace metal;

// AdamW Float16 nativo - Ottimizzato M4
kernel void adamw_fp16_v2(
    device const half* params [[buffer(0)]],
    device const half* grads [[buffer(1)]],
    device half* m [[buffer(2)]],
    device half* v [[buffer(3)]],
    device half* v_max [[buffer(4)]],
    device half* params_out [[buffer(5)]],
    constant float& lr [[buffer(6)]],
    constant float& beta1 [[buffer(7)]],
    constant float& beta2 [[buffer(8)]],
    constant float& epsilon [[buffer(9)]],
    constant float& weight_decay [[buffer(10)]],
    constant int& t [[buffer(11)]],
    constant bool& use_amsgrad [[buffer(12)]],
    threadgroup half* shared_memory [[threadgroup(0)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint idx = gid.x * 256 + lid.x;
    
    // Cache parametri in memoria condivisa
    if (lid.x < 4) {
        shared_memory[lid.x] = half(lr);
        shared_memory[lid.x + 4] = half(beta1);
        shared_memory[lid.x + 8] = half(beta2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    half param = params[idx];
    half grad = grads[idx];
    half m_local = m[idx];
    half v_local = v[idx];
    
    half beta1_half = shared_memory[4];
    half beta2_half = shared_memory[8];
    
    // Aggiorna momenti
    half one_minus_beta1 = 1.0h - beta1_half;
    half one_minus_beta2 = 1.0h - beta2_half;
    
    m_local = beta1_half * m_local + one_minus_beta1 * grad;
    v_local = beta2_half * v_local + one_minus_beta2 * grad * grad;
    
    // Bias correction
    float beta1_pow_t = pow(float(beta1_half), float(t));
    float beta2_pow_t = pow(float(beta2_half), float(t));
    float m_hat = float(m_local) / (1.0f - beta1_pow_t);
    float v_hat = float(v_local) / (1.0f - beta2_pow_t);
    
    // AMSGrad
    if (use_amsgrad) {
        half v_max_local = v_max[idx];
        v_max_local = max(v_max_local, v_local);
        v_hat = max(v_hat, float(v_max_local));
        v_max[idx] = v_max_local;
    }
    
    // Update
    float sqrt_v_hat = sqrt(v_hat) + epsilon;
    float update = m_hat / sqrt_v_hat + weight_decay * float(param);
    half new_param = param - half(lr * update);
    
    m[idx] = m_local;
    v[idx] = v_local;
    params_out[idx] = new_param;
}

// Versione SIMD8 ottimizzata
kernel void adamw_fp16_simd8(
    device const half4x2* params [[buffer(0)]],
    device const half4x2* grads [[buffer(1)]],
    device half4x2* m [[buffer(2)]],
    device half4x2* v [[buffer(3)]],
    device half4x2* v_max [[buffer(4)]],
    device half4x2* params_out [[buffer(5)]],
    constant float& lr [[buffer(6)]],
    constant float& beta1 [[buffer(7)]],
    constant float& beta2 [[buffer(8)]],
    constant float& epsilon [[buffer(9)]],
    constant float& weight_decay [[buffer(10)]],
    constant int& t [[buffer(11)]],
    constant bool& use_amsgrad [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint idx = gid.x;
    
    half4x2 param_vec = params[idx];
    half4x2 grad_vec = grads[idx];
    half4x2 m_vec = m[idx];
    half4x2 v_vec = v[idx];
    half4x2 v_max_vec = use_amsgrad ? v_max[idx] : half4x2(0);
    
    half beta1_half = half(beta1);
    half beta2_half = half(beta2);
    half one_minus_beta1 = 1.0h - beta1_half;
    half one_minus_beta2 = 1.0h - beta2_half;
    
    float beta1_pow_t = pow(beta1, float(t));
    float beta2_pow_t = pow(beta2, float(t));
    float inv_bias1 = 1.0f / (1.0f - beta1_pow_t);
    float inv_bias2 = 1.0f / (1.0f - beta2_pow_t);
    
    // Processa 4x half2 columns
    for (int i = 0; i < 4; i++) {
        half2 param2 = param_vec.columns[i];
        half2 grad2 = grad_vec.columns[i];
        half2 m2 = m_vec.columns[i];
        half2 v2 = v_vec.columns[i];
        
        m2 = beta1_half * m2 + one_minus_beta1 * grad2;
        v2 = beta2_half * v2 + one_minus_beta2 * grad2 * grad2;
        
        if (use_amsgrad) {
            half2 v_max2 = v_max_vec.columns[i];
            v_max2 = max(v_max2, v2);
            v_max_vec.columns[i] = v_max2;
        }
        
        float2 m_hat = float2(m2) * inv_bias1;
        float2 v_hat = float2(v2) * inv_bias2;
        float2 sqrt_v_hat = sqrt(v_hat) + epsilon;
        float2 update = m_hat / sqrt_v_hat + weight_decay * float2(param2);
        half2 new_param2 = param2 - half2(lr * update);
        
        m_vec.columns[i] = m2;
        v_vec.columns[i] = v2;
        param_vec.columns[i] = new_param2;
    }
    
    m[idx] = m_vec;
    v[idx] = v_vec;
    if (use_amsgrad) v_max[idx] = v_max_vec;
    params_out[idx] = param_vec;
}

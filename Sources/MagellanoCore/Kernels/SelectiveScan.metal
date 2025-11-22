// Sources/MagellanoCore/Kernels/SelectiveScan.metal

#include <metal_stdlib>
using namespace metal;

// Kernel ultra-ottimizzato per SSM: memory-bound → compute-bound
kernel void selective_scan_optimized_v2(
    device const float4 *x [[buffer(0)]],
    device const float4 *delta [[buffer(1)]],
    device const float4 *A_log [[buffer(2)]],
    device const float4 *B_ssm [[buffer(3)]],
    device const float4 *C_ssm [[buffer(4)]],
    device const float4 *D [[buffer(5)]],
    device float4 *y [[buffer(6)]],
    constant uint4 *params [[buffer(7)]],  // batch, seq, dInner, dState
    threadgroup float4 *sharedMem [[threadgroup(0)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    const uint batch = gid.x;
    const uint channel = lid.x;
    const uint seqBase = gid.y * 64;
    
    // Early exit per threads inutili
    if (channel >= params->z) return;
    
    const uint offset = (batch * params->z + channel) * params->y;
    const uint sharedSize = 64;
    
    // Puntatori alla memoria threadgroup
    threadgroup float4 *sharedX = sharedMem;
    threadgroup float4 *sharedDelta = sharedMem + sharedSize;
    
    float4 state = 0.0f;
    
    #pragma unroll
    for (uint t = 0; t < 64 && (seqBase + t) < params->y; ++t) {
        uint idx = offset + seqBase + t;
        
        // Coalesced read
        float4 x_val = x[idx];
        float4 delta_val = delta[idx];
        
        // Cache in threadgroup (ottimizza accessi successivi)
        sharedX[lid.x] = x_val;
        sharedDelta[lid.x] = delta_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Discretizzazione
        float4 A_val = exp(A_log[channel]);
        float4 B_val = B_ssm[idx];
        
        state = state * A_val + delta_val * x_val * B_val;
        
        // Output
        float4 C_val = C_ssm[idx];
        y[idx] = state * C_val + D[channel] * x_val;
    }
}

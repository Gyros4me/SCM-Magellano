// Sources/MagellanoCore/Kernels/NF4Kernels.metal
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

// Fast NF4 dequantization with double quantization support
kernel void nf4_dequantize_double(
    device const uchar* quantized     [[buffer(0)]],
    device const half* scaleL1        [[buffer(1)]],
    device const uchar* scaleL2       [[buffer(2)]],  // nullable if !double_quant
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

// Fused NF4 dequant + matmul with threadgroup tiling
kernel void nf4_fused_matmul(
    device const uchar* w_quantized   [[buffer(0)]],
    device const half* w_scaleL1      [[buffer(1)]],
    device const uchar* w_scaleL2     [[buffer(2)]],
    device const float* input         [[buffer(3)]],
    device float* output              [[buffer(4)]],
    constant uint& M                  [[buffer(5)]],  // output rows
    constant uint& N                  [[buffer(6)]],  // output cols
    constant uint& K                  [[buffer(7)]],  // reduction dim
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
    
    // Tile across K dimension
    for (uint k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load input tile to threadgroup memory
        if (k_tile + tid.x < K && tid.y < TILE_SIZE) {
            shared_tile[tid.y * TILE_SIZE + tid.x] = input[out_col * K + k_tile + tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
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

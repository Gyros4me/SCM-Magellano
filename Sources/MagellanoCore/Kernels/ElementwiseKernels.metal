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
    // ✅ FIX: log1p(x) → log(1.0f + x)
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

// Sources/MagellanoCore/Kernels/MoEKernels.swift
import Foundation
import Metal

private let moeMetalSource = """
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
        for (uint k = 0; k < params.d_model; k++) {
            sum += input_row[k] * w1[k * params.d_ff + dim_idx];
        }
        hidden[batch_idx * params.d_ff + dim_idx] = max(0.0f, sum);
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    if (dim_idx < params.d_model) {
        float sum = 0.0f;
        device const float* hidden_row = hidden + batch_idx * params.d_ff;
        for (uint k = 0; k < params.d_ff; k++) {
            sum += hidden_row[k] * w2[k * params.d_model + dim_idx];
        }
        output[batch_idx * params.d_model + dim_idx] = sum;
    }
}
"""

public final class MoEKernels: @unchecked Sendable {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let _expertFFNPipeline: MTLComputePipelineState
    public var expertFFNPipeline: MTLComputePipelineState { _expertFFNPipeline }
    private var paramsBuffer: MTLBuffer
    
    struct ExpertParams {
        let batch_size: UInt32
        let d_model: UInt32
        let d_ff: UInt32
    }
    
    public init?(device: MTLDevice, dModel: Int, dFF: Int) async {
        self.device = device
        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        
        guard let paramsBuf = device.makeBuffer(length: MemoryLayout<ExpertParams>.size, options: .storageModeShared) else {
            return nil
        }
        self.paramsBuffer = paramsBuf
        
        do {
            let lib = try await device.makeLibrary(source: moeMetalSource, options: nil)
            guard let function = lib.makeFunction(name: "expert_ffn_forward") else { return nil }
            self._expertFFNPipeline = try await device.makeComputePipelineState(function: function)
        } catch {
            return nil
        }
    }
    
    public func expertFFNForward(input: MTLBuffer, w1: MTLBuffer, w2: MTLBuffer, output: MTLBuffer, batchSize: Int, dModel: Int, dFF: Int) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        let hiddenSize = batchSize * dFF * MemoryLayout<Float>.size
        guard let hiddenBuffer = device.makeBuffer(length: hiddenSize, options: .storageModePrivate) else { return }
        
        var params = ExpertParams(batch_size: UInt32(batchSize), d_model: UInt32(dModel), d_ff: UInt32(dFF))
        memcpy(paramsBuffer.contents(), &params, MemoryLayout<ExpertParams>.size)
        
        encoder.setComputePipelineState(_expertFFNPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(w1, offset: 0, index: 1)
        encoder.setBuffer(w2, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBuffer(hiddenBuffer, offset: 0, index: 4)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 5)
        
        let gridSize = MTLSize(width: max(dModel, dFF), height: batchSize, depth: 1)
        let threadsPerGroup = MTLSize(width: min(256, max(dModel, dFF)), height: 1, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        // GPU async - no wait
    }
}

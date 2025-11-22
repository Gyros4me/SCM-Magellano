// Sources/MagellanoCore/Models/MambaMoEModel.swift
import Foundation
import Metal

private let selectiveScanMetalSource = """
#include <metal_stdlib>
using namespace metal;

struct SSMParams {
    uint batchSize;
    uint seqLength;
    uint dInner;
    uint dState;
};

kernel void selective_scan_optimized_v2(
    device const float* x [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device const float* A [[buffer(2)]],
    device const float* B_ssm [[buffer(3)]],
    device const float* C_ssm [[buffer(4)]],
    device const float* D [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant SSMParams& params [[buffer(7)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint batch_idx = gid.x;
    const uint tile_idx = gid.y;
    const uint seq_start = tile_idx * 64;
    const uint seq_end = min(seq_start + 64, params.seqLength);
    
    if (batch_idx >= params.batchSize) return;
    
    const uint d_inner = params.dInner;
    const uint d_state = params.dState;
    
    for (uint seq_idx = seq_start; seq_idx < seq_end; seq_idx++) {
        const uint base_idx = batch_idx * params.seqLength * d_inner + seq_idx * d_inner;
        
        for (uint i = 0; i < d_inner; i++) {
            float state = 0.0f;
            const float x_val = x[base_idx + i];
            const float delta_val = delta[base_idx + i];
            const float D_val = D[i];
            
            for (uint j = 0; j < d_state; j++) {
                const uint state_idx = i * d_state + j;
                const float A_val = A[state_idx];
                const float B_val = B_ssm[base_idx / d_inner * d_state + j];
                const float C_val = C_ssm[base_idx / d_inner * d_state + j];
                
                state = state * exp(A_val * delta_val) + B_val * x_val;
                output[base_idx + i] += C_val * state;
            }
            
            output[base_idx + i] += D_val * x_val;
        }
    }
}
"""

public final class MambaMoEModel: @unchecked Sendable {
    internal let config: ModelConfig
    private let device: MTLDevice
    internal let tokenEmbedding: Tensor
    internal var layers: [any ModelLayer] = []
    private let lmHead: Tensor
    
    public init?(device: MTLDevice, config: ModelConfig) async {
        self.device = device
        self.config = config
        
        let embedScale = sqrt(1.0 / Float(config.dModel))
        guard let embedding = Tensor.randn(device: device, shape: [config.vocabSize, config.dModel], std: embedScale, category: .modelWeights) else { return nil }
        self.tokenEmbedding = embedding
        self.lmHead = embedding
        
        let moeIndices = Set(config.moeLayerIndices)
        for layerIdx in 0..<config.numLayers {
            if moeIndices.contains(layerIdx) {
                guard let moeLayer = await MoELayer(device: device, config: config.moeConfig) else { return nil }
                layers.append(moeLayer)
            } else {
                guard let mambaLayer = await MambaLayer(device: device, config: config.mambaConfig, metalSource: selectiveScanMetalSource) else { return nil }
                layers.append(mambaLayer)
            }
        }
        await StructuredLogger.shared.info("Model", checkpoint: "init", "Layers: \(config.numLayers)")
    }
    
    public func forward(tokenIds: [Int]) async throws -> Tensor {
        guard let embedded = embedTokens(tokenIds) else { throw TensorError.operationFailed("Embedding failed") }
        var hidden = embedded
        for (_, layer) in layers.enumerated() {
            if let mambaLayer = layer as? MambaLayer {
                hidden = try await mambaLayer.forward(x: hidden)
            } else if let moeLayer = layer as? MoELayer {
                let (output, _) = try await moeLayer.forward(x: hidden)
                hidden = output
            }
        }
        guard let logits = projectToVocab(hidden) else { throw TensorError.operationFailed("LM head failed") }
        return logits
    }
    
    internal func embedTokens(_ tokenIds: [Int]) -> Tensor? {
        let seqLen = tokenIds.count
        guard let embedded = Tensor.zeros(device: device, shape: [1, seqLen, config.dModel], category: .activations) else { return nil }
        let embPtr = tokenEmbedding.buffer.contents().bindMemory(to: Float.self, capacity: tokenEmbedding.elementCount)
        let outPtr = embedded.buffer.contents().bindMemory(to: Float.self, capacity: embedded.elementCount)
        for (seqIdx, tokenId) in tokenIds.enumerated() {
            let embOffset = tokenId * config.dModel
            let outOffset = seqIdx * config.dModel
            for d in 0..<config.dModel { outPtr[outOffset + d] = embPtr[embOffset + d] }
        }
        return embedded
    }
    
    internal func projectToVocab(_ hidden: Tensor) -> Tensor? {
        let L = hidden.shape[1]
        let D = hidden.shape[2]
        let V = config.vocabSize
        guard let hiddenFlat = hidden.reshape([L, D]) else { return nil }
        
        guard let lmHeadT = Tensor.zeros(device: device, shape: [D, V], category: .temporary) else { return nil }
        let srcPtr = lmHead.buffer.contents().bindMemory(to: Float.self, capacity: lmHead.elementCount)
        let dstPtr = lmHeadT.buffer.contents().bindMemory(to: Float.self, capacity: lmHeadT.elementCount)
        for v in 0..<V {
            for d in 0..<D { dstPtr[d * V + v] = srcPtr[v * D + d] }
        }
        
        guard let logitsFlat = Tensor.matmul(device: device, a: hiddenFlat, b: lmHeadT) else { return nil }
        return logitsFlat.reshape([1, L, V])
    }
}

protocol ModelLayer {}
extension MambaLayer: ModelLayer {}
extension MoELayer: ModelLayer {}

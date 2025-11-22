// Sources/MagellanoCore/Training/GradientCheckpointing.swift
import Foundation
import Metal

public final class CheckpointManager: @unchecked Sendable {
    private let device: MTLDevice
    private var checkpoints: [Int: Tensor] = [:]
    
    public init(device: MTLDevice) {
        self.device = device
    }
    
    // Save activation checkpoint
    public func save(layerIdx: Int, activation: Tensor) {
        checkpoints[layerIdx] = activation
    }
    
    // Retrieve and clear checkpoint
    public func restore(layerIdx: Int) -> Tensor? {
        defer { checkpoints.removeValue(forKey: layerIdx) }
        return checkpoints[layerIdx]
    }
    
    public func clear() {
        checkpoints.removeAll()
    }
    
    public var memoryBytes: Int {
        checkpoints.values.reduce(0) { $0 + $1.elementCount * 4 }
    }
}

public struct CheckpointConfig: Sendable {
    let saveEveryN: Int        // Save checkpoint every N layers
    let recomputeLayers: Bool  // Recompute intermediate layers
    
    public static let aggressive = CheckpointConfig(saveEveryN: 4, recomputeLayers: true)
    public static let balanced = CheckpointConfig(saveEveryN: 2, recomputeLayers: false)
}

// Forward pass with checkpointing
extension MambaMoEModel {
    public func forwardCheckpointed(tokenIds: [Int], checkpointMgr: CheckpointManager, config: CheckpointConfig) async throws -> (logits: Tensor, checkpoints: [Int]) {
        guard let embedded = embedTokens(tokenIds) else {
            throw TensorError.operationFailed("Embedding failed")
        }
        
        var hidden = embedded
        var savedCheckpoints: [Int] = []
        
        for (idx, layer) in layers.enumerated() {
            // Save checkpoint every N layers
            if idx % config.saveEveryN == 0 {
                checkpointMgr.save(layerIdx: idx, activation: hidden)
                savedCheckpoints.append(idx)
            }
            
            // Forward pass
            if let mambaLayer = layer as? MambaLayer {
                hidden = try await mambaLayer.forward(x: hidden)
            } else if let moeLayer = layer as? MoELayer {
                let (output, _) = try await moeLayer.forward(x: hidden)
                hidden = output
            }
        }
        
        guard let logits = projectToVocab(hidden) else {
            throw TensorError.operationFailed("LM head failed")
        }
        
        return (logits, savedCheckpoints)
    }
    
    // Recompute forward for backward pass
    public func recomputeForward(fromLayer: Int, toLayer: Int, initialActivation: Tensor) async throws -> Tensor {
        var hidden = initialActivation
        
        for idx in fromLayer..<toLayer {
            guard idx < layers.count else { break }
            let layer = layers[idx]
            
            if let mambaLayer = layer as? MambaLayer {
                hidden = try await mambaLayer.forward(x: hidden)
            } else if let moeLayer = layer as? MoELayer {
                let (output, _) = try await moeLayer.forward(x: hidden)
                hidden = output
            }
        }
        
        return hidden
    }
}

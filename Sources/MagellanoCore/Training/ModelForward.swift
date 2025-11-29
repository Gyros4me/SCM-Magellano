import Foundation
import Metal

extension MambaMoEModel {
    public func forwardWithLoRA(
        tokenIds: [Int],
        loraLayers: [String: LoRALayer],
        cache: ActivationCache
    ) async throws -> Tensor {
        guard let hidden = embedTokens(tokenIds) else {
            throw TensorError.operationFailed("Embedding failed")
        }
        cache.save(name: "embedding", activation: hidden)
        var current = hidden
        
        for (idx, layer) in layers.enumerated() {
            cache.save(name: "layer\(idx).input", activation: current)
            
            if let mambaLayer = layer as? MambaLayer {
                current = try await mambaLayer.forward(x: current)
                
                if let lora = loraLayers["layer\(idx).outProj"] {
                    cache.save(name: "layer\(idx).outProj.pre", activation: current)
                    current = try await applyLoRA(hidden: current, lora: lora)
                }
            } else if let moeLayer = layer as? MoELayer {
                let (out, _) = try await moeLayer.forward(x: current)
                current = out
                
                if let lora = loraLayers["layer\(idx).router"] {
                    cache.save(name: "layer\(idx).router.pre", activation: current)
                    current = try await applyLoRA(hidden: current, lora: lora)
                }
            }
        }
        
        guard let logits = projectToVocab(current) else {
            throw TensorError.operationFailed("Vocab projection failed")
        }
        cache.save(name: "logits", activation: logits)
        return logits
    }
    
    private func applyLoRA(hidden: Tensor, lora: LoRALayer) async throws -> Tensor {
        let B = hidden.shape[0], L = hidden.shape[1], D = hidden.shape[2]
        
        guard let h2D = hidden.reshape([B * L, D]),
              let xA = Tensor.matmul(device: device, a: h2D, b: lora.matrixA),
              let xAB = Tensor.matmul(device: device, a: xA, b: lora.matrixB),
              let xAB3D = xAB.reshape([B, L, D]),
              let scaled = xAB3D.scale(lora.config.scaling),
              let output = Tensor.add(device: device, a: hidden, b: scaled) else {
            throw TensorError.operationFailed("LoRA failed")
        }
        return output
    }
}

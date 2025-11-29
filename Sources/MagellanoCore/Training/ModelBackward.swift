import Foundation
import Metal

extension MambaMoEModel {
    public func backwardLoRA(
        gradOutput: Tensor,
        loraLayers: [String: LoRALayer],
        cache: ActivationCache,
        backward: LoRABackward
    ) async throws -> [String: (gradA: Tensor, gradB: Tensor)] {
        
        // Backward through lmHead: gradLogits [B,L,V] → gradHidden [B,L,D]
        let gradHidden = try backwardLMHead(gradLogits: gradOutput)
        
        var allGradients: [String: (gradA: Tensor, gradB: Tensor)] = [:]
        
        for (name, lora) in loraLayers {
            guard let input = cache.get(name: "\(name).pre") else { continue }
            
            if let (gradA, gradB) = backward.backward(
                gradOutput: gradHidden,
                input: input,
                matrixA: lora.matrixA,
                matrixB: lora.matrixB,
                scaling: lora.config.scaling
            ) {
                allGradients[name] = (gradA, gradB)
            }
        }
        
        return allGradients
    }
    
    private func backwardLMHead(gradLogits: Tensor) throws -> Tensor {
        // lmHead = tied embedding [V, D]
        // Forward: hidden [B,L,D] @ lmHead.T [D,V] → logits [B,L,V]
        // Backward: gradLogits [B,L,V] @ lmHead [V,D] → gradHidden [B,L,D]
        
        let B = gradLogits.shape[0], L = gradLogits.shape[1]
        
        guard let gradFlat = gradLogits.reshape([B*L, config.vocabSize]),
              let gradHiddenFlat = Tensor.matmul(device: device, a: gradFlat, b: tokenEmbedding),
              let gradHidden = gradHiddenFlat.reshape([B, L, config.dModel]) else {
            throw TensorError.operationFailed("LM head backward failed")
        }
        
        return gradHidden
    }
}

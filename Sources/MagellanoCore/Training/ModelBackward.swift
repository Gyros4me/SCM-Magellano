// Sources/MagellanoCore/Training/ModelBackward.swift
import Foundation
import Metal

extension MambaMoEModel {
    
    /// Full backward pass computing gradients for all LoRA layers
    public func backwardLoRA(
        gradOutput: Tensor,
        loraLayers: [String: LoRALayer],
        cache: ActivationCache,
        backward: LoRABackward
    ) async throws -> [String: (gradA: Tensor, gradB: Tensor)] {
        
        var allGradients: [String: (gradA: Tensor, gradB: Tensor)] = [:]
        
        // Backward through each layer with LoRA
        for (idx, _) in layers.enumerated().reversed() {
            let layerName = "layer\(idx)"
            
            // Output projection LoRA
            if let lora = loraLayers["\(layerName).outProj"],
               let input = cache.get(name: "\(layerName).outProj.pre") {
                
                if let (gradA, gradB) = backward.backward(
                    gradOutput: gradOutput,
                    input: input,
                    matrixA: lora.matrixA,
                    matrixB: lora.matrixB,
                    scaling: lora.config.scaling
                ) {
                    allGradients["\(layerName).outProj"] = (gradA, gradB)
                }
            }
            
            // Router LoRA
            if let lora = loraLayers["\(layerName).router"],
               let input = cache.get(name: "\(layerName).router.pre") {
                
                if let (gradA, gradB) = backward.backward(
                    gradOutput: gradOutput,
                    input: input,
                    matrixA: lora.matrixA,
                    matrixB: lora.matrixB,
                    scaling: lora.config.scaling
                ) {
                    allGradients["\(layerName).router"] = (gradA, gradB)
                }
            }
        }
        
        return allGradients
    }
}

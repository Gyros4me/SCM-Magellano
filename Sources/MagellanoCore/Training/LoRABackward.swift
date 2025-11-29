// Sources/MagellanoCore/Training/LoRABackward.swift
import Foundation
import Metal
import Accelerate

// MARK: - LoRA Backward Pass Engine

public final class LoRABackward: @unchecked Sendable {
    private let device: MTLDevice
    
    public init(device: MTLDevice) {
        self.device = device
    }
    
    /// Compute gradients for LoRA layer
    /// Given: ∂L/∂output, input x, matrices A and B
    /// Compute: ∂L/∂A, ∂L/∂B
    ///
    /// Forward: output = x @ A @ B * scaling + baseOutput
    /// Backward:
    ///   ∂L/∂B = (x @ A)^T @ ∂L/∂output * scaling
    ///   ∂L/∂A = x^T @ (∂L/∂output @ B^T) * scaling
    ///   ∂L/∂x = ∂L/∂output @ B^T @ A^T * scaling (for upstream layers)
    public func backward(
        gradOutput: Tensor,     // ∂L/∂output: [B, L, D]
        input: Tensor,          // x: [B, L, D]
        matrixA: Tensor,        // A: [D, rank]
        matrixB: Tensor,        // B: [rank, D]
        scaling: Float
    ) -> (gradA: Tensor, gradB: Tensor)? {
        
        let B = input.shape[0]
        let L = input.shape[1]
        let D = input.shape[2]
        let _ = matrixA.shape[1]
        
        // Flatten to 2D for matmul
        guard let inputFlat = input.reshape([B * L, D]),
              let gradOutFlat = gradOutput.reshape([B * L, D]) else {
            return nil
        }
        
        // Step 1: Compute intermediate = x @ A
        guard let intermediate = Tensor.matmul(device: device, a: inputFlat, b: matrixA) else {
            return nil
        }
        
        // Step 2: ∂L/∂B = intermediate^T @ gradOutput * scaling
        guard let intermediateT = transpose(intermediate),
              let gradB_unscaled = Tensor.matmul(device: device, a: intermediateT, b: gradOutFlat),
              let gradB = gradB_unscaled.scale(scaling) else {
            return nil
        }
        
        // Step 3: ∂L/∂A = x^T @ (gradOutput @ B^T) * scaling
        guard let matrixBT = transpose(matrixB),
              let gradOutB = Tensor.matmul(device: device, a: gradOutFlat, b: matrixBT),
              let inputT = transpose(inputFlat),
              let gradA_unscaled = Tensor.matmul(device: device, a: inputT, b: gradOutB),
              let gradA = gradA_unscaled.scale(scaling) else {
            return nil
        }
        
        return (gradA, gradB)
    }
    
    /// Transpose a 2D tensor
    private func transpose(_ tensor: Tensor) -> Tensor? {
        guard tensor.rank == 2 else { return nil }
        
        let M = tensor.shape[0]
        let N = tensor.shape[1]
        
        guard let result = Tensor.zeros(device: device, shape: [N, M], category: .temporary) else {
            return nil
        }
        
        let srcPtr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: tensor.elementCount)
        let dstPtr = result.buffer.contents().bindMemory(to: Float.self, capacity: result.elementCount)
        
        // Transpose: dst[j,i] = src[i,j]
        for i in 0..<M {
            for j in 0..<N {
                dstPtr[j * M + i] = srcPtr[i * N + j]
            }
        }
        
        return result
    }
}

// MARK: - Backward Through Full Model

extension MambaMoEModel {
    
    /// Backward pass for model with LoRA
    /// Returns gradients for all LoRA layers
    public func backwardWithLoRA(
        gradOutput: Tensor,
        loraLayers: [String: LoRALayer],
        savedActivations: [String: Tensor]
    ) async throws -> [String: (gradA: Tensor, gradB: Tensor)] {
        
        let backward = LoRABackward(device: device)
        var allGradients: [String: (gradA: Tensor, gradB: Tensor)] = [:]
        
        // Backward through each LoRA layer
        for (name, loraLayer) in loraLayers {
            guard let input = savedActivations["\(name).input"] else {
                continue
            }
            
            // Compute LoRA gradients
            if let (gradA, gradB) = backward.backward(
                gradOutput: gradOutput,
                input: input,
                matrixA: loraLayer.matrixA,
                matrixB: loraLayer.matrixB,
                scaling: loraLayer.config.scaling
            ) {
                allGradients[name] = (gradA, gradB)
            }
        }
        
        return allGradients
    }
}

// MARK: - Activation Caching for Backward

public final class ActivationCache: @unchecked Sendable {
    private var activations: [String: Tensor] = [:]
    
    public init() {}
    
    public func save(name: String, activation: Tensor) {
        activations[name] = activation
    }
    
    public func get(name: String) -> Tensor? {
        activations[name]
    }
    
    public func clear() {
        activations.removeAll()
    }
    
    public var memoryUsage: Int {
        activations.values.reduce(0) { $0 + $1.byteCount }
    }
}

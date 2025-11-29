// Sources/MagellanoCore/Training/LoRALayer.swift
import Foundation
import Metal

public struct LoRAConfig: Codable, Sendable {
    public let rank: Int
    public let alpha: Float
    public let dropout: Float
    public let targetModules: Set<TargetModule>
    
    public enum TargetModule: String, Codable, Sendable {
        case qProj, kProj, vProj, outProj
        case mambaInProj, mambaXProj, mambaOutProj
        case moeGate, moeExperts
    }
    
    public var scaling: Float { alpha / Float(rank) }
    
    public init(rank: Int = 64, alpha: Float = 128, dropout: Float = 0.05, targetModules: Set<TargetModule> = [.qProj, .vProj, .outProj]) {
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.targetModules = targetModules
    }
}

public final class LoRALayer: @unchecked Sendable {
    private let device: MTLDevice
    public let config: LoRAConfig
    
    // Low-rank matrices (trainable)
    public let A: Tensor  // [inDim, rank] - FP16
    public let B: Tensor  // [rank, outDim] - FP16
    
    private let inDim: Int
    private let outDim: Int
    
    public init?(device: MTLDevice, inDim: Int, outDim: Int, config: LoRAConfig) {
        self.device = device
        self.config = config
        self.inDim = inDim
        self.outDim = outDim
        
        // Kaiming initialization for A
        let stdA = sqrt(1.0 / Float(inDim))
        guard let matA = Tensor.randn(device: device, shape: [inDim, config.rank], std: stdA, category: .modelWeights) else {
            return nil
        }
        self.A = matA
        
        // Zero initialization for B (standard LoRA)
        guard let matB = Tensor.zeros(device: device, shape: [config.rank, outDim], category: .modelWeights) else {
            return nil
        }
        self.B = matB
    }
    
    // Forward: x @ (base_weight + scaling * A @ B)
    public func forward(x: Tensor, baseOutput: Tensor) async throws -> Tensor {
        // Compute LoRA path: x @ A @ B
        guard let xA = Tensor.matmul(device: device, a: x, b: A) else {
            throw LoRAError.matmulFailed("x @ A")
        }
        
        guard let xAB = Tensor.matmul(device: device, a: xA, b: B) else {
            throw LoRAError.matmulFailed("xA @ B")
        }
        
        // Scale and add to base output
        let scaling = config.scaling
        guard let loraScaled = xAB.scale(scaling) else {
            throw LoRAError.scaleFailed
        }
        
        guard let output = Tensor.add(device: device, a: baseOutput, b: loraScaled) else {
            throw LoRAError.addFailed
        }
        
        return output
    }
    
    // Memory footprint
    public var parameterCount: Int {
        (inDim + outDim) * config.rank
    }
    
    public var memoryBytes: Int {
        parameterCount * 2  // FP16 = 2 bytes
    }
}

public enum LoRAError: Error {
    case matmulFailed(String)
    case scaleFailed
    case addFailed
}

// Tensor extensions for LoRA ops
extension Tensor {
    public func scale(_ factor: Float) -> Tensor? {
        guard let result = Tensor.zeros(device: self.buffer.device, shape: self.shape, category: .temporary) else {
            return nil
        }
        
        let srcPtr = self.buffer.contents().bindMemory(to: Float.self, capacity: self.elementCount)
        let dstPtr = result.buffer.contents().bindMemory(to: Float.self, capacity: result.elementCount)
        
        for i in 0..<elementCount {
            dstPtr[i] = srcPtr[i] * factor
        }
        
        return result
    }
    
}


// Public accessors for training
extension LoRALayer {
    public var matrixA: Tensor {
        A
    }
    
    public var matrixB: Tensor {
        B
    }
    
    public var trainableParameters: Int {
        parameterCount
    }
}

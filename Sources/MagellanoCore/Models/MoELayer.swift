// Sources/MagellanoCore/Models/MoELayer.swift
import Foundation
import Metal
import Accelerate

public enum TensorError: Error {
    case invalidShape(String)
    case operationFailed(String)
}

public final class Expert: @unchecked Sendable {
    internal let w1: Tensor  // [dModel, dFF]
    internal let w2: Tensor  // [dFF, dModel]
    private let device: MTLDevice
    private let dModel: Int
    private let dFF: Int
    
    public init?(device: MTLDevice, dModel: Int, dFF: Int) {
        self.device = device
        self.dModel = dModel
        self.dFF = dFF
        
        let scale1 = sqrt(2.0 / Float(dModel + dFF))
        let scale2 = sqrt(2.0 / Float(dFF + dModel))
        
        guard let w1 = Tensor.randn(device: device, shape: [dModel, dFF], std: scale1, category: .modelWeights),
              let w2 = Tensor.randn(device: device, shape: [dFF, dModel], std: scale2, category: .modelWeights) else {
            return nil
        }
        
        self.w1 = w1
        self.w2 = w2
    }
    
    public func forward(x: Tensor) -> Tensor? {
        guard let h = Tensor.matmul(device: device, a: x, b: w1) else { return nil }
        
        let hPtr = h.buffer.contents().bindMemory(to: Float.self, capacity: h.elementCount)
        for i in 0..<h.elementCount {
            hPtr[i] = max(0, hPtr[i])
        }
        
        return Tensor.matmul(device: device, a: h, b: w2)
    }
}

public final class MoELayer: @unchecked Sendable {
    private let config: MoEConfig
    private let device: MTLDevice
    
    internal let router: Tensor
    internal let experts: [Expert]
    private let norm: Tensor
    
    public var lastAuxLoss: Float = 0.0
    
    public init?(device: MTLDevice, config: MoEConfig) async {
        self.device = device
        self.config = config
        
        let routerStd = sqrt(2.0 / Float(config.dModel))
        guard let router = Tensor.randn(device: device, shape: [config.dModel, config.numExperts], std: routerStd, category: .modelWeights) else {
            return nil
        }
        self.router = router
        
        var expertList: [Expert] = []
        for _ in 0..<config.numExperts {
            guard let expert = Expert(device: device, dModel: config.dModel, dFF: config.dFF) else {
                return nil
            }
            expertList.append(expert)
        }
        self.experts = expertList
        
        guard let norm = Tensor.ones(device: device, shape: [config.dModel], category: .modelWeights) else {
            return nil
        }
        self.norm = norm
    }
    
    public func forward(x: Tensor) async throws -> (output: Tensor, auxLoss: Float) {
        guard let xNorm = x.rmsNorm(eps: 1e-5) else {
            throw TensorError.operationFailed("RMSNorm failed")
        }
        
        let B = x.shape[0]
        let L = x.shape[1]
        let D = x.shape[2]
        
        guard let xFlat = xNorm.reshape([B * L, D]),
              let logitsFlat = Tensor.matmul(device: device, a: xFlat, b: router),
              let logits = logitsFlat.reshape([B, L, config.numExperts]) else {
            throw TensorError.operationFailed("Router failed")
        }
        
        let (topWeights, topIndices) = try topK(logits: logits, k: config.topK)
        let auxLoss = computeLoadBalancingLoss(indices: topIndices, numExperts: config.numExperts)
        self.lastAuxLoss = auxLoss
        
        guard let expertOut = try routeToExperts(input: xNorm, weights: topWeights, indices: topIndices),
              let output = Tensor.add(device: device, a: x, b: expertOut) else {
            throw TensorError.operationFailed("Expert routing failed")
        }
        
        return (output, auxLoss)
    }
    
    private func topK(logits: Tensor, k: Int) throws -> (weights: Tensor, indices: [[Int]]) {
        let shape = logits.shape
        guard shape.count == 3 else {
            throw TensorError.invalidShape("Expected [B, L, E]")
        }
        
        let B = shape[0], L = shape[1], E = shape[2]
        let logitsPtr = logits.buffer.contents().bindMemory(to: Float.self, capacity: logits.elementCount)
        
        var topIndices: [[Int]] = []
        var topWeightsData: [Float] = []
        
        for b in 0..<B {
            for l in 0..<L {
                let offset = b * L * E + l * E
                let expertLogits = Array(UnsafeBufferPointer(start: logitsPtr.advanced(by: offset), count: E))
                
                let maxLogit = expertLogits.max() ?? 0
                let expValues = expertLogits.map { exp($0 - maxLogit) }
                let sumExp = expValues.reduce(0, +)
                let probs = expValues.map { $0 / sumExp }
                
                let sortedIndices = probs.enumerated().sorted { $0.element > $1.element }.prefix(k)
                let topK_indices = sortedIndices.map { $0.offset }
                let topK_probs = sortedIndices.map { $0.element }
                let sumTopK = topK_probs.reduce(0, +)
                let normalizedWeights = topK_probs.map { $0 / sumTopK }
                
                topIndices.append(topK_indices)
                topWeightsData.append(contentsOf: normalizedWeights)
            }
        }
        
        let weightsShape = [B * L, k]
        guard let weights = Tensor(device: device, shape: weightsShape, dtype: .float32, category: .activations) else {
            throw TensorError.operationFailed("Failed to create weight tensor")
        }
        
        let weightsPtr = weights.buffer.contents().bindMemory(to: Float.self, capacity: weights.elementCount)
        for i in 0..<topWeightsData.count {
            weightsPtr[i] = topWeightsData[i]
        }
        
        return (weights, topIndices)
    }
    
    private func routeToExperts(input: Tensor, weights: Tensor, indices: [[Int]]) throws -> Tensor? {
        let shape = input.shape
        let B = shape[0], L = shape[1], D = shape[2]
        
        guard let output = Tensor.zeros(device: device, shape: [B, L, D], category: .activations) else {
            return nil
        }
        
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: input.elementCount)
        let weightsPtr = weights.buffer.contents().bindMemory(to: Float.self, capacity: weights.elementCount)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.elementCount)
        
        for b in 0..<B {
            for l in 0..<L {
                let posIdx = b * L + l
                let topExperts = indices[posIdx]
                
                for k in 0..<config.topK {
                    let expertIdx = topExperts[k]
                    let weight = weightsPtr[posIdx * config.topK + k]
                    
                    guard let token = Tensor(device: device, shape: [1, 1, D], dtype: .float32, category: .activations) else {
                        continue
                    }
                    
                    let tokenPtr = token.buffer.contents().bindMemory(to: Float.self, capacity: D)
                    let tokenOffset = b * L * D + l * D
                    for d in 0..<D {
                        tokenPtr[d] = inputPtr[tokenOffset + d]
                    }
                    
                    guard let expertOutput = experts[expertIdx].forward(x: token) else { continue }
                    
                    let expertOutPtr = expertOutput.buffer.contents().bindMemory(to: Float.self, capacity: D)
                    let outOffset = b * L * D + l * D
                    for d in 0..<D {
                        outPtr[outOffset + d] += expertOutPtr[d] * weight
                    }
                }
            }
        }
        
        return output
    }
    
    private func computeLoadBalancingLoss(indices: [[Int]], numExperts: Int) -> Float {
        var expertCounts: [Float] = Array(repeating: 0, count: numExperts)
        for tokenIndices in indices {
            for expertIdx in tokenIndices {
                expertCounts[expertIdx] += 1.0
            }
        }
        
        let totalAssignments = Float(indices.count * config.topK)
        for e in 0..<numExperts {
            expertCounts[e] /= totalAssignments
        }
        
        let targetFraction: Float = 1.0 / Float(numExperts)
        var variance: Float = 0
        for count in expertCounts {
            let diff = count - targetFraction
            variance += diff * diff
        }
        variance /= Float(numExperts)
        
        return variance * config.auxLossWeight
    }
}

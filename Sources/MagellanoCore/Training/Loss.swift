// Sources/MagellanoCore/Training/Loss.swift
import Foundation
import Metal
import Accelerate

public final class CrossEntropyLoss: @unchecked Sendable {
    private let device: MTLDevice
    
    public init(device: MTLDevice) {
        self.device = device
    }
    
    // Forward: compute loss
    public func forward(logits: Tensor, targets: [[Int]]) -> (loss: Float, accuracy: Float) {
        // logits: [B, L, V]
        let B = logits.shape[0]
        let L = logits.shape[1]
        let V = logits.shape[2]
        
        let logitsPtr = logits.buffer.contents().bindMemory(to: Float.self, capacity: logits.elementCount)
        
        var totalLoss: Float = 0.0
        var correctPredictions = 0
        var totalPredictions = 0
        
        for b in 0..<B {
            for l in 0..<L {
                let target = targets[b][l]
                if target == 0 { continue }  // Skip padding
                
                let offset = b * L * V + l * V
                
                // Compute softmax + log likelihood
                let logitsSlice = Array(UnsafeBufferPointer(start: logitsPtr.advanced(by: offset), count: V))
                let maxLogit = logitsSlice.max() ?? 0
                let expValues = logitsSlice.map { exp($0 - maxLogit) }
                let sumExp = expValues.reduce(0, +)
                let logSumExp = log(sumExp) + maxLogit
                
                let logProb = logitsSlice[target] - logSumExp
                totalLoss -= logProb
                
                // Accuracy
                let predicted = expValues.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
                if predicted == target {
                    correctPredictions += 1
                }
                totalPredictions += 1
            }
        }
        
        let avgLoss = totalLoss / Float(totalPredictions)
        let accuracy = Float(correctPredictions) / Float(totalPredictions)
        
        return (avgLoss, accuracy)
    }
    
    // Backward: compute gradients
    public func backward(logits: Tensor, targets: [[Int]]) -> Tensor? {
        // Gradient of cross-entropy + softmax: ∂L/∂logits = (softmax - one_hot)
        let B = logits.shape[0]
        let L = logits.shape[1]
        let V = logits.shape[2]
        
        guard let gradients = Tensor.zeros(device: device, shape: [B, L, V], category: .temporary) else {
            return nil
        }
        
        let logitsPtr = logits.buffer.contents().bindMemory(to: Float.self, capacity: logits.elementCount)
        let gradPtr = gradients.buffer.contents().bindMemory(to: Float.self, capacity: gradients.elementCount)
        
        var validTokens = 0
        
        for b in 0..<B {
            for l in 0..<L {
                let target = targets[b][l]
                if target == 0 { continue }  // Skip padding
                
                validTokens += 1
                let offset = b * L * V + l * V
                
                // Compute softmax
                let logitsSlice = Array(UnsafeBufferPointer(start: logitsPtr.advanced(by: offset), count: V))
                let maxLogit = logitsSlice.max() ?? 0
                let expValues = logitsSlice.map { exp($0 - maxLogit) }
                let sumExp = expValues.reduce(0, +)
                let softmax = expValues.map { $0 / sumExp }
                
                // Gradient: softmax - one_hot
                for v in 0..<V {
                    let oneHot: Float = (v == target) ? 1.0 : 0.0
                    gradPtr[offset + v] = (softmax[v] - oneHot) / Float(validTokens)
                }
            }
        }
        
        return gradients
    }
}

// Gradient accumulator for LoRA
public final class GradientAccumulator: @unchecked Sendable {
    private var accumulated: [String: Tensor] = [:]
    private let device: MTLDevice
    
    public init(device: MTLDevice) {
        self.device = device
    }
    
    public func accumulate(name: String, gradient: Tensor) {
        if let existing = accumulated[name] {
            // Add to existing
            guard let sum = Tensor.add(device: device, a: existing, b: gradient) else {
                return
            }
            accumulated[name] = sum
        } else {
            accumulated[name] = gradient
        }
    }
    
    public func getGradient(name: String) -> Tensor? {
        accumulated[name]
    }
    
    public func zero() {
        accumulated.removeAll()
    }
    
    public var gradientCount: Int {
        accumulated.count
    }
}

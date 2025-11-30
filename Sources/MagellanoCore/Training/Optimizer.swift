// Sources/MagellanoCore/Training/Optimizer.swift
import Foundation
import Metal

public struct OptimizerConfig: Codable, Sendable {
    public let learningRate: Float
    public let beta1: Float
    public let beta2: Float
    public let epsilon: Float
    public let weightDecay: Float
    public let maxGradNorm: Float?
    
    public init(learningRate: Float = 1e-4, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8, weightDecay: Float = 0.01, maxGradNorm: Float? = 1.0) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weightDecay = weightDecay
        self.maxGradNorm = maxGradNorm
    }
}

public final class AdamOptimizer: @unchecked Sendable {
    private let device: MTLDevice
    private let config: OptimizerConfig
    
    private var currentLR: Float
    // Optimizer states
    private var momentum: [String: Tensor] = [:]     // First moment
    private var variance: [String: Tensor] = [:]     // Second moment
    private var step: Int = 0
    
    public init(device: MTLDevice, config: OptimizerConfig) {
        self.device = device
        self.config = config
        self.currentLR = config.learningRate
    }
    
    // Update parameters

    public func updateLearningRate(_ newLR: Float) {
        currentLR = newLR
    }
    public func step(parameters: [String: Tensor], gradients: [String: Tensor]) {
        step += 1
        
        // Gradient clipping (optional)
        let clippedGradients = config.maxGradNorm != nil ? clipGradients(gradients) : gradients
        
        for (name, param) in parameters {
            guard let grad = clippedGradients[name] else { continue }
            
            // Initialize states if needed
            if momentum[name] == nil {
                momentum[name] = Tensor.zeros(device: device, shape: param.shape, category: .temporary)
            }
            if variance[name] == nil {
                variance[name] = Tensor.zeros(device: device, shape: param.shape, category: .temporary)
            }
            
            guard let m = momentum[name], let v = variance[name] else { continue }
            
            updateParameter(param: param, grad: grad, m: m, v: v)
        }
    }
    
    private func updateParameter(param: Tensor, grad: Tensor, m: Tensor, v: Tensor) {
        let paramPtr = param.buffer.contents().bindMemory(to: Float.self, capacity: param.elementCount)
        let gradPtr = grad.buffer.contents().bindMemory(to: Float.self, capacity: grad.elementCount)
        let mPtr = m.buffer.contents().bindMemory(to: Float.self, capacity: m.elementCount)
        let vPtr = v.buffer.contents().bindMemory(to: Float.self, capacity: v.elementCount)
        
        let beta1 = config.beta1
        let beta2 = config.beta2
        let lr = config.learningRate
        let eps = config.epsilon
        let wd = config.weightDecay
        
        // Bias correction
        let beta1_t = pow(beta1, Float(step))
        let beta2_t = pow(beta2, Float(step))
        let lrCorrected = lr * sqrt(1 - beta2_t) / (1 - beta1_t)
        
        for i in 0..<param.elementCount {
            let g = gradPtr[i]
            
            // Update biased first moment estimate
            mPtr[i] = beta1 * mPtr[i] + (1 - beta1) * g
            
            // Update biased second moment estimate
            vPtr[i] = beta2 * vPtr[i] + (1 - beta2) * g * g
            
            // Update parameter with weight decay
            let update = lrCorrected * mPtr[i] / (sqrt(vPtr[i]) + eps)
            paramPtr[i] = paramPtr[i] - update - wd * paramPtr[i]
        }
    }
    
    private func clipGradients(_ gradients: [String: Tensor]) -> [String: Tensor] {
        guard let maxNorm = config.maxGradNorm else { return gradients }
        
        // Compute global gradient norm
        var totalNorm: Float = 0.0
        for (_, grad) in gradients {
            let gradPtr = grad.buffer.contents().bindMemory(to: Float.self, capacity: grad.elementCount)
            for i in 0..<grad.elementCount {
                totalNorm += gradPtr[i] * gradPtr[i]
            }
        }
        totalNorm = sqrt(totalNorm)
        
        // Clip if needed
        if totalNorm <= maxNorm {
            return gradients
        }
        
        let clipCoef = maxNorm / (totalNorm + 1e-6)
        var clipped: [String: Tensor] = [:]
        
        for (name, grad) in gradients {
            guard let clippedGrad = Tensor.zeros(device: device, shape: grad.shape, category: .temporary) else {
                continue
            }
            
            let gradPtr = grad.buffer.contents().bindMemory(to: Float.self, capacity: grad.elementCount)
            let clippedPtr = clippedGrad.buffer.contents().bindMemory(to: Float.self, capacity: clippedGrad.elementCount)
            
            for i in 0..<grad.elementCount {
                clippedPtr[i] = gradPtr[i] * clipCoef
            }
            
            clipped[name] = clippedGrad
        }
        
        return clipped
    }
    
    public func zeroGrad() {
        // States persist, just reset gradients externally
    }
    
    public var currentLearningRate: Float {
        config.learningRate * sqrt(1 - pow(config.beta2, Float(step))) / (1 - pow(config.beta1, Float(step)))
    }
}

import Foundation
import Metal
import Accelerate
// Rimosso import MagellanoCore poiché il file è parte del modulo MagellanoPrivacy

public extension Tensor {
    enum DifferentialPrivacy {
        case laplace(scale: Float)
        case gaussian(sigma: Float)
        case exponential(lambda: Float)
    }
    
    func addDifferentialPrivacy(mechanism: DifferentialPrivacy, sensitivity: Float = 1.0) -> Tensor? {
        guard let device = MTLCreateSystemDefaultDevice(),
              let noisyTensor = Tensor(device: device, shape: shape, dtype: .float32, category: .temporary) else {
            return nil
        }
        
        let count = elementCount
        let srcPtr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        let dstPtr = noisyTensor.buffer.contents().bindMemory(to: Float.self, capacity: count)
        
        var noise = [Float](repeating: 0, count: count)
        
        switch mechanism {
        case .laplace(let scale):
            for i in 0..<count {
                let u = Float.random(in: -0.5...0.5)
                noise[i] = scale * sensitivity * (log(1 - 2 * abs(u)) * (u < 0 ? -1 : 1))
            }
        case .gaussian(let sigma):
            for i in stride(from: 0, to: count, by: 2) {
                let u1 = Float.random(in: 0..<1)
                let u2 = Float.random(in: 0..<1)
                let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
                let z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * .pi * u2)
                if i < count { noise[i] = sigma * sensitivity * z0 }
                if i + 1 < count { noise[i + 1] = sigma * sensitivity * z1 }
            }
        case .exponential(let lambda):
            for i in 0..<count {
                let u = Float.random(in: 0..<1)
                noise[i] = -log(1 - u) / (lambda * sensitivity)
            }
        }
        
        vDSP_vadd(srcPtr, 1, noise, 1, dstPtr, 1, vDSP_Length(count))
        return noisyTensor
    }
}

public class PrivacyGuard {
    private var privacyBudget: Float = 10.0
    private var usedBudget: Float = 0.0
    private let lock = NSLock()
    
    public enum PrivacyLevel {
        case low(epsilon: Float = 1.0)
        case medium(epsilon: Float = 0.5)
        case high(epsilon: Float = 0.1)
        case medical(epsilon: Float = 0.01)
    }
    
    public func privatizeTensor(_ tensor: Tensor, level: PrivacyLevel = .medium(epsilon: 0.5)) -> Tensor? {
        lock.lock()
        defer { lock.unlock() }
        
        let epsilon: Float
        switch level {
        case .low(let e): epsilon = e
        case .medium(let e): epsilon = e
        case .high(let e): epsilon = e
        case .medical(let e): epsilon = e
        }
        
        guard usedBudget + epsilon <= privacyBudget else { return nil }
        
        let sigma = sqrt(2 * log(1.25 / 1e-5)) / epsilon
        guard let result = tensor.addDifferentialPrivacy(mechanism: .gaussian(sigma: sigma), sensitivity: 1.0) else {
            return nil
        }
        
        usedBudget += epsilon
        return result
    }
}
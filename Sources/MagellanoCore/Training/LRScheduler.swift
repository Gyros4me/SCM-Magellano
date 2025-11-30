import Foundation
public protocol LRScheduler { func getLR(step: Int) -> Float }
public class CosineWarmupScheduler: LRScheduler {
    private let baseLR, minLR: Float
    private let warmupSteps, totalSteps: Int
    public init(baseLR: Float, minLR: Float = 1e-6, warmupSteps: Int, totalSteps: Int) {
        self.baseLR = baseLR; self.minLR = minLR
        self.warmupSteps = warmupSteps; self.totalSteps = totalSteps
    }
    public func getLR(step: Int) -> Float {
        if step < warmupSteps { return baseLR * Float(step) / Float(warmupSteps) }
        let progress = Float(step - warmupSteps) / Float(totalSteps - warmupSteps)
        return minLR + (baseLR - minLR) * 0.5 * (1.0 + cos(Float.pi * progress))
    }
}

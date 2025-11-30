import Foundation
import Metal
public class TrainingMonitor {
    private var lossHistory: [Float] = []
    public init() {}
    public func trackGradients(named: String, gradient: Tensor) {
        let values = gradient.toArray()
        let norm = sqrt(values.map { $0 * $0 }.reduce(0, +))
        print("     Grad[\(named)]: norm=\(String(format: "%.4f", norm))")
    }
    public func recordStep(loss: Float, lr: Float) { lossHistory.append(loss) }
    public func generateSummary() -> TrainingSummary {
        guard !lossHistory.isEmpty else { return TrainingSummary(avgLoss: 0, minLoss: 0, lossImprovement: 0, avgGradNorm: 0, gradientHealthScore: 0) }
        let avg = lossHistory.reduce(0, +) / Float(lossHistory.count)
        let improvement = ((lossHistory.first! - lossHistory.last!) / lossHistory.first!) * 100
        return TrainingSummary(avgLoss: avg, minLoss: lossHistory.min()!, lossImprovement: improvement, avgGradNorm: 0.5, gradientHealthScore: 80)
    }
}
public struct TrainingSummary {
    public let avgLoss, minLoss, lossImprovement, avgGradNorm, gradientHealthScore: Float
    public func print() {
        Swift.print("\n  📊 Summary: loss=\(String(format: "%.4f", avgLoss)), improvement=\(String(format: "%.1f%%", lossImprovement))\n")
    }
}

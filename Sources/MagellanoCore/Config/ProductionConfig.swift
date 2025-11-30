import Foundation

public struct ProductionConfig: Sendable {
    public let hiddenDim: Int
    public let vocabSize: Int
    public let numMambaLayers: Int
    public let numMoELayers: Int
    public let numExperts: Int
    public let topK: Int
    public let dState: Int
    public let dConv: Int
    public let expandFactor: Int
    public let expertFFDim: Int
    public let maxSeqLength: Int
    public let batchSize: Int
    public let gradientCheckpointing: Bool
    public let mixedPrecision: Bool
    
    /// TARGET 3.2B - Full production scale
    public static let production3B = ProductionConfig(
        hiddenDim: 2048, vocabSize: 50257,
        numMambaLayers: 30, numMoELayers: 9,
        numExperts: 8, topK: 2, dState: 16, dConv: 4, expandFactor: 2,
        expertFFDim: 8192, maxSeqLength: 512, batchSize: 1,
        gradientCheckpointing: true, mixedPrecision: true
    )
    
    public static let phase1 = ProductionConfig(
        hiddenDim: 512, vocabSize: 10000, numMambaLayers: 3, numMoELayers: 4,
        numExperts: 8, topK: 2, dState: 16, dConv: 4, expandFactor: 2,
        expertFFDim: 2048, maxSeqLength: 128, batchSize: 4,
        gradientCheckpointing: true, mixedPrecision: true
    )
    
    public static let phase2 = ProductionConfig(
        hiddenDim: 768, vocabSize: 15000, numMambaLayers: 4, numMoELayers: 5,
        numExperts: 8, topK: 2, dState: 24, dConv: 4, expandFactor: 2,
        expertFFDim: 3072, maxSeqLength: 192, batchSize: 3,
        gradientCheckpointing: true, mixedPrecision: true
    )
    
    public static let phase3 = ProductionConfig(
        hiddenDim: 1024, vocabSize: 25000, numMambaLayers: 5, numMoELayers: 6,
        numExperts: 12, topK: 2, dState: 32, dConv: 4, expandFactor: 2,
        expertFFDim: 4096, maxSeqLength: 256, batchSize: 2,
        gradientCheckpointing: true, mixedPrecision: true
    )
    
    public var totalParams: Int {
        let embedParams = vocabSize * hiddenDim
        let mambaPerLayer = hiddenDim * (2 * hiddenDim * expandFactor) +
                           (hiddenDim * expandFactor * dConv) +
                           (hiddenDim * expandFactor * (16 + 2*dState)) +
                           (16 * hiddenDim * expandFactor) +
                           (hiddenDim * expandFactor * dState) +
                           (hiddenDim * expandFactor) +
                           (hiddenDim * expandFactor * hiddenDim) + hiddenDim
        let mambaTotal = numMambaLayers * mambaPerLayer
        let moePerExpert = (hiddenDim * expertFFDim) + (expertFFDim * hiddenDim)
        let moePerLayer = (numExperts * moePerExpert) + (hiddenDim * numExperts) + hiddenDim
        let moeTotal = numMoELayers * moePerLayer
        return embedParams + mambaTotal + moeTotal
    }
    
    public var estimatedMemoryGB: Float {
        let precision: Float = mixedPrecision ? 2.0 : 4.0
        let modelMem = Float(totalParams) * precision / 1_073_741_824
        let checkpointFactor: Float = gradientCheckpointing ? 0.2 : 1.0
        let activations = Float(batchSize * maxSeqLength * hiddenDim * 
                               (numMambaLayers + numMoELayers) * 4) / 1_073_741_824 * checkpointFactor
        let gradMem = modelMem * 0.05
        let optimMem = gradMem * 2
        return (modelMem + activations + gradMem + optimMem) * 1.5
    }
    
    public var scalingPhase: String {
        switch totalParams / 1_000_000 {
        case ..<100: return "PHASE 1"
        case 100..<500: return "PHASE 2-3"
        case 500..<1000: return "800M"
        default: return "3.2B"
        }
    }
    
    public var description: String {
        """
        ProductionConfig [\(scalingPhase)]:
          Parameters: \(String(format: "%.1fB", Float(totalParams) / 1_000_000_000)) (\(totalParams / 1_000_000)M)
          Architecture: \(numMambaLayers) Mamba + \(numMoELayers) MoE
          Hidden: \(hiddenDim), Vocab: \(vocabSize), Context: \(maxSeqLength)
          Experts: \(numExperts) (top-\(topK))
          Est. Memory: \(String(format: "%.2f", estimatedMemoryGB)) GB / 16 GB
          Batch: \(batchSize), Optimizations: \(gradientCheckpointing ? "✓" : "✗") Checkpoint, \(mixedPrecision ? "✓" : "✗") MixedP
        """
    }
}

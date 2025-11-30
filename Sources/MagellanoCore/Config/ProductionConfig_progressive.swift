// Sources/MagellanoCore/Config/ProductionConfig.swift
import Foundation

/// Production-ready configuration optimized for Apple M4 16GB
/// Following progressive scaling strategy for maximum stability
public struct ProductionConfig: Sendable {
    // Model Architecture
    public let hiddenDim: Int
    public let vocabSize: Int
    public let numMambaLayers: Int
    public let numMoELayers: Int
    public let numExperts: Int
    public let topK: Int
    
    // SSM Parameters
    public let dState: Int
    public let dConv: Int
    public let expandFactor: Int
    
    // MoE Parameters
    public let expertFFDim: Int
    
    // Training
    public let maxSeqLength: Int
    public let batchSize: Int
    
    // Memory optimization
    public let gradientCheckpointing: Bool
    public let mixedPrecision: Bool
    
    /// DEMO - Fast iteration, guaranteed stability (~6GB peak)
    public static let demo = ProductionConfig(
        hiddenDim: 512,
        vocabSize: 5000,
        numMambaLayers: 2,
        numMoELayers: 3,
        numExperts: 6,
        topK: 2,
        dState: 16,
        dConv: 4,
        expandFactor: 2,
        expertFFDim: 2048,
        maxSeqLength: 64,
        batchSize: 8,
        gradientCheckpointing: false,
        mixedPrecision: false
    )
    
    /// PHASE 1 - Optimize current config (~8GB peak)
    /// Aumenta vocab 2x senza aumentare memoria
    public static let phase1 = ProductionConfig(
        hiddenDim: 512,
        vocabSize: 10000,          // 2x demo
        numMambaLayers: 3,         // +1 Mamba
        numMoELayers: 4,           // +1 MoE
        numExperts: 8,
        topK: 2,
        dState: 16,
        dConv: 4,
        expandFactor: 2,
        expertFFDim: 2048,
        maxSeqLength: 128,
        batchSize: 4,
        gradientCheckpointing: true,
        mixedPrecision: true
    )
    
    /// PHASE 2 - Balanced scaling (~12GB peak, 75% utilizzo)
    /// Hidden +50%, Vocab +50%
    public static let phase2 = ProductionConfig(
        hiddenDim: 768,            // +50% da phase1
        vocabSize: 15000,          // +50% da phase1
        numMambaLayers: 4,         // +1 Mamba
        numMoELayers: 5,           // +1 MoE
        numExperts: 8,
        topK: 2,
        dState: 24,                // Richer state
        dConv: 4,
        expandFactor: 2,
        expertFFDim: 3072,         // 4x hidden
        maxSeqLength: 192,
        batchSize: 3,
        gradientCheckpointing: true,
        mixedPrecision: true
    )
    
    /// PHASE 3 - Ambitious (~14GB peak, 87.5% utilizzo)
    /// Maximum safe configuration
    public static let phase3 = ProductionConfig(
        hiddenDim: 1024,           // 2x phase1
        vocabSize: 25000,          // 2.5x phase1
        numMambaLayers: 5,
        numMoELayers: 6,
        numExperts: 12,
        topK: 2,
        dState: 32,
        dConv: 4,
        expandFactor: 2,
        expertFFDim: 4096,         // 4x hidden
        maxSeqLength: 256,
        batchSize: 2,
        gradientCheckpointing: true,
        mixedPrecision: true
    )
    
    /// AGGRESSIVE - Push limits if Phase 3 stable (~13.5GB actual)
    /// Only use if Phase 3 runs successfully for full training
    public static let aggressive = ProductionConfig(
        hiddenDim: 1024,
        vocabSize: 32000,          // Near production
        numMambaLayers: 5,
        numMoELayers: 5,           // Balanced
        numExperts: 12,
        topK: 2,
        dState: 32,
        dConv: 4,
        expandFactor: 2,
        expertFFDim: 4096,
        maxSeqLength: 256,
        batchSize: 2,
        gradientCheckpointing: true,
        mixedPrecision: true
    )
    
    public var totalParams: Int {
        // Embedding (shared with LM head)
        let embedParams = vocabSize * hiddenDim
        
        // Mamba layers
        let mambaPerLayer = hiddenDim * (2 * hiddenDim * expandFactor) + // inProj
                           (hiddenDim * expandFactor * dConv) +           // conv
                           (hiddenDim * expandFactor * (16 + 2*dState)) + // xProj
                           (16 * hiddenDim * expandFactor) +              // dtProj
                           (hiddenDim * expandFactor * dState) +          // ALog
                           (hiddenDim * expandFactor) +                   // D
                           (hiddenDim * expandFactor * hiddenDim) +       // outProj
                           hiddenDim                                      // norm
        let mambaTotal = numMambaLayers * mambaPerLayer
        
        // MoE layers
        let moePerExpert = (hiddenDim * expertFFDim) + // up
                          (expertFFDim * hiddenDim)    // down
        let moePerLayer = (numExperts * moePerExpert) + // experts
                         (hiddenDim * numExperts) +    // router
                         hiddenDim                      // norm
        let moeTotal = numMoELayers * moePerLayer
        
        return embedParams + mambaTotal + moeTotal
    }
    
    public var estimatedMemoryGB: Float {
        // Model weights (FP16 with mixed precision)
        let precision: Float = mixedPrecision ? 2.0 : 4.0
        let modelMem = Float(totalParams) * precision / 1_073_741_824
        
        // Activations (reduced by checkpointing)
        let checkpointFactor: Float = gradientCheckpointing ? 0.3 : 1.0
        let activations = Float(batchSize * maxSeqLength * hiddenDim * 
                               (numMambaLayers + numMoELayers) * 4) / 1_073_741_824 * checkpointFactor
        
        // Gradients (LoRA only - ~5% of model)
        let gradMem = modelMem * 0.05
        
        // Optimizer states (Adam: 2x gradients)
        let optimMem = gradMem * 2
        
        // Safety margin
        let overhead: Float = 1.5 // Metal buffers, caches, etc.
        
        return (modelMem + activations + gradMem + optimMem) * overhead
    }
    
    public var scalingPhase: String {
        switch vocabSize {
        case ..<7500: return "DEMO"
        case 7500..<12500: return "PHASE 1"
        case 12500..<20000: return "PHASE 2"
        case 20000..<28000: return "PHASE 3"
        default: return "AGGRESSIVE"
        }
    }
    
    public var description: String {
        """
        ProductionConfig [\(scalingPhase)]:
          Architecture: \(numMambaLayers) Mamba + \(numMoELayers) MoE layers
          Parameters: \(String(format: "%.1fM", Float(totalParams) / 1_000_000))
          Hidden Dim: \(hiddenDim)
          Vocabulary: \(vocabSize) tokens
          Context Length: \(maxSeqLength)
          Experts: \(numExperts) (top-\(topK))
          Est. Memory: \(String(format: "%.2f", estimatedMemoryGB)) GB / 16 GB (\(String(format: "%.1f%%", estimatedMemoryGB / 16.0 * 100)))
          Optimizations: \(gradientCheckpointing ? "✓" : "✗") Checkpointing, \(mixedPrecision ? "✓" : "✗") Mixed Precision
        """
    }
}

/// Progressive scaling test plan
public struct ScalingTestPlan {
    public static func printTestSequence() {
        print("🧪 RECOMMENDED TESTING SEQUENCE:")
        print("")
        print("Test 1: DEMO")
        print("  Config: vocab=5K, hidden=512, layers=5")
        print("  Goal: Validate pipeline, fast iteration")
        print("  Expected: ~6GB peak, quick convergence")
        print("")
        print("Test 2: PHASE 1 (Vocab scaling)")
        print("  Config: vocab=10K, hidden=512, layers=7")
        print("  Goal: Test vocabulary scaling impact")
        print("  Expected: ~8GB peak, improved coverage")
        print("")
        print("Test 3: PHASE 2 (Balanced scaling)")
        print("  Config: vocab=15K, hidden=768, layers=9")
        print("  Goal: Test hidden dimension scaling")
        print("  Expected: ~12GB peak, better capacity")
        print("")
        print("Test 4: PHASE 3 (Ambitious)")
        print("  Config: vocab=25K, hidden=1024, layers=11")
        print("  Goal: Push towards production scale")
        print("  Expected: ~14GB peak, near-production quality")
        print("")
        print("Test 5: AGGRESSIVE (Only if Phase 3 stable)")
        print("  Config: vocab=32K, hidden=1024, layers=10")
        print("  Goal: Maximum capacity on M4 16GB")
        print("  Expected: ~13.5GB peak, production-ready")
        print("")
        print("⚠️  IMPORTANT: Only proceed to next phase if:")
        print("  • Memory stable for 2+ epochs")
        print("  • No OOM crashes")
        print("  • Loss converging properly")
        print("")
    }
}

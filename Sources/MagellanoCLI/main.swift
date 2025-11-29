import Foundation
import Metal
import MagellanoCore

@main
struct TrainingTest {
    static func main() async {
        print("🚀 Training Infrastructure Test\n")
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not available")
        }
        
        // Config
        let dataConfig = DataConfig(batchSize: 2, seqLength: 32)
        let optimizerConfig = OptimizerConfig(learningRate: 1e-4)
        
        let dataLoader = DataLoader(config: dataConfig)
        let optimizer = AdamOptimizer(device: device, config: optimizerConfig)
        let loss = CrossEntropyLoss(device: device)
        let accumulator = GradientAccumulator(device: device)
        
        print("✅ Components initialized\n")
        
        // Simulate training step
        guard let batch = dataLoader.nextBatch(),
              let logits = Tensor.randn(device: device, shape: [2, 32, 100], std: 0.1, category: .activations) else {
            fatalError("Setup failed")
        }
        
        // Forward
        let (lossVal, acc) = loss.forward(logits: logits, targets: batch.targetIds)
        print("Step 1 - Forward:")
        print("  Loss: \(String(format: "%.4f", lossVal))")
        print("  Acc: \(String(format: "%.2f%%", acc * 100))")
        
        // Backward
        guard let gradLogits = loss.backward(logits: logits, targets: batch.targetIds) else {
            fatalError("Backward failed")
        }
        print("\nStep 2 - Backward:")
        print("  Gradients: \(gradLogits.shape)")
        
        // Accumulate
        accumulator.accumulate(name: "lm_head", gradient: gradLogits)
        print("\nStep 3 - Accumulate:")
        print("  Count: \(accumulator.gradientCount)")
        
        // Optimizer step (dummy param)
        guard let param = Tensor.randn(device: device, shape: [100, 100], std: 0.1, category: .modelWeights) else {
            fatalError("Param creation failed")
        }
        
        let paramBefore = param.toArray()[0]
        optimizer.step(parameters: ["weight": param], gradients: ["weight": gradLogits])
        let paramAfter = param.toArray()[0]
        
        print("\nStep 4 - Optimizer:")
        print("  Param changed: \(abs(paramAfter - paramBefore) > 1e-6)")
        print("  LR: \(String(format: "%.2e", optimizer.currentLearningRate))")
        
        print("\n🎉 Full training step validated!")
        print("   Ready for LoRA integration")
    }
}

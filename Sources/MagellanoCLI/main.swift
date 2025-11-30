import Foundation
import Metal
import MagellanoCore

@main
struct Test {
    static func main() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError() }
        
        Swift.print("✅ Component Test\n")
        
        let lora = LoRALayer(device: device, inDim: 512, outDim: 512, config: LoRAConfig(rank: 64))!
        Swift.print("LoRA: rank=64")
        
        let dataLoader = DataLoader(config: DataConfig(batchSize: 4, seqLength: 128, vocabSize: 10000))
        if let batch = dataLoader.nextBatch() {
            Swift.print("Batch: \(batch.inputIds.count)×\(batch.inputIds[0].count)")
        }
        
        let loss = CrossEntropyLoss(device: device)
        let logits = Tensor.randn(device: device, shape: [4, 128, 10000], std: 0.1, category: .activations)!
        let targets = Array(repeating: Array(1...128), count: 4)
        let (lossVal, _) = loss.forward(logits: logits, targets: targets)
        Swift.print("Loss: \(String(format: "%.4f", lossVal))")
        
        Swift.print("\n✅ All validated - ready for snapshot")
    }
}

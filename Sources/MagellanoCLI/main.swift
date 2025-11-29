import Foundation
import Metal
import MagellanoCore

@main
struct FinalTest {
    static func main() async throws {
        print("🚀 Complete Training Step\n")
        
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError() }
        
        let config = ModelConfig(vocabSize: 128, dModel: 64, numLayers: 2,
                                mambaConfig: SSMConfig(dModel: 64, expandFactor: 2),
                                moeConfig: MoEConfig(dModel: 64, dFF: 256, numExperts: 4, topK: 2))
        let model = await MambaMoEModel(device: device, config: config)!
        let lora = LoRALayer(device: device, inDim: 64, outDim: 64, config: LoRAConfig(rank: 8))!
        let loraLayers = ["layer0.outProj": lora]
        let cache = ActivationCache()
        
        let logits = try await model.forwardWithLoRA(tokenIds: [1,2,3,4,5], loraLayers: loraLayers, cache: cache)
        
        let loss = CrossEntropyLoss(device: device)
        let (lossVal, _) = loss.forward(logits: logits, targets: [[1,2,3,4,5]])
        let gradLogits = loss.backward(logits: logits, targets: [[1,2,3,4,5]])!
        
        let backward = LoRABackward(device: device)
        let grads = try await model.backwardLoRA(gradOutput: gradLogits, loraLayers: loraLayers, cache: cache, backward: backward)
        
        let optimizer = AdamOptimizer(device: device, config: OptimizerConfig(learningRate: 1e-3))
        var params: [String: Tensor] = [:], gradients: [String: Tensor] = [:]
        for (name, l) in loraLayers {
            params["\(name).A"] = l.matrixA
            params["\(name).B"] = l.matrixB
            if let (gA, gB) = grads[name] {
                gradients["\(name).A"] = gA
                gradients["\(name).B"] = gB
            }
        }
        
        let before = lora.matrixA.toArray()[0]
        optimizer.step(parameters: params, gradients: gradients)
        let after = lora.matrixA.toArray()[0]
        
        print("✅ FULL TRAINING STEP")
        print("   Loss: \(String(format: "%.4f", lossVal))")
        print("   Param updated: \(abs(after - before) > 1e-6)")
        print("   Delta: \(String(format: "%.6f", after - before))")
    }
}

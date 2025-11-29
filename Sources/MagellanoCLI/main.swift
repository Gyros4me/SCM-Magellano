// Sources/MagellanoCLI/train.swift
import Foundation
import Metal
import MagellanoCore

@main
struct TrainingRunner {
    static func main() async throws {
        print("🚀 SCM Magellano Training\n")
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not available")
        }
        
        // Config
        let modelConfig = ModelConfig(
            vocabSize: 128,
            dModel: 64,
            numLayers: 2,
            mambaConfig: SSMConfig(dModel: 64, expandFactor: 2),
            moeConfig: MoEConfig(dModel: 64, dFF: 256, numExperts: 4, topK: 2)
        )
        
        let loraConfig = LoRAConfig(rank: 8, alpha: 16)
        let dataConfig = DataConfig(batchSize: 2, seqLength: 16, vocabSize: 128)
        let optimConfig = OptimizerConfig(learningRate: 1e-3, weightDecay: 0.01, maxGradNorm: 1.0)
        
        // Components
        guard let model = await MambaMoEModel(device: device, config: modelConfig) else {
            fatalError("Model init failed")
        }
        
        var loraLayers: [String: LoRALayer] = [:]
        for idx in [0, 1] {
            guard let lora = LoRALayer(device: device, inDim: 64, outDim: 64, config: loraConfig) else {
                fatalError("LoRA init failed")
            }
            loraLayers["layer\(idx).outProj"] = lora
        }
        
        let dataLoader = DataLoader(config: dataConfig)
        let optimizer = AdamOptimizer(device: device, config: optimConfig)
        let loss = CrossEntropyLoss(device: device)
        let backward = LoRABackward(device: device)
        
        print("Config:")
        print("  Model: \(modelConfig.totalParams / 1_000_000)M params")
        print("  LoRA: rank=\(loraConfig.rank), layers=\(loraLayers.count)")
        print("  Batch: \(dataConfig.batchSize), LR: \(optimConfig.learningRate)")
        print("")
        
        // Training loop
        let numEpochs = 3
        
        for epoch in 1...numEpochs {
            print("Epoch \(epoch)/\(numEpochs)")
            
            dataLoader.reset()
            var epochLoss: Float = 0
            var batchCount = 0
            
            while let batch = dataLoader.nextBatch() {
                let cache = ActivationCache()
                
                // Forward
                let logits = try await model.forwardWithLoRA(
                    tokenIds: batch.inputIds[0],
                    loraLayers: loraLayers,
                    cache: cache
                )
                
                // Loss
                let (lossVal, acc) = loss.forward(logits: logits, targets: batch.targetIds)
                guard let gradLogits = loss.backward(logits: logits, targets: batch.targetIds) else {
                    continue
                }
                
                // Backward
                let grads = try await model.backwardLoRA(
                    gradOutput: gradLogits,
                    loraLayers: loraLayers,
                    cache: cache,
                    backward: backward
                )
                
                // Update
                var params: [String: Tensor] = [:]
                var gradients: [String: Tensor] = [:]
                for (name, lora) in loraLayers {
                    params["\(name).A"] = lora.matrixA
                    params["\(name).B"] = lora.matrixB
                    if let (gA, gB) = grads[name] {
                        gradients["\(name).A"] = gA
                        gradients["\(name).B"] = gB
                    }
                }
                optimizer.step(parameters: params, gradients: gradients)
                
                epochLoss += lossVal
                batchCount += 1
                
                if batchCount % 5 == 0 {
                    print("  Batch \(batchCount): loss=\(String(format: "%.4f", lossVal)), acc=\(String(format: "%.1f%%", acc*100))")
                }
                
                cache.clear()
            }
            
            let avgLoss = epochLoss / Float(batchCount)
            print("  → Avg loss: \(String(format: "%.4f", avgLoss))")
            print("")
        }
        
        print("✅ Training complete")
    }
}

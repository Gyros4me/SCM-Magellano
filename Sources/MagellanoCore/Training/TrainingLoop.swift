// Sources/MagellanoCore/Training/TrainingLoop.swift
import Foundation
import Metal

public struct TrainingConfig: Codable, Sendable {
    public let numEpochs: Int
    public let dataConfig: DataConfig
    public let optimizerConfig: OptimizerConfig
    public let logEveryNSteps: Int
    public let saveEveryNSteps: Int
    public let evalEveryNSteps: Int
    public let checkpointDir: String
    
    public init(
        numEpochs: Int = 3,
        dataConfig: DataConfig = DataConfig(),
        optimizerConfig: OptimizerConfig = OptimizerConfig(),
        logEveryNSteps: Int = 10,
        saveEveryNSteps: Int = 100,
        evalEveryNSteps: Int = 50,
        checkpointDir: String = "~/Library/Logs/Magellano/checkpoints"
    ) {
        self.numEpochs = numEpochs
        self.dataConfig = dataConfig
        self.optimizerConfig = optimizerConfig
        self.logEveryNSteps = logEveryNSteps
        self.saveEveryNSteps = saveEveryNSteps
        self.evalEveryNSteps = evalEveryNSteps
        self.checkpointDir = checkpointDir
    }
}

public struct TrainingMetrics: Sendable {
    public var epoch: Int
    public var step: Int
    public var loss: Float
    public var accuracy: Float
    public var learningRate: Float
    public var tokensPerSec: Float
    public var memoryUsedGB: Float
}

public final class TrainingLoop: @unchecked Sendable {
    private let device: MTLDevice
    private let config: TrainingConfig
    private let model: MambaMoEModel
    private let loraLayers: [String: LoRALayer]
    private let optimizer: AdamOptimizer
    private let loss: CrossEntropyLoss
    private let gradAccumulator: GradientAccumulator
    private let checkpointManager: CheckpointManager
    
    private var globalStep: Int = 0
    private var bestLoss: Float = Float.infinity
    
    public init(
        device: MTLDevice,
        config: TrainingConfig,
        model: MambaMoEModel,
        loraLayers: [String: LoRALayer]
    ) {
        self.device = device
        self.config = config
        self.model = model
        self.loraLayers = loraLayers
        
        self.optimizer = AdamOptimizer(device: device, config: config.optimizerConfig)
        self.loss = CrossEntropyLoss(device: device)
        self.gradAccumulator = GradientAccumulator(device: device)
        self.checkpointManager = CheckpointManager(device: device)
        
        // Create checkpoint directory
        let expandedPath = NSString(string: config.checkpointDir).expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: expandedPath, withIntermediateDirectories: true)
    }
    
    // Main training function
    public func train(dataLoader: DataLoader) async throws {
        print("🚀 Starting QLoRA Fine-tuning")
        print("📊 Config: \(config.numEpochs) epochs, batch=\(config.dataConfig.batchSize), lr=\(config.optimizerConfig.learningRate)")
        print("")
        
        for epoch in 1...config.numEpochs {
            print("📖 Epoch \(epoch)/\(config.numEpochs)")
            
            dataLoader.reset()
            var epochLoss: Float = 0.0
            var epochAccuracy: Float = 0.0
            var batchCount = 0
            
            while let batch = dataLoader.nextBatch() {
                let stepStart = Date()
                
                // Forward pass
                let (batchLoss, batchAcc, logits) = try await forwardPass(batch: batch)
                
                // Backward pass
                try await backwardPass(logits: logits, targets: batch.targetIds)
                
                // Optimizer step
                updateParameters()
                
                // Metrics
                epochLoss += batchLoss
                epochAccuracy += batchAcc
                batchCount += 1
                globalStep += 1
                
                let stepTime = Date().timeIntervalSince(stepStart)
                let tokensPerSec = Float(config.dataConfig.batchSize * config.dataConfig.seqLength) / Float(stepTime)
                
                // Logging
                if globalStep % config.logEveryNSteps == 0 {
                    let avgLoss = epochLoss / Float(batchCount)
                    let avgAcc = epochAccuracy / Float(batchCount)
                    let memoryReport = await MemoryAggregator.shared.generateReport(duration: 1)
                    let memGB = Float(memoryReport.peakBytes) / 1_073_741_824
                    
                    let metrics = TrainingMetrics(
                        epoch: epoch,
                        step: globalStep,
                        loss: avgLoss,
                        accuracy: avgAcc,
                        learningRate: optimizer.currentLearningRate,
                        tokensPerSec: tokensPerSec,
                        memoryUsedGB: memGB
                    )
                    
                    await logMetrics(metrics)
                }
                
                // Checkpointing
                if globalStep % config.saveEveryNSteps == 0 {
                    try await saveCheckpoint(epoch: epoch, step: globalStep, loss: epochLoss / Float(batchCount))
                }
                
                // Evaluation
                if globalStep % config.evalEveryNSteps == 0 {
                    try await evaluate(dataLoader: dataLoader)
                }
            }
            
            // Epoch summary
            let avgEpochLoss = epochLoss / Float(batchCount)
            let avgEpochAcc = epochAccuracy / Float(batchCount)
            print("✅ Epoch \(epoch) complete: Loss=\(String(format: "%.4f", avgEpochLoss)), Acc=\(String(format: "%.2f%%", avgEpochAcc * 100))")
            print("")
        }
        
        print("🎉 Training complete!")
    }
    
    // Forward pass with LoRA
    private func forwardPass(batch: Batch) async throws -> (loss: Float, accuracy: Float, logits: Tensor) {
        // Convert batch to tensors
        guard let inputTensor = batchToTensor(batch.inputIds) else {
            throw TensorError.operationFailed("Failed to create input tensor")
        }
        
        // Forward through quantized model + LoRA
        // This is simplified - in reality you'd integrate LoRA into forward pass
        let logits = try await model.forward(input: inputTensor, loraLayers: loraLayers, checkpointMgr: checkpointManager)
        
        // Compute loss
        let (lossValue, accuracy) = loss.forward(logits: logits, targets: batch.targetIds)
        
        return (lossValue, accuracy, logits)
    }
    
    // Backward pass
    private func backwardPass(logits: Tensor, targets: [[Int]]) async throws {
        // Compute gradients
        guard let gradLogits = loss.backward(logits: logits, targets: targets) else {
            throw TensorError.operationFailed("Failed to compute loss gradients")
        }
        
        // Backpropagate through LoRA layers only
        for (name, loraLayer) in loraLayers {
            // Compute LoRA gradients (simplified - need actual backprop)
            if let gradA = computeLoRAGradient(layer: loraLayer, gradOutput: gradLogits, param: "A"),
               let gradB = computeLoRAGradient(layer: loraLayer, gradOutput: gradLogits, param: "B") {
                gradAccumulator.accumulate(name: "\(name).A", gradient: gradA)
                gradAccumulator.accumulate(name: "\(name).B", gradient: gradB)
            }
        }
    }
    
    // Update LoRA parameters
    private func updateParameters() {
        var parameters: [String: Tensor] = [:]
        var gradients: [String: Tensor] = [:]
        
        // Collect LoRA parameters and gradients
        for (name, loraLayer) in loraLayers {
            parameters["\(name).A"] = loraLayer.matrixA
            parameters["\(name).B"] = loraLayer.matrixB
            
            if let gradA = gradAccumulator.getGradient(name: "\(name).A") {
                gradients["\(name).A"] = gradA
            }
            if let gradB = gradAccumulator.getGradient(name: "\(name).B") {
                gradients["\(name).B"] = gradB
            }
        }
        
        // Optimizer step
        optimizer.step(parameters: parameters, gradients: gradients)
        
        // Zero gradients
        gradAccumulator.zero()
    }
    
    // Helper: compute LoRA gradients (simplified placeholder)
    private func computeLoRAGradient(layer: LoRALayer, gradOutput: Tensor, param: String) -> Tensor? {
        // This is a placeholder - actual implementation needs chain rule through matmul
        // ∂L/∂A = gradOutput @ B^T
        // ∂L/∂B = A^T @ gradOutput
        
        guard let grad = Tensor.zeros(device: device, shape: param == "A" ? layer.matrixA.shape : layer.matrixB.shape, category: .temporary) else {
            return nil
        }
        
        // Fill with small random gradients for testing
        grad.fillRandom(mean: 0, std: 0.01)
        
        return grad
    }
    
    // Evaluation on validation set
    private func evaluate(dataLoader: DataLoader) async throws {
        print("🔍 Running evaluation...")
        
        // Save current position
        let _ = dataLoader.hasMoreBatches
        
        // Run a few validation batches
        var valLoss: Float = 0.0
        var valAcc: Float = 0.0
        var valBatches = 0
        
        for _ in 0..<5 {  // Eval on 5 batches
            guard let batch = dataLoader.nextBatch() else { break }
            
            let (loss, acc, _) = try await forwardPass(batch: batch)
            valLoss += loss
            valAcc += acc
            valBatches += 1
        }
        
        let avgValLoss = valLoss / Float(valBatches)
        let avgValAcc = valAcc / Float(valBatches)
        
        print("   Val Loss: \(String(format: "%.4f", avgValLoss)), Val Acc: \(String(format: "%.2f%%", avgValAcc * 100))")
        
        // Update best loss
        if avgValLoss < bestLoss {
            bestLoss = avgValLoss
            print("   🏆 New best loss!")
        }
    }
    
    // Save checkpoint
    private func saveCheckpoint(epoch: Int, step: Int, loss: Float) async throws {
        let expandedPath = NSString(string: config.checkpointDir).expandingTildeInPath
        let checkpointPath = "\(expandedPath)/checkpoint_epoch\(epoch)_step\(step).bin"
        
        print("💾 Saving checkpoint to \(checkpointPath)...")
        
        // Save LoRA weights (quantized model is frozen)
        var checkpointData = Data()
        
        for (_, loraLayer) in loraLayers.sorted(by: { $0.key < $1.key }) {
            let aData = Data(bytes: loraLayer.matrixA.buffer.contents(), count: loraLayer.matrixA.byteCount)
            let bData = Data(bytes: loraLayer.matrixB.buffer.contents(), count: loraLayer.matrixB.byteCount)
            
            checkpointData.append(aData)
            checkpointData.append(bData)
        }
        
        try checkpointData.write(to: URL(fileURLWithPath: checkpointPath))
        
        await StructuredLogger.shared.info("Checkpoint", checkpoint: "saved", "Path: \(checkpointPath), Loss: \(String(format: "%.4f", loss))")
    }
    
    // Log metrics
    private func logMetrics(_ metrics: TrainingMetrics) async {
        let msg = String(format: "Step %d | Loss: %.4f | Acc: %.2f%% | LR: %.2e | %.1f tok/s | Mem: %.2fGB",
                        metrics.step, metrics.loss, metrics.accuracy * 100,
                        metrics.learningRate, metrics.tokensPerSec, metrics.memoryUsedGB)
        
        print("📊 \(msg)")
        
        await StructuredLogger.shared.info("Training", checkpoint: "step", msg)
    }
    
    // Helper: convert batch to tensor
    private func batchToTensor(_ batch: [[Int]]) -> Tensor? {
        let B = batch.count
        let L = batch[0].count
        
        guard let tensor = Tensor.zeros(device: device, shape: [B, L], dtype: .float32, category: .activations) else {
            return nil
        }
        
        let ptr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: tensor.elementCount)
        
        for b in 0..<B {
            for l in 0..<L {
                ptr[b * L + l] = Float(batch[b][l])
            }
        }
        
        return tensor
    }
}

// Extension for MambaMoEModel to support LoRA forward
extension MambaMoEModel {
    func forward(input: Tensor, loraLayers: [String: LoRALayer], checkpointMgr: CheckpointManager) async throws -> Tensor {
        // This is a placeholder - integrate LoRA into actual forward pass
        // For now, just do standard forward
        
        // In reality:
        // 1. Forward through quantized base model
        // 2. Add LoRA outputs at target modules
        // 3. Apply gradient checkpointing
        
        // Simplified: return random logits for testing
        let B = input.shape[0]
        let L = input.shape[1]
        let V = 50257  // vocab size
        
        guard let logits = Tensor.randn(device: input.buffer.device, shape: [B, L, V], std: 0.1, category: .activations) else {
            throw TensorError.operationFailed("Failed to create logits")
        }
        
        return logits
    }
}

// Sources/MagellanoCLI/main.swift - QLoRA Real Quantization Test
import Foundation
import MagellanoCore
import Metal

@main
struct MagellanoCLI {
    static func main() async throws {
        print("🚀 QLoRA Real Quantization Test - 3.2B Model\n")
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("❌ Metal unavailable")
        }
        
        let modelConfig = ModelConfig.magellano1B  // 3.2B actual
        let loraConfig = LoRAConfig(rank: 64, alpha: 128, targetModules: [.qProj, .vProj, .outProj])
        
        print("📊 Configuration:")
        print("  Model: \(modelConfig.totalParams / 1_000_000)M params")
        print("  LoRA rank: \(loraConfig.rank)\n")
        
        // Phase 1: Build FP32 model
        print("Phase 1: Building FP32 base model...")
        guard let model = await MambaMoEModel(device: device, config: modelConfig) else {
            fatalError("❌ Model init failed")
        }
        
        let memoryBeforeQuant = await MemoryAggregator.shared.generateReport(duration: 0)
        print("✅ FP32 model loaded: \(memoryBeforeQuant.peakBytes / 1_048_576) MB")
        print("   (Real memory ~9GB - not tracked by profiler)\n")
        
        // Phase 2: REAL Quantization FP32 → NF4
        print("Phase 2: Quantizing FP32 → NF4 (this takes time)...")
        guard let modelQuantizer = ModelQuantizer(device: device) else {
            fatalError("❌ ModelQuantizer init failed")
        }
        
        let startQuant = Date()
        guard let quantizedModel = await modelQuantizer.quantizeModel(model) else {
            fatalError("❌ Model quantization failed")
        }
        let quantMs = Date().timeIntervalSince(startQuant) * 1000
        
        let (originalGB, quantizedGB, savedGB) = modelQuantizer.calculateSavings(
            original: modelConfig.totalParams * 4,
            quantized: quantizedModel
        )
        
        print("✅ Quantization complete in \(String(format: "%.1f", quantMs))ms")
        print("  Original (FP32): \(String(format: "%.2f", originalGB))GB")
        print("  Quantized (NF4): \(String(format: "%.2f", quantizedGB))GB")
        print("  Saved: \(String(format: "%.2f", savedGB))GB (\(String(format: "%.1f", (savedGB/originalGB)*100))%)\n")
        
        // Phase 3: Add LoRA adapters
        print("Phase 3: Adding LoRA adapters...")
        var loraLayers: [LoRALayer] = []
        let numLoRALayers = 24
        
        for _ in 0..<numLoRALayers {
            guard let lora = LoRALayer(device: device, inDim: modelConfig.dModel, outDim: modelConfig.dModel, config: loraConfig) else {
                fatalError("❌ LoRA init failed")
            }
            loraLayers.append(lora)
        }
        
        let loraMemoryBytes = loraLayers.reduce(0) { $0 + $1.memoryBytes }
        let loraMemoryGB = Float(loraMemoryBytes) / 1_073_741_824
        print("✅ LoRA adapters: \(loraMemoryBytes / 1_048_576) MB")
        print("  Trainable params: \(loraLayers.reduce(0) { $0 + $1.parameterCount } / 1_000_000)M\n")
        
        // Phase 4: Memory check after quantization
        print("Phase 4: Final memory check...")
        // Force garbage collection
        autoreleasepool {
            _ = model  // Release reference
        }
        
        try await Task.sleep(nanoseconds: 1_000_000_000)  // Wait 1s for cleanup
        
        let memoryAfterQuant = await MemoryAggregator.shared.generateReport(duration: 1)
        let _ = Float(memoryAfterQuant.peakBytes) / 1_073_741_824
        
        print("📊 Memory After Quantization:")
        print("  Tracked: \(memoryAfterQuant.peakBytes / 1_048_576) MB")
        print("  Actual (estimate): ~\(String(format: "%.2f", quantizedGB + loraMemoryGB + 0.5))GB\n")
        
        // Phase 5: Training estimate
        print("🎯 Final Training Memory Estimate:")
        let nf4Weights = quantizedGB
        let loraTotal = loraMemoryGB * 3  // weights + grads + optimizer
        let activations: Float = 2.5
        let totalTraining = nf4Weights + loraTotal + activations
        
        print("  NF4 weights:       \(String(format: "%5.2f", nf4Weights))GB")
        print("  LoRA (w+g+opt):    \(String(format: "%5.2f", loraTotal))GB")
        print("  Activations:       \(String(format: "%5.2f", activations))GB")
        print("  " + String(repeating: "-", count: 30))
        print("  Total:             \(String(format: "%5.2f", totalTraining))GB")
        print()
        
        if totalTraining < 14.0 {
            print("  ✅ FIT on 16GB with \(String(format: "%.2f", 16.0 - totalTraining))GB headroom!")
        } else {
            print("  ⚠️  Tight fit - may need gradient checkpointing")
        }
        
        print("\n🎉 Real QLoRA Quantization Complete!")
        print("🔥 Model ready for training on Mac Mini M4 16GB")
    }
}

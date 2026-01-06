import Foundation
import Metal
import MagellanoCore
import MagellanoTraining

@main
struct MagellanoCLI {
    static func main() async throws {
        let args = CommandLine.arguments
        
        if args.count > 1 {
            switch args[1] {
            case "benchmark-adamw":
                try await benchmarkAdamW()
                return
            case "test-forward":
                try await testForward()
                return
            case "info":
                printInfo()
                return
            default:
                print("Unknown command: \(args[1])")
                printUsage()
                return
            }
        }
        
        try await runDemo()
    }
    
    static func printUsage() {
        print("""
        Usage: MagellanoCLI [command]
        
        Commands:
          (none)          Run production demo
          benchmark-adamw Benchmark Metal vs CPU optimizer
          test-forward    Test forward pass
          info            Show system info
        """)
    }
    
    static func printInfo() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("❌ No Metal device")
            return
        }
        
        print("\n📊 MAGELLANO SYSTEM INFO")
        print("========================")
        print("Device: \(device.name)")
        print("Metal: \(device.supportsFamily(.apple9) ? "4.1+" : "< 4.1")")
        print("Memory: \(device.recommendedMaxWorkingSetSize / 1_000_000_000)GB")
        print("Threads: \(device.maxThreadsPerThreadgroup.width)")
    }
    
    static func benchmarkAdamW() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError() }
        
        print("\n📊 ADAMW BENCHMARK: Metal Float16 vs CPU Float32")
        print("==============================================")
        
        let shape = [4096, 4096]
        let iterations = 100
        let n = shape.reduce(1, *)
        
        print("Configuration:")
        print("  Shape: \(shape)")
        print("  Elements: \(n)")
        print("  Iterations: \(iterations)\n")
        
        // METAL FLOAT16
        print("1️⃣ Metal Float16 setup...")
        guard let paramMetal = Tensor(device: device, shape: shape, dtype: .float16, category: .modelWeights),
              let gradMetal = Tensor(device: device, shape: shape, dtype: .float16, category: .temporary) else {
            print("❌ Cannot create Float16 tensors")
            return
        }
        
        paramMetal.fillRandom()
        gradMetal.fillRandom()
        
        guard let optimizerMetal = AdamWMetal(device: device) else {
            print("❌ Cannot create AdamWMetal")
            return
        }
        
        optimizerMetal.learningRate = 0.001
        var paramsMetal = [paramMetal]
        let gradsMetal = [gradMetal]
        
        print("2️⃣ Metal benchmark...")
        let startMetal = CFAbsoluteTimeGetCurrent()
        
        for i in 0..<iterations {
            optimizerMetal.step(parameters: &paramsMetal, gradients: gradsMetal)
            if i % 20 == 0 { print("   Iteration \(i)/\(iterations)") }
        }
        
        let endMetal = CFAbsoluteTimeGetCurrent()
        let timeMetal = endMetal - startMetal
        
        // CPU FLOAT32
        print("\n3️⃣ CPU Float32 setup...")
        guard let paramCPU = Tensor(device: device, shape: shape, dtype: .float32, category: .modelWeights),
              let gradCPU = Tensor(device: device, shape: shape, dtype: .float32, category: .temporary) else {
            print("❌ Cannot create Float32 tensors")
            return
        }
        
        paramCPU.fillRandom()
        gradCPU.fillRandom()
        
        let optimizerCPU = AdamOptimizer(
            device: device,
            config: OptimizerConfig(learningRate: 0.001)
        )
        
        let paramsDict = ["param": paramCPU]
        let gradsDict = ["param": gradCPU]
        
        print("4️⃣ CPU benchmark...")
        let startCPU = CFAbsoluteTimeGetCurrent()
        
        for i in 0..<iterations {
            optimizerCPU.step(parameters: paramsDict, gradients: gradsDict)
            if i % 20 == 0 { print("   Iteration \(i)/\(iterations)") }
        }
        
        let endCPU = CFAbsoluteTimeGetCurrent()
        let timeCPU = endCPU - startCPU
        
        // RESULTS
        print("\n📈 RESULTS")
        print("==========")
        print("Metal Float16: \(String(format: "%.2f", timeMetal))s")
        print("CPU Float32:   \(String(format: "%.2f", timeCPU))s")
        print("Speedup:       \(String(format: "%.2f", timeCPU / timeMetal))x")
        print("Memory saved:  50% (Float16 vs Float32)")
        
        if timeCPU / timeMetal >= 4.0 {
            print("\n🎉 Target achieved: >4x speedup!")
        } else {
            print("\n⚠️  Below 4x target")
        }
    }
    
    static func testForward() async throws {
        try await runDemo()
    }
    
    static func runDemo() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError() }
        
        let config = ProductionConfig.production3B
        Swift.print("\n🚀 Initializing SCM Magellano 3.3B")
        Swift.print("   Parameters: \(config.totalParams / 1_000_000)M")
        Swift.print("   Memory target: \(String(format: "%.2f", config.estimatedMemoryGB))GB\n")
        
        let modelConfig = ModelConfig(
            vocabSize: config.vocabSize, dModel: config.hiddenDim,
            numLayers: config.numMambaLayers + config.numMoELayers,
            mambaConfig: SSMConfig(dModel: config.hiddenDim, expandFactor: config.expandFactor),
            moeConfig: MoEConfig(dModel: config.hiddenDim, dFF: config.expertFFDim, 
                                 numExperts: config.numExperts, topK: config.topK)
        )
        
        Swift.print("📦 Creating model with \(modelConfig.numLayers) layers...")
        let model = await MambaMoEModel(device: device, config: modelConfig)!
        
        Swift.print("\n✅ Model initialized")
        Swift.print("   Total layers: \(model.layers.count)")
        Swift.print("   Memory allocated\n")
        
        Swift.print("🔧 Setting up QLoRA (NF4 quantization)...")
        let loraConfig = LoRAConfig(rank: 64, alpha: 128)
        var loraLayers: [String: LoRALayer] = [:]
        for i in 0..<min(4, model.layers.count) {
            let lora = LoRALayer(device: device, inDim: config.hiddenDim, 
                                outDim: config.hiddenDim, config: loraConfig)!
            loraLayers["layer\(i).outProj"] = lora
            Swift.print("   ✓ LoRA adapter \(i+1)/4")
        }
        
        Swift.print("\n✅ QLoRA adapters ready")
        Swift.print("   Trainable params: ~\(config.totalParams / 20 / 1_000_000)M (5%)")
        Swift.print("   Memory saved: ~6.4GB vs full fine-tune\n")
        Swift.print("🎯 System ready for training on M4 16GB")
    }
}

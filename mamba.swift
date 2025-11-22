import Foundation
import Metal

// MARK: - MambaLayer per selective_scan_optimized_v2
public final class MambaLayer {
    private let device: MTLDevice
    private let config: SSMConfig
    private let pipelineState: MTLComputePipelineState
    private let commandQueue: MTLCommandQueue
    
    // MARK: - Buffer per kernel ottimizzato
    private var deltaBuffer: MTLBuffer!
    private var A_logBuffer: MTLBuffer!
    private var B_ssmBuffer: MTLBuffer!
    private var C_ssmBuffer: MTLBuffer!
    private var DBuffer: MTLBuffer!
    private var paramsBuffer: MTLBuffer!
    
    // MARK: - Configurazione Debug
    public var gridDebugConfig: GridDebugConfig = GridDebugConfig()
    
    // MARK: - Initialization
    public init?(
        device: MTLDevice,
        config: SSMConfig,
        metalSource: String
    ) async {
        self.device = device
        self.config = config
        
        guard let queue = device.makeCommandQueue() else {
            print("❌ Failed to create command queue")
            return nil
        }
        self.commandQueue = queue
        
        // Compile Metal source
        guard let library = try? await device.makeLibrary(source: metalSource, options: nil) else {
            print("❌ Failed to compile Metal library")
            return nil
        }
        
        guard let kernel = library.makeFunction(name: "selective_scan_optimized_v2") else {
            print("❌ Kernel 'selective_scan_optimized_v2' not found")
            return nil
        }
        
        do {
            self.pipelineState = try await device.makeComputePipelineState(function: kernel)
        } catch {
            print("❌ Failed to create pipeline state: \(error)")
            return nil
        }
        
        // Inizializza buffer SSM con dimensioni corrette
        guard let delta = device.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let A_log = device.makeBuffer(length: 16 * 4, options: .storageModeShared),
              let B_ssm = device.makeBuffer(length: 512 * 16 * 4, options: .storageModeShared),
              let C_ssm = device.makeBuffer(length: 512 * 16 * 4, options: .storageModeShared),
              let D = device.makeBuffer(length: 512 * 4, options: .storageModeShared),
              let params = device.makeBuffer(length: 16, options: .storageModeShared) else {
            print("❌ Buffer allocation failed")
            return nil
        }
        
        self.deltaBuffer = delta
        self.A_logBuffer = A_log
        self.B_ssmBuffer = B_ssm
        self.C_ssmBuffer = C_ssm
        self.DBuffer = D
        self.paramsBuffer = params
        
        // Popola con dati di test/placeholder
        initializeSSMBuffers()
        
        print("✅ MambaLayer Opt initialized")
    }
    
    // MARK: - Forward Pass Standard
    public func forward(x: Tensor) async throws -> Tensor {
        let batchSize = x.shape[0]
        let seqLen = x.shape[1]
        let hiddenDim = x.shape[2]
        
        guard let outputBuffer = device.makeBuffer(
            length: batchSize * seqLen * hiddenDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw NSError(domain: "MambaLayer", code: -1)
        }
        
        // Setup params: batch, seq, dInner, dState
        var params = (UInt32(batchSize), UInt32(seqLen), UInt32(hiddenDim), UInt32(config.dState))
        memcpy(paramsBuffer.contents(), &params, 16)
        
        let threads = 64
        let groups = (hiddenDim + threads - 1) / threads
        
        // Threadgroup size: 256 threads x float4
        let threadgroupMemLength = 256 * 4 * MemoryLayout<Float>.stride
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(x.buffer, offset: 0, index: 0)
        encoder.setBuffer(deltaBuffer, offset: 0, index: 1)
        encoder.setBuffer(A_logBuffer, offset: 0, index: 2)
        encoder.setBuffer(B_ssmBuffer, offset: 0, index: 3)
        encoder.setBuffer(C_ssmBuffer, offset: 0, index: 4)
        encoder.setBuffer(DBuffer, offset: 0, index: 5)
        encoder.setBuffer(outputBuffer, offset: 0, index: 6)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 7)
        encoder.setThreadgroupMemoryLength(threadgroupMemLength, index: 0)
        
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: seqLen, depth: batchSize),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        
        commandBuffer.commit()
        await commandBuffer.completed()  // ✅ CORRETTO
        
        return Tensor(buffer: outputBuffer, shape: x.shape, category: .activations)
    }
}

// MARK: - Configurazione Debug
public struct GridDebugConfig {
    public var gridSweep: Bool
    public var fixedThreads: Int
    public let cpuBaselineMs: Double
    
    public init(gridSweep: Bool = false, fixedThreads: Int = 64, cpuBaselineMs: Double = 127.0) {
        self.gridSweep = gridSweep
        self.fixedThreads = fixedThreads
        self.cpuBaselineMs = cpuBaselineMs
    }
}

// MARK: - MambaLayer Estensione Debug
public extension MambaLayer {
    
    /// Forward pass con debug del grid
    func forwardWithDebug(x: Tensor) async throws -> (output: Tensor, metrics: [String: Double]) {
        let batchSize = x.shape[0]
        let seqLen = x.shape[1]
        let hiddenDim = x.shape[2]
        var metrics: [String: Double] = [:]
        
        // Buffer di output
        guard let outputBuffer = device.makeBuffer(
            length: batchSize * seqLen * hiddenDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw NSError(domain: "MambaLayer", code: -1)
        }
        
        // Setup params
        var params = (UInt32(batchSize), UInt32(seqLen), UInt32(hiddenDim), UInt32(config.dState))
        memcpy(paramsBuffer.contents(), &params, 16)
        
        // 🔍 SWEEP DEL GRID (eseguito solo se attivo)
        if gridDebugConfig.gridSweep {
            print("\n🔍 GRID SWEEP TEST per shape [\(batchSize), \(seqLen), \(hiddenDim)]")
            let candidates = [64, 96, 128, 192, 256]
            var bestTime = Double.infinity
            var bestThreads = 64
            
            for threads in candidates {
                let groups = (hiddenDim + threads - 1) / threads
                let threadgroupMemLength = threads * 4 * MemoryLayout<Float>.stride
                
                let gpuMs = await measureGPUExecution(
                    input: x,
                    output: outputBuffer,
                    threadgroups: MTLSize(width: groups, height: seqLen, depth: batchSize),
                    threadsPerGroup: MTLSize(width: threads, height: 1, depth: 1),
                    hiddenDim: hiddenDim,
                    threadgroupMemLength: threadgroupMemLength
                )
                print("  T=\(threads) G=\(groups): \(String(format: "%.2f", gpuMs))ms")
                
                if gpuMs < bestTime {
                    bestTime = gpuMs
                    bestThreads = threads
                }
            }
            
            // Salva risultati nelle metriche
            metrics["bestThreads"] = Double(bestThreads)
            metrics["bestTime"] = bestTime
            
            // Disabilita sweep per le prossime chiamate
            gridDebugConfig.gridSweep = false
            gridDebugConfig.fixedThreads = bestThreads
            
            print("🏆 Miglior configurazione: T=\(bestThreads) → \(String(format: "%.2f", bestTime))ms")
        }
        
        // 🚀 ESECUZIONE CON CONFIG OTTIMALE
        let threads = gridDebugConfig.fixedThreads
        let groups = (hiddenDim + threads - 1) / threads
        let threadgroupMemLength = threads * 4 * MemoryLayout<Float>.stride
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(x.buffer, offset: 0, index: 0)
        encoder.setBuffer(deltaBuffer, offset: 0, index: 1)
        encoder.setBuffer(A_logBuffer, offset: 0, index: 2)
        encoder.setBuffer(B_ssmBuffer, offset: 0, index: 3)
        encoder.setBuffer(C_ssmBuffer, offset: 0, index: 4)
        encoder.setBuffer(DBuffer, offset: 0, index: 5)
        encoder.setBuffer(outputBuffer, offset: 0, index: 6)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 7)
        encoder.setThreadgroupMemoryLength(threadgroupMemLength, index: 0)
        
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: seqLen, depth: batchSize),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        
        commandBuffer.commit()
        await commandBuffer.completed()
        
        let output = Tensor(buffer: outputBuffer, shape: x.shape, category: .activations)
        metrics["gpuTime"] = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000
        
        return (output, metrics)
    }
    
    // MARK: - Misurazione GPU
    private func measureGPUExecution(
        input: Tensor,
        output: MTLBuffer,
        threadgroups: MTLSize,
        threadsPerGroup: MTLSize,
        hiddenDim: Int,
        threadgroupMemLength: Int
    ) async -> Double {
        
        let desc = MTLCommandBufferDescriptor()
        desc.errorOptions = .encoderExecutionStatus
        
        let commandBuffer = commandQueue.makeCommandBuffer(descriptor: desc)!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(deltaBuffer, offset: 0, index: 1)
        encoder.setBuffer(A_logBuffer, offset: 0, index: 2)
        encoder.setBuffer(B_ssmBuffer, offset: 0, index: 3)
        encoder.setBuffer(C_ssmBuffer, offset: 0, index: 4)
        encoder.setBuffer(DBuffer, offset: 0, index: 5)
        encoder.setBuffer(output, offset: 0, index: 6)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 7)
        encoder.setThreadgroupMemoryLength(threadgroupMemLength, index: 0)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        await commandBuffer.completed()
        
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000
    }
    
    // MARK: - Inizializza Buffer SSM
    private func initializeSSMBuffers() {
    // Delta: tutti 1.0
    let ptr = deltaBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<512 { ptr[i] = 1.0 }
    
    // A_log: valori logaritmici da -0.1 a -2.0
    let ptrA = A_logBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<16 { ptrA[i] = -0.5 - Float(i) * 0.1 }
    
    // B_ssm, C_ssm: matrici identità approssimate
    let bptr = B_ssmBuffer.contents().assumingMemoryBound(to: Float.self)
    let cptr = C_ssmBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(512*16) {
        bptr[i] = (i % 16 == i % 512) ? 1.0 : 0.0
        cptr[i] = (i % 16 == i % 512) ? 1.0 : 0.0
    }
    
    // D: bias a zero
    let ptrD = DBuffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<512 { ptrD[i] = 0.0 }
        }
    }
}

import Foundation
import Metal
import MagellanoCore

public class AdamWMetal: Optimizer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState
    private let pipelineStateSIMD8: MTLComputePipelineState
    
    private var mBuffers: [String: MTLBuffer] = [:]
    private var vBuffers: [String: MTLBuffer] = [:]
    private var vMaxBuffers: [String: MTLBuffer] = [:]
    
    public var learningRate: Float = 1e-3
    public var beta1: Float = 0.9
    public var beta2: Float = 0.999
    public var epsilon: Float = 1e-8
    public var weightDecay: Float = 0.01
    public var useAmsGrad = true
    
    private var t = 0
    private let lock = NSLock()
    
    public init?(device: MTLDevice) {
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            print("❌ Cannot create command queue")
            return nil
        }
        
        // Carica metallib compilato
        let cwd = FileManager.default.currentDirectoryPath
        let metalLibPath = "\(cwd)/AdamW.metallib"
        let url = URL(fileURLWithPath: metalLibPath)
        
        guard let library = try? device.makeLibrary(URL: url) else {
            print("❌ Cannot load AdamW.metallib from: \(metalLibPath)")
            return nil
        }
        
        guard let function = library.makeFunction(name: "adamw_fp16_v2"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            print("❌ Cannot create pipeline for adamw_fp16_v2")
            return nil
        }
        
        self.commandQueue = queue
        self.pipelineState = pipeline
        
        // SIMD8 kernel (fallback a standard se non disponibile)
        if let functionSIMD8 = library.makeFunction(name: "adamw_fp16_simd8"),
           let pipelineSIMD8 = try? device.makeComputePipelineState(function: functionSIMD8) {
            self.pipelineStateSIMD8 = pipelineSIMD8
        } else {
            self.pipelineStateSIMD8 = pipeline
        }
    }
    
    public func step(parameters: inout [Tensor], gradients: [Tensor]) {
        lock.lock()
        t += 1
        lock.unlock()
        
        for i in 0..<parameters.count {
            updateParameter(&parameters[i], gradient: gradients[i], index: i)
        }
    }
    
    private func updateParameter(_ parameter: inout Tensor, gradient: Tensor, index: Int) {
        let n = parameter.elementCount
        let key = "param_\(index)"
        let bufferSize = n * MemoryLayout<UInt16>.size
        
        // Inizializza buffer momenti
        if mBuffers[key] == nil {
            guard let mBuf = device.makeBuffer(length: bufferSize, options: .storageModePrivate),
                  let vBuf = device.makeBuffer(length: bufferSize, options: .storageModePrivate) else {
                return
            }
            
            let cmdBuf = commandQueue.makeCommandBuffer()
            let blit = cmdBuf?.makeBlitCommandEncoder()
            blit?.fill(buffer: mBuf, range: 0..<bufferSize, value: 0)
            blit?.fill(buffer: vBuf, range: 0..<bufferSize, value: 0)
            blit?.endEncoding()
            cmdBuf?.commit()
            cmdBuf?.waitUntilCompleted()
            
            mBuffers[key] = mBuf
            vBuffers[key] = vBuf
            
            if useAmsGrad {
                guard let vMaxBuf = device.makeBuffer(length: bufferSize, options: .storageModePrivate) else {
                    return
                }
                vMaxBuffers[key] = vMaxBuf
            }
        }
        
        // Output buffer
        guard let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModePrivate) else {
            return
        }
        
        // Seleziona kernel
        let useSIMD8 = (n % 8 == 0) && (n >= 1024)
        let pipeline = useSIMD8 ? pipelineStateSIMD8 : pipelineState
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(parameter.buffer, offset: 0, index: 0)
        encoder.setBuffer(gradient.buffer, offset: 0, index: 1)
        encoder.setBuffer(mBuffers[key], offset: 0, index: 2)
        encoder.setBuffer(vBuffers[key], offset: 0, index: 3)
        encoder.setBuffer(vMaxBuffers[key], offset: 0, index: 4)
        encoder.setBuffer(outputBuffer, offset: 0, index: 5)
        
        var lr = learningRate, b1 = beta1, b2 = beta2
        var eps = epsilon, wd = weightDecay, step = t
        var amsgrad = useAmsGrad
        
        encoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 6)
        encoder.setBytes(&b1, length: MemoryLayout<Float>.size, index: 7)
        encoder.setBytes(&b2, length: MemoryLayout<Float>.size, index: 8)
        encoder.setBytes(&eps, length: MemoryLayout<Float>.size, index: 9)
        encoder.setBytes(&wd, length: MemoryLayout<Float>.size, index: 10)
        encoder.setBytes(&step, length: MemoryLayout<Int>.size, index: 11)
        encoder.setBytes(&amsgrad, length: MemoryLayout<Bool>.size, index: 12)
        
        let elementsPerGroup = useSIMD8 ? 8 : 256
        let threadgroups = MTLSize(width: (n + elementsPerGroup - 1) / elementsPerGroup, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: useSIMD8 ? 1 : 256, height: 1, depth: 1)
        
        encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float16>.size, index: 0)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Copy output → parameter
        let copyCmdBuf = commandQueue.makeCommandBuffer()
        let copyEncoder = copyCmdBuf?.makeBlitCommandEncoder()
        copyEncoder?.copy(from: outputBuffer, sourceOffset: 0, to: parameter.buffer, destinationOffset: 0, size: bufferSize)
        copyEncoder?.endEncoding()
        copyCmdBuf?.commit()
        copyCmdBuf?.waitUntilCompleted()
    }
    
    public func zeroGradients() {}
    
    public func getState() -> [String: Any] {
        ["t": t, "learning_rate": learningRate, "beta1": beta1, "beta2": beta2,
         "epsilon": epsilon, "weight_decay": weightDecay, "use_amsgrad": useAmsGrad]
    }
    
    public func loadState(_ state: [String: Any]) {
        t = state["t"] as? Int ?? 0
        learningRate = state["learning_rate"] as? Float ?? 1e-3
        beta1 = state["beta1"] as? Float ?? 0.9
        beta2 = state["beta2"] as? Float ?? 0.999
        epsilon = state["epsilon"] as? Float ?? 1e-8
        weightDecay = state["weight_decay"] as? Float ?? 0.01
        useAmsGrad = state["use_amsgrad"] as? Bool ?? true
    }
}

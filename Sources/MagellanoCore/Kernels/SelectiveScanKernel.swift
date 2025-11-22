// Sources/MagellanoCore/Kernels/SelectiveScanKernel.swift

import Foundation
import Metal

/// GPU-only Selective Scan Executor - Ultra-optimized version
public final class SelectiveScanKernel {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let tgMemSize: Int
    
    // ✅ Pipeline accessibile pubblicamente
    private let _pipelineState: MTLComputePipelineState
    public var pipelineState: MTLComputePipelineState { _pipelineState }
    
    private var paramsBuffer: MTLBuffer
    private let params: Params
    
    struct Params: ContiguousBytes {
        let batchSize: UInt32
        let seqLength: UInt32
        let dInner: UInt32
        let dState: UInt32
        
        func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
            var copy = self
            return try Swift.withUnsafeBytes(of: &copy, body)
        }
    }
    
    public init?(device: MTLDevice, metalSource: String, config: SSMConfig) async {
        self.device = device
        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        
        let threadsPerGroup = min(256, config.dInner)
        self.tgMemSize = threadsPerGroup * 4 * MemoryLayout<Float>.size * 2
        
        self.params = Params(batchSize: 0, seqLength: 0, dInner: UInt32(config.dInner), dState: UInt32(config.dState))
        
        guard let paramsBuf = device.makeBuffer(length: MemoryLayout<Params>.size, options: .storageModeShared) else {
            return nil
        }
        self.paramsBuffer = paramsBuf
        
        do {
            let lib = try await device.makeLibrary(source: metalSource, options: nil)
            guard let kernelFunc = lib.makeFunction(name: "selective_scan_optimized_v2") else {
                print("❌ Kernel function 'selective_scan_optimized_v2' not found")
                return nil
            }
            self._pipelineState = try await device.makeComputePipelineState(function: kernelFunc)
        } catch {
            print("❌ Metal kernel compilation failed: \(error)")
            return nil
        }
    }
    
    public func encode(
        commandBuffer: MTLCommandBuffer,
        x: Tensor,
        delta: Tensor,
        A: Tensor,
        B_ssm: Tensor,
        C_ssm: Tensor,
        D: Tensor,
        output: Tensor
    ) {
        #if DEBUG
        precondition(x.shape.count == 3 && delta.shape == x.shape, "Invalid tensor shapes")
        #endif
        
        var params = Params(
            batchSize: UInt32(x.shape[0]),
            seqLength: UInt32(x.shape[1]),
            dInner: UInt32(x.shape[2]),
            dState: UInt32(A.shape[1])
        )
        
        memcpy(paramsBuffer.contents(), &params, MemoryLayout<Params>.size)
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create compute encoder")
        }
        
        encoder.setComputePipelineState(_pipelineState)
        encoder.setBuffer(x.buffer, offset: 0, index: 0)
        encoder.setBuffer(delta.buffer, offset: 0, index: 1)
        encoder.setBuffer(A.buffer, offset: 0, index: 2)
        encoder.setBuffer(B_ssm.buffer, offset: 0, index: 3)
        encoder.setBuffer(C_ssm.buffer, offset: 0, index: 4)
        encoder.setBuffer(D.buffer, offset: 0, index: 5)
        encoder.setBuffer(output.buffer, offset: 0, index: 6)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 7)
        encoder.setThreadgroupMemoryLength(tgMemSize, index: 0)
        
        // ✅ CORRETTO: Griglia per tiling 64x
        let threadsPerGroup = min(256, x.shape[2])
        let numSeqTiles = (x.shape[1] + 63) / 64  // Numero di tiles di 64 elementi
        
        let gridSize = MTLSize(
            width: x.shape[0],       // batch dimension (2)
            height: numSeqTiles,     // tiles di sequenza (4 per seqLen 256)
            depth: 1                 // non usato
        )
        let threadGroupSize = MTLSize(
            width: threadsPerGroup,  // canali per gruppo (256)
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}

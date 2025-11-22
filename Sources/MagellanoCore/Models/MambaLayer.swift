// Sources/MagellanoCore/Models/MambaLayer.swift
import Foundation
import Metal
import Accelerate

public final class MambaLayer: @unchecked Sendable {
    private let device: MTLDevice
    private let config: SSMConfig
    
    // Learnable parameters (internal per quantizzazione)
    internal let inProj: Tensor       // [dModel, 2*dInner] - projects to x and z
    internal let convWeight: Tensor   // [dInner, 1, dConv]
    internal let xProj: Tensor        // [dInner, dtRank + 2*dState]
    internal let dtProj: Tensor       // [dtRank, dInner]
    internal let ALog: Tensor         // [dInner, dState] - log(A) for stability
    internal let D: Tensor            // [dInner] - skip connection
    internal let outProj: Tensor      // [dInner, dModel]
    internal let norm: Tensor         // [dModel] - RMSNorm weights
    
    // Metal kernel (optional)
    private let pipelineState: MTLComputePipelineState?
    private let commandQueue: MTLCommandQueue?
    
    public init?(device: MTLDevice, config: SSMConfig, metalSource: String) async {
        self.device = device
        self.config = config
        
        let dModel = config.dModel
        let dInner = config.dInner
        let dState = config.dState
        let dConv = config.dConv
        let dtRank = config.dtRank
        
        // Initialize all weight tensors
        let scale = sqrt(2.0 / Float(dModel))
        
        guard let inProj = Tensor.randn(device: device, shape: [dModel, 2 * dInner], std: scale, category: .modelWeights),
              let convWeight = Tensor.randn(device: device, shape: [dInner, 1, dConv], std: 0.02, category: .modelWeights),
              let xProj = Tensor.randn(device: device, shape: [dInner, dtRank + 2 * dState], std: scale, category: .modelWeights),
              let dtProj = Tensor.randn(device: device, shape: [dtRank, dInner], std: scale, category: .modelWeights),
              let ALog = Tensor.randn(device: device, shape: [dInner, dState], std: 0.1, category: .modelWeights),
              let D = Tensor.ones(device: device, shape: [dInner], category: .modelWeights),
              let outProj = Tensor.randn(device: device, shape: [dInner, dModel], std: scale, category: .modelWeights),
              let norm = Tensor.ones(device: device, shape: [dModel], category: .modelWeights) else {
            return nil
        }
        
        self.inProj = inProj
        self.convWeight = convWeight
        self.xProj = xProj
        self.dtProj = dtProj
        self.ALog = ALog
        self.D = D
        self.outProj = outProj
        self.norm = norm
        
        // Try to compile Metal kernel
        if let library = try? await device.makeLibrary(source: metalSource, options: nil),
           let kernel = library.makeFunction(name: "selective_scan_optimized_v2"),
           let pipeline = try? await device.makeComputePipelineState(function: kernel),
           let queue = device.makeCommandQueue() {
            self.pipelineState = pipeline
            self.commandQueue = queue
            print("✅ MambaLayer Opt initialized")
        } else {
            self.pipelineState = nil
            self.commandQueue = nil
            print("⚠️  MambaLayer CPU fallback (Metal kernel unavailable)")
        }
    }
    
    public func forward(x: Tensor) async throws -> Tensor {
        // Pre-norm
        guard let xNorm = x.rmsNorm(eps: 1e-5) else {
            throw TensorError.operationFailed("RMSNorm failed")
        }
        
        let B = x.shape[0]
        let L = x.shape[1]
        let D = x.shape[2]
        
        // Project to 2*dInner and split
        guard let xFlat = xNorm.reshape([B * L, D]),
              let projected = Tensor.matmul(device: device, a: xFlat, b: inProj) else {
            throw TensorError.operationFailed("inProj matmul failed")
        }
        
        // Split into x and z paths
        let dInner = config.dInner
        guard let xPath = projected.slice(dim: 1, start: 0, end: dInner),
              let zPath = projected.slice(dim: 1, start: dInner, end: 2 * dInner) else {
            throw TensorError.operationFailed("Split failed")
        }
        
        // Apply SiLU to z
        guard let zSilu = zPath.silu() else {
            throw TensorError.operationFailed("SiLU failed")
        }
        
        // Selective scan on x (CPU fallback)
        guard let ssm_out = selectiveScanCPU(x: xPath) else {
            throw TensorError.operationFailed("SSM failed")
        }
        
        // Element-wise multiply with gate
        guard let gated = Tensor.multiply(device: device, a: ssm_out, b: zSilu) else {
            throw TensorError.operationFailed("Gating failed")
        }
        
        // Project back to dModel
        guard let output = Tensor.matmul(device: device, a: gated, b: outProj) else {
            throw TensorError.operationFailed("outProj failed")
        }
        
        // Reshape and residual
        guard let outputReshaped = output.reshape([B, L, D]),
              let result = Tensor.add(device: device, a: x, b: outputReshaped) else {
            throw TensorError.operationFailed("Residual failed")
        }
        
        return result
    }
    
    // CPU fallback for selective scan
    private func selectiveScanCPU(x: Tensor) -> Tensor? {
        // Simplified SSM: just apply D skip connection for now
        let dInner = config.dInner
        guard let output = Tensor.zeros(device: device, shape: x.shape, category: .activations) else {
            return nil
        }
        
        let xPtr = x.buffer.contents().bindMemory(to: Float.self, capacity: x.elementCount)
        let dPtr = D.buffer.contents().bindMemory(to: Float.self, capacity: dInner)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.elementCount)
        
        let BL = x.shape[0]
        for i in 0..<BL {
            for j in 0..<dInner {
                outPtr[i * dInner + j] = xPtr[i * dInner + j] * dPtr[j]
            }
        }
        
        return output
    }
}



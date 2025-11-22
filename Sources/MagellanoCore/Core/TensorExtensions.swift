import Foundation
import Accelerate
import Metal

extension Tensor {
    /// Softplus activation: ln(1 + exp(x))
    public func softplus() -> Tensor? {
        // Crea nuovo buffer sullo stesso device
        let device = buffer.device
        guard let newBuffer = device.makeBuffer(
            length: elementCount * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            return nil
        }
        
        let srcPtr = buffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        let dstPtr = newBuffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        
        // Softplus: log(1 + exp(x))
        for i in 0..<elementCount {
            let x = srcPtr[i]
            dstPtr[i] = x > 20.0 ? x : log1p(exp(x))
        }
        
        return Tensor(buffer: newBuffer, shape: shape, category: .activations)
    }
}

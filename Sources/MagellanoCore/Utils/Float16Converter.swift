import Foundation
import Accelerate

public extension Tensor {
    /// Converti Float32 → Float16 (batch efficiente)
    func toFloat16() -> Tensor? {
        guard dtype == .float32 else { return nil }
        let device = buffer.device
        
        let n = elementCount
        guard let result = Tensor(device: device, shape: shape, dtype: .float16) else {
            return nil
        }
        
        let srcPtr = buffer.contents().bindMemory(to: Float.self, capacity: n)
        let dstPtr = result.buffer.contents().bindMemory(to: UInt16.self, capacity: n)
        
        // Conversione batch con vImage (4096 elementi/volta)
        let batchSize = 4096
        var remaining = n
        var offset = 0
        
        while remaining > 0 {
            let current = min(batchSize, remaining)
            
            var src = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: srcPtr + offset),
                height: 1,
                width: vImagePixelCount(current),
                rowBytes: current * MemoryLayout<Float>.size
            )
            
            var dst = vImage_Buffer(
                data: dstPtr + offset,
                height: 1,
                width: vImagePixelCount(current),
                rowBytes: current * MemoryLayout<UInt16>.size
            )
            
            guard vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0) == kvImageNoError else {
                return nil
            }
            
            offset += current
            remaining -= current
        }
        
        return result
    }
    
    /// Converti Float16 → Float32
    func toFloat32() -> Tensor? {
        guard dtype == .float16 else { return nil }
        let device = buffer.device
        
        let n = elementCount
        guard let result = Tensor(device: device, shape: shape, dtype: .float32) else {
            return nil
        }
        
        let srcPtr = buffer.contents().bindMemory(to: UInt16.self, capacity: n)
        let dstPtr = result.buffer.contents().bindMemory(to: Float.self, capacity: n)
        
        let batchSize = 4096
        var remaining = n
        var offset = 0
        
        while remaining > 0 {
            let current = min(batchSize, remaining)
            
            var src = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: srcPtr + offset),
                height: 1,
                width: vImagePixelCount(current),
                rowBytes: current * MemoryLayout<UInt16>.size
            )
            
            var dst = vImage_Buffer(
                data: dstPtr + offset,
                height: 1,
                width: vImagePixelCount(current),
                rowBytes: current * MemoryLayout<Float>.size
            )
            
            guard vImageConvert_Planar16FtoPlanarF(&src, &dst, 0) == kvImageNoError else {
                return nil
            }
            
            offset += current
            remaining -= current
        }
        
        return result
    }
    
    /// Auto-conversione per mixed precision
    func ensureFloat16() -> Tensor? {
        if dtype == .float16 { return self }
        if elementCount < 1024 {
            print("⚠️ Tensor <1K: mantieni Float32")
            return self
        }
        return toFloat16()
    }
    
    /// Calcola errore conversione Float32→Float16→Float32
    func float16PrecisionLoss() -> Float? {
        guard dtype == .float32,
              let fp16 = toFloat16(),
              let restored = fp16.toFloat32() else {
            return nil
        }
        
        let orig = toArray()
        let rest = restored.toArray()
        
        var sumDiff: Float = 0
        for i in 0..<orig.count {
            sumDiff += abs(orig[i] - rest[i])
        }
        
        return sumDiff / Float(orig.count)
    }
}
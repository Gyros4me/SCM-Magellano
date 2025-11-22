// Sources/MagellanoCore/Core/Tensor.swift

import Foundation
import Metal
import Accelerate

public final class Tensor: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let shape: [Int]
    public let dtype: DataType
    private let tracked: TrackedBuffer
    
    public enum DataType: Sendable {
        case float32, float16, int8, nf4
        
        var bytesPerElement: Int {
            switch self {
            case .float32: return 4
            case .float16: return 2
            case .int8: return 1
            case .nf4: return 1 // 2 elements per byte
            }
        }
    }
    
    // MARK: - Initializers
    
    public init(buffer: MTLBuffer, shape: [Int], dtype: DataType = .float32, category: MemorySnapshot.MemoryCategory = .temporary) {
        self.buffer = buffer
        self.shape = shape
        self.dtype = dtype
        self.tracked = TrackedBuffer(buffer: buffer, category: category, label: "tensor_\(shape)")
    }
    
    public convenience init?(device: MTLDevice, shape: [Int], dtype: DataType = .float32, category: MemorySnapshot.MemoryCategory = .temporary) {
        let elementCount = shape.reduce(1, *)
        let byteCount = elementCount * dtype.bytesPerElement
        
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        
        self.init(buffer: buffer, shape: shape, dtype: dtype, category: category)
    }
    
    // MARK: - Properties
    
    public var elementCount: Int {
        shape.reduce(1, *)
    }
    
    public var byteCount: Int {
        elementCount * dtype.bytesPerElement
    }
    
    public var rank: Int {
        shape.count
    }
    
    // MARK: - Data Access (Float32 only for now)
    
    public func toArray() -> [Float] {
        guard dtype == .float32 else { fatalError("toArray() only supports float32") }
        
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        return Array(UnsafeBufferPointer(start: ptr, count: elementCount))
    }
    
    public func fill(_ value: Float) {
        guard dtype == .float32 else { return }
        
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        for i in 0..<elementCount {
            ptr[i] = value
        }
    }
    
    public func fillRandom(mean: Float = 0.0, std: Float = 1.0) {
        guard dtype == .float32 else { return }
        
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        for i in 0..<elementCount {
            // Box-Muller transform
            let u1 = Float.random(in: 0..<1)
            let u2 = Float.random(in: 0..<1)
            let z = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
            ptr[i] = mean + std * z
        }
    }
    
    // MARK: - Shape Operations
    
    public func reshape(_ newShape: [Int]) -> Tensor? {
        let newElementCount = newShape.reduce(1, *)
        guard newElementCount == elementCount else { return nil }
        
        return Tensor(buffer: buffer, shape: newShape, dtype: dtype, category: tracked.category)
    }
    
    public func view(as newShape: [Int]) -> Tensor? {
        reshape(newShape)
    }
    
    // MARK: - Static Constructors
    
    public static func zeros(device: MTLDevice, shape: [Int], dtype: DataType = .float32, category: MemorySnapshot.MemoryCategory = .temporary) -> Tensor? {
        guard let tensor = Tensor(device: device, shape: shape, dtype: dtype, category: category) else {
            return nil
        }
        tensor.fill(0.0)
        return tensor
    }
    
    public static func ones(device: MTLDevice, shape: [Int], dtype: DataType = .float32, category: MemorySnapshot.MemoryCategory = .temporary) -> Tensor? {
        guard let tensor = Tensor(device: device, shape: shape, dtype: dtype, category: category) else {
            return nil
        }
        tensor.fill(1.0)
        return tensor
    }
    
    public static func randn(device: MTLDevice, shape: [Int], mean: Float = 0.0, std: Float = 1.0, dtype: DataType = .float32, category: MemorySnapshot.MemoryCategory = .temporary) -> Tensor? {
        guard let tensor = Tensor(device: device, shape: shape, dtype: dtype, category: category) else {
            return nil
        }
        tensor.fillRandom(mean: mean, std: std)
        return tensor
    }
}

// MARK: - CPU Operations (Accelerate framework)

extension Tensor {
    
    /// Matrix multiplication C = A @ B usando Accelerate
    public static func matmul(device: MTLDevice, a: Tensor, b: Tensor) -> Tensor? {
        guard a.dtype == .float32 && b.dtype == .float32 else { return nil }
        guard a.rank == 2 && b.rank == 2 else { return nil }
        guard a.shape[1] == b.shape[0] else { return nil }
        
        let M = a.shape[0]
        let K = a.shape[1]
        let N = b.shape[1]
        
        guard let c = Tensor.zeros(device: device, shape: [M, N], category: .temporary) else {
            return nil
        }
        
        let aPtr = a.buffer.contents().bindMemory(to: Float.self, capacity: a.elementCount)
        let bPtr = b.buffer.contents().bindMemory(to: Float.self, capacity: b.elementCount)
        let cPtr = c.buffer.contents().bindMemory(to: Float.self, capacity: c.elementCount)
        
        // GEMM: C = alpha*A*B + beta*C
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            Int32(M), Int32(N), Int32(K),
            1.0, // alpha
            aPtr, Int32(K),
            bPtr, Int32(N),
            0.0, // beta
            cPtr, Int32(N)
        )
        
        return c
    }
    
    /// Element-wise addition
    public static func add(device: MTLDevice, a: Tensor, b: Tensor) -> Tensor? {
        guard a.shape == b.shape && a.dtype == .float32 && b.dtype == .float32 else { return nil }
        
        guard let c = Tensor.zeros(device: device, shape: a.shape, category: .temporary) else {
            return nil
        }
        
        let aPtr = a.buffer.contents().bindMemory(to: Float.self, capacity: a.elementCount)
        let bPtr = b.buffer.contents().bindMemory(to: Float.self, capacity: b.elementCount)
        let cPtr = c.buffer.contents().bindMemory(to: Float.self, capacity: c.elementCount)
        
        vDSP_vadd(aPtr, 1, bPtr, 1, cPtr, 1, vDSP_Length(a.elementCount))
        
        return c
    }
    
    /// RMSNorm (Root Mean Square Normalization)
    public func rmsNorm(eps: Float = 1e-6) -> Tensor? {
        guard dtype == .float32 else { return nil }
        
        let device = MTLCreateSystemDefaultDevice()!
        guard let output = Tensor.zeros(device: device, shape: shape, category: .temporary) else {
            return nil
        }
        
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        
        // Compute RMS per feature (last dimension)
        let featureDim = shape.last ?? 1
        let batchSize = elementCount / featureDim
        
        for b in 0..<batchSize {
            let offset = b * featureDim
            
            // Compute mean square
            var sumSquares: Float = 0.0
            for i in 0..<featureDim {
                let val = ptr[offset + i]
                sumSquares += val * val
            }
            
            let rms = sqrt(sumSquares / Float(featureDim) + eps)
            
            // Normalize
            for i in 0..<featureDim {
                outPtr[offset + i] = ptr[offset + i] / rms
            }
        }
        
        return output
    }
    
    /// SiLU activation: x * sigmoid(x)
    public func silu() -> Tensor? {
        guard dtype == .float32 else { return nil }
        
        let device = MTLCreateSystemDefaultDevice()!
        guard let output = Tensor.zeros(device: device, shape: shape, category: .temporary) else {
            return nil
        }
        
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        
        for i in 0..<elementCount {
            let x = ptr[i]
            let sigmoid = 1.0 / (1.0 + exp(-x))
            outPtr[i] = x * sigmoid
        }
        
        return output
    }
}

// MARK: - Debug

extension Tensor: CustomStringConvertible {
    public var description: String {
        "Tensor(shape: \(shape), dtype: \(dtype), bytes: \(byteCount))"
    }
}

extension Tensor {
    func slice(dim: Int, start: Int, end: Int) -> Tensor? {
        guard dim == 1, shape.count == 2 else { return nil }
        let rows = shape[0], cols = shape[1], sliceWidth = end - start
        guard let result = Tensor.zeros(device: self.buffer.device, shape: [rows, sliceWidth], category: .temporary) else { return nil }
        let srcPtr = self.buffer.contents().bindMemory(to: Float.self, capacity: self.elementCount)
        let dstPtr = result.buffer.contents().bindMemory(to: Float.self, capacity: result.elementCount)
        for r in 0..<rows {
            for c in 0..<sliceWidth {
                dstPtr[r * sliceWidth + c] = srcPtr[r * cols + start + c]
            }
        }
        return result
    }
    
    static func multiply(device: MTLDevice, a: Tensor, b: Tensor) -> Tensor? {
        guard a.shape == b.shape else { return nil }
        guard let result = Tensor.zeros(device: device, shape: a.shape, category: .temporary) else { return nil }
        let aPtr = a.buffer.contents().bindMemory(to: Float.self, capacity: a.elementCount)
        let bPtr = b.buffer.contents().bindMemory(to: Float.self, capacity: b.elementCount)
        let rPtr = result.buffer.contents().bindMemory(to: Float.self, capacity: result.elementCount)
        for i in 0..<a.elementCount { rPtr[i] = aPtr[i] * bPtr[i] }
        return result
    }
}

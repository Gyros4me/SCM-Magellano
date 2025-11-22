// Sources/MagellanoCore/Quantization/NF4Quantizer.swift
import Foundation
import Metal

public final class NF4Quantizer: @unchecked Sendable {
    private let device: MTLDevice
    
    // NF4 lookup table (optimized for normal distributions)
    private static let nf4Table: [Float] = [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ]
    
    public struct QuantizedTensor {
        let quantized: MTLBuffer
        let scaleL1: MTLBuffer       // FP16 per block
        let scaleL2: MTLBuffer?      // FP8 per superblock (double quant)
        let blockSize: Int
        let superblockSize: Int
        let shape: [Int]
        let doubleQuant: Bool
    }
    
    public init?(device: MTLDevice) {
        self.device = device
    }
    
    // Main quantization with double quantization option
    public func quantize(tensor: Tensor, blockSize: Int = 64, doubleQuant: Bool = true) -> QuantizedTensor? {
        let totalElements = tensor.elementCount
        let numBlocks = (totalElements + blockSize - 1) / blockSize
        let superblockSize = 256
        let numSuperblocks = doubleQuant ? (numBlocks + 3) / 4 : 0
        
        let quantizedSize = (totalElements + 1) / 2
        guard let quantizedBuf = device.makeBuffer(length: quantizedSize, options: .storageModeShared),
              let scaleL1Buf = device.makeBuffer(length: numBlocks * 2, options: .storageModeShared) else {
            return nil
        }
        
        let scaleL2Buf = doubleQuant ? device.makeBuffer(length: numSuperblocks, options: .storageModeShared) : nil
        
        let srcPtr = tensor.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let dstPtr = quantizedBuf.contents().bindMemory(to: UInt8.self, capacity: quantizedSize)
        let scaleL1Ptr = scaleL1Buf.contents().bindMemory(to: Float16.self, capacity: numBlocks)
        
        // First level: per-block quantization
        var blockScales = [Float](repeating: 0, count: numBlocks)
        
        for blockIdx in 0..<numBlocks {
            let blockStart = blockIdx * blockSize
            let blockEnd = min(blockStart + blockSize, totalElements)
            
            var absmax: Float = 0.0
            for i in blockStart..<blockEnd {
                absmax = max(absmax, abs(srcPtr[i]))
            }
            
            let scale = absmax / 1.0
            blockScales[blockIdx] = scale
            scaleL1Ptr[blockIdx] = Float16(scale)
            
            for i in blockStart..<blockEnd {
                let val = srcPtr[i] / max(scale, 1e-8)
                let quantIdx = findNearestNF4(val)
                
                let byteIdx = i / 2
                let isLowNibble = (i % 2) == 0
                
                if isLowNibble {
                    dstPtr[byteIdx] = (dstPtr[byteIdx] & 0xF0) | UInt8(quantIdx)
                } else {
                    dstPtr[byteIdx] = (dstPtr[byteIdx] & 0x0F) | (UInt8(quantIdx) << 4)
                }
            }
        }
        
        // Second level: quantize scales (FP16 → FP8)
        if doubleQuant, let scaleL2Buf = scaleL2Buf {
            let scaleL2Ptr = scaleL2Buf.contents().bindMemory(to: UInt8.self, capacity: numSuperblocks)
            
            for sbIdx in 0..<numSuperblocks {
                let sbStart = sbIdx * 4
                let sbEnd = min(sbStart + 4, numBlocks)
                
                var sbAbsmax: Float = 0.0
                for i in sbStart..<sbEnd {
                    sbAbsmax = max(sbAbsmax, blockScales[i])
                }
                
                let sbScale = sbAbsmax / 127.0  // FP8 E4M3 range
                scaleL2Ptr[sbIdx] = UInt8(min(sbScale * 127, 127))
                
                // Requantize L1 scales
                for i in sbStart..<sbEnd {
                    let quantVal = blockScales[i] / max(sbScale, 1e-8)
                    scaleL1Ptr[i] = Float16(quantVal)
                }
            }
        }
        
        return QuantizedTensor(
            quantized: quantizedBuf,
            scaleL1: scaleL1Buf,
            scaleL2: scaleL2Buf,
            blockSize: blockSize,
            superblockSize: superblockSize,
            shape: tensor.shape,
            doubleQuant: doubleQuant
        )
    }
    
    // Fast dequantization
    public func dequantize(_ quantized: QuantizedTensor) -> Tensor? {
        let totalElements = quantized.shape.reduce(1, *)
        guard let output = Tensor.zeros(device: device, shape: quantized.shape, category: .temporary) else {
            return nil
        }
        
        let srcPtr = quantized.quantized.contents().bindMemory(to: UInt8.self, capacity: (totalElements + 1) / 2)
        let scaleL1Ptr = quantized.scaleL1.contents().bindMemory(to: Float16.self, capacity: (totalElements + quantized.blockSize - 1) / quantized.blockSize)
        let dstPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        
        let numBlocks = (totalElements + quantized.blockSize - 1) / quantized.blockSize
        
        if quantized.doubleQuant, let scaleL2Buf = quantized.scaleL2 {
            let scaleL2Ptr = scaleL2Buf.contents().bindMemory(to: UInt8.self, capacity: (numBlocks + 3) / 4)
            
            for blockIdx in 0..<numBlocks {
                let blockStart = blockIdx * quantized.blockSize
                let blockEnd = min(blockStart + quantized.blockSize, totalElements)
                
                let sbIdx = blockIdx / 4
                let sbScale = Float(scaleL2Ptr[sbIdx]) / 127.0
                let blockScale = scaleL1Ptr[blockIdx].floatValue * sbScale
                
                for i in blockStart..<blockEnd {
                    let byteIdx = i / 2
                    let isLowNibble = (i % 2) == 0
                    let quantIdx = isLowNibble ? Int(srcPtr[byteIdx] & 0x0F) : Int((srcPtr[byteIdx] >> 4) & 0x0F)
                    dstPtr[i] = Self.nf4Table[quantIdx] * blockScale
                }
            }
        } else {
            for blockIdx in 0..<numBlocks {
                let blockStart = blockIdx * quantized.blockSize
                let blockEnd = min(blockStart + quantized.blockSize, totalElements)
                let scale = scaleL1Ptr[blockIdx].floatValue
                
                for i in blockStart..<blockEnd {
                    let byteIdx = i / 2
                    let isLowNibble = (i % 2) == 0
                    let quantIdx = isLowNibble ? Int(srcPtr[byteIdx] & 0x0F) : Int((srcPtr[byteIdx] >> 4) & 0x0F)
                    dstPtr[i] = Self.nf4Table[quantIdx] * scale
                }
            }
        }
        
        return output
    }
    
    private func findNearestNF4(_ value: Float) -> Int {
        var minDist = Float.infinity
        var bestIdx = 0
        for (idx, tableVal) in Self.nf4Table.enumerated() {
            let dist = abs(value - tableVal)
            if dist < minDist {
                minDist = dist
                bestIdx = idx
            }
        }
        return bestIdx
    }
}

struct Float16 {
    let bits: UInt16
    
    init(_ value: Float) {
        let bits32 = value.bitPattern
        let sign = (bits32 >> 31) & 0x1
        let exp = Int32((bits32 >> 23) & 0xFF) - 127 + 15
        let mant = (bits32 >> 13) & 0x3FF
        
        if exp <= 0 {
            self.bits = UInt16(sign << 15)
        } else if exp >= 31 {
            self.bits = UInt16((sign << 15) | 0x7C00)
        } else {
            self.bits = UInt16((sign << 15) | (UInt32(exp) << 10) | mant)
        }
    }
    
    var floatValue: Float {
        let sign = (bits >> 15) & 0x1
        let exp = Int32((bits >> 10) & 0x1F)
        let mant = bits & 0x3FF
        
        if exp == 0 { return sign == 1 ? -0.0 : 0.0 }
        if exp == 31 { return sign == 1 ? -Float.infinity : Float.infinity }
        
        let exp32 = (exp - 15 + 127) << 23
        let mant32 = UInt32(mant) << 13
        let sign32 = UInt32(sign) << 31
        
        return Float(bitPattern: sign32 | UInt32(exp32) | mant32)
    }
}

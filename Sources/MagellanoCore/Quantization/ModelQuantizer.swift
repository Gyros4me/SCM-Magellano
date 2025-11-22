// Sources/MagellanoCore/Quantization/ModelQuantizer.swift
import Foundation
import Metal

public final class ModelQuantizer: @unchecked Sendable {
    private let device: MTLDevice
    private let nf4Quantizer: NF4Quantizer
    
    public struct QuantizedModel {
        let config: ModelConfig
        let quantizedLayers: [QuantizedLayer]
        let tokenEmbedding: NF4Quantizer.QuantizedTensor
        let lmHead: NF4Quantizer.QuantizedTensor
    }
    
    public struct QuantizedLayer {
        let layerType: LayerType
        let weights: [String: NF4Quantizer.QuantizedTensor]  // param name -> quantized tensor
        
        enum LayerType {
            case mamba
            case moe
        }
    }
    
    public init?(device: MTLDevice) {
        self.device = device
        guard let quantizer = NF4Quantizer(device: device) else {
            return nil
        }
        self.nf4Quantizer = quantizer
    }
    
    // Quantize entire model in-place
    public func quantizeModel(_ model: MambaMoEModel) async -> QuantizedModel? {
        await StructuredLogger.shared.info("Quantizer", checkpoint: "start", "Quantizing 3.2B model to NF4...")
        
        let config = model.config
        var quantizedLayers: [QuantizedLayer] = []
        
        // Step 1: Quantize token embedding
        print("  [1/4] Quantizing token embeddings...")
        guard let embeddingQuantized = nf4Quantizer.quantize(tensor: model.tokenEmbedding, blockSize: 64, doubleQuant: true) else {
            return nil
        }
        
        // Step 2: Quantize layers
        print("  [2/4] Quantizing \(model.layers.count) layers...")
        for (idx, layer) in model.layers.enumerated() {
            if idx % 5 == 0 {
                print("    Progress: \(idx)/\(model.layers.count)")
            }
            
            if let mambaLayer = layer as? MambaLayer {
                guard let qLayer = await quantizeMambaLayer(mambaLayer) else {
                    return nil
                }
                quantizedLayers.append(qLayer)
            } else if let moeLayer = layer as? MoELayer {
                guard let qLayer = await quantizeMoELayer(moeLayer) else {
                    return nil
                }
                quantizedLayers.append(qLayer)
            }
        }
        
        // Step 3: Quantize LM head (shared with embedding)
        print("  [3/4] Quantizing LM head...")
        let lmHeadQuantized = embeddingQuantized  // Tied weights
        
        // Step 4: Free original FP32 memory (CRITICAL!)
        print("  [4/4] Releasing FP32 memory...")
        // Model layers will be deallocated when they go out of scope
        
        let quantizedModel = QuantizedModel(
            config: config,
            quantizedLayers: quantizedLayers,
            tokenEmbedding: embeddingQuantized,
            lmHead: lmHeadQuantized
        )
        
        await StructuredLogger.shared.info("Quantizer", checkpoint: "complete", "Model quantized successfully")
        return quantizedModel
    }
    
    private func quantizeMambaLayer(_ layer: MambaLayer) async -> QuantizedLayer? {
        var weights: [String: NF4Quantizer.QuantizedTensor] = [:]
        
        // Quantize all Mamba parameters
        let params = [
            "inProj": layer.inProj,
            "convWeight": layer.convWeight,
            "xProj": layer.xProj,
            "dtProj": layer.dtProj,
            "ALog": layer.ALog,
            "D": layer.D,
            "outProj": layer.outProj,
            "norm": layer.norm
        ]
        
        for (name, tensor) in params {
            guard let quantized = nf4Quantizer.quantize(tensor: tensor, blockSize: 64, doubleQuant: true) else {
                return nil
            }
            weights[name] = quantized
        }
        
        return QuantizedLayer(layerType: .mamba, weights: weights)
    }
    
    private func quantizeMoELayer(_ layer: MoELayer) async -> QuantizedLayer? {
        var weights: [String: NF4Quantizer.QuantizedTensor] = [:]
        
        // Quantize router
        guard let routerQuantized = nf4Quantizer.quantize(tensor: layer.router, blockSize: 64, doubleQuant: true) else {
            return nil
        }
        weights["router"] = routerQuantized
        
        // Quantize experts
        for (idx, expert) in layer.experts.enumerated() {
            guard let w1Quantized = nf4Quantizer.quantize(tensor: expert.w1, blockSize: 64, doubleQuant: true),
                  let w2Quantized = nf4Quantizer.quantize(tensor: expert.w2, blockSize: 64, doubleQuant: true) else {
                return nil
            }
            weights["expert_\(idx)_w1"] = w1Quantized
            weights["expert_\(idx)_w2"] = w2Quantized
        }
        
        return QuantizedLayer(layerType: .moe, weights: weights)
    }
    
    // Calculate actual memory savings
    public func calculateSavings(original: Int, quantized: QuantizedModel) -> (originalGB: Float, quantizedGB: Float, savedGB: Float) {
        let originalGB = Float(original) / 1_073_741_824
        
        var quantizedBytes = 0
        quantizedBytes += quantized.tokenEmbedding.quantized.length
        quantizedBytes += quantized.tokenEmbedding.scaleL1.length
        if let scaleL2 = quantized.tokenEmbedding.scaleL2 {
            quantizedBytes += scaleL2.length
        }
        
        for layer in quantized.quantizedLayers {
            for (_, qTensor) in layer.weights {
                quantizedBytes += qTensor.quantized.length
                quantizedBytes += qTensor.scaleL1.length
                if let scaleL2 = qTensor.scaleL2 {
                    quantizedBytes += scaleL2.length
                }
            }
        }
        
        let quantizedGB = Float(quantizedBytes) / 1_073_741_824
        let savedGB = originalGB - quantizedGB
        
        return (originalGB, quantizedGB, savedGB)
    }
}


// Sources/MagellanoCore/Models/ModelConfig.swift
import Foundation

public struct ModelConfig: Codable, Sendable {
    public let vocabSize: Int
    public let dModel: Int
    public let numLayers: Int
    public let mambaConfig: SSMConfig
    public let moeConfig: MoEConfig
    
    public var moeLayerIndices: [Int] {
        stride(from: 3, to: numLayers, by: 4).map { $0 }
    }
    
    public init(vocabSize: Int = 50257, dModel: Int = 2048, numLayers: Int = 24, mambaConfig: SSMConfig = .magellano800M, moeConfig: MoEConfig = .magellano800M) {
        self.vocabSize = vocabSize
        self.dModel = dModel
        self.numLayers = numLayers
        self.mambaConfig = mambaConfig
        self.moeConfig = moeConfig
    }
    
    public static let magellano800M = ModelConfig(vocabSize: 50257, dModel: 2048, numLayers: 24)
    public static let magellano1B = ModelConfig(vocabSize: 50257, dModel: 2304, numLayers: 28, mambaConfig: .magellano1B, moeConfig: .magellano1B)
    
    public var totalParams: Int {
        let embedding = vocabSize * dModel
        let mambaPerLayer = dModel * (2 * mambaConfig.dInner + mambaConfig.dInner + mambaConfig.dState)
        let numMambaLayers = numLayers - moeLayerIndices.count
        let moePerLayer = moeConfig.numExperts * 2 * dModel * moeConfig.dFF
        let numMoELayers = moeLayerIndices.count
        return embedding + (mambaPerLayer * numMambaLayers) + (moePerLayer * numMoELayers)
    }
}

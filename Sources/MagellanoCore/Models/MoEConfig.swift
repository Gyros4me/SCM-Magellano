// Sources/MagellanoCore/Models/MoEConfig.swift
import Foundation

public struct MoEConfig: Codable, Sendable {
    public let dModel: Int           // Hidden dimension (2048)
    public let dFF: Int              // Expert FFN dimension (4 * dModel = 8192)
    public let numExperts: Int       // Number of experts (8)
    public let topK: Int             // Experts activated per token (2)
    public let capacityFactor: Float // Load balancing buffer (1.25)
    public let auxLossWeight: Float  // Load balancing loss weight (0.01)
    
    // Derived
    public var expertsActive: Int { topK }
    public var expertCapacity: Int {
        Int(ceil(Float(dModel) * capacityFactor / Float(numExperts)))
    }
    
    public init(
        dModel: Int = 2048,
        dFF: Int = 8192,
        numExperts: Int = 8,
        topK: Int = 2,
        capacityFactor: Float = 1.25,
        auxLossWeight: Float = 0.01
    ) {
        self.dModel = dModel
        self.dFF = dFF
        self.numExperts = numExperts
        self.topK = topK
        self.capacityFactor = capacityFactor
        self.auxLossWeight = auxLossWeight
    }
    
    // 800M params preset (4 MoE layers)
    public static let magellano800M = MoEConfig(
        dModel: 2048,
        dFF: 8192,
        numExperts: 8,
        topK: 2
    )
    
    // 1B params preset
    public static let magellano1B = MoEConfig(
        dModel: 2304,
        dFF: 9216,
        numExperts: 8,
        topK: 2
    )
}

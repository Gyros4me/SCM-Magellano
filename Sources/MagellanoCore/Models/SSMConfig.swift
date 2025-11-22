// Sources/MagellanoCore/Models/SSMConfig.swift

import Foundation

public struct SSMConfig: Codable, Sendable {
    public let dModel: Int          // Model dimension (2048)
    public let dState: Int          // SSM state dimension (16)
    public let dConv: Int           // Conv1d kernel size (4)
    public let expandFactor: Int    // Expansion factor (2)
    public let dtRank: Int          // Δ projection rank (dModel/16)
    public let numLayers: Int       // Number of Mamba layers (24)
    
    // Derived
    public var dInner: Int { dModel * expandFactor }
    
    public init(
        dModel: Int = 2048,
        dState: Int = 16,
        dConv: Int = 4,
        expandFactor: Int = 2,
        numLayers: Int = 24
    ) {
        self.dModel = dModel
        self.dState = dState
        self.dConv = dConv
        self.expandFactor = expandFactor
        self.dtRank = max(dModel / 16, 1)
        self.numLayers = numLayers
    }
    
    // 800M params preset
    public static let magellano800M = SSMConfig(
        dModel: 2048,
        dState: 16,
        dConv: 4,
        expandFactor: 2,
        numLayers: 24
    )
    
    // 1B params preset
    public static let magellano1B = SSMConfig(
        dModel: 2304,
        dState: 16,
        dConv: 4,
        expandFactor: 2,
        numLayers: 28
    )
}

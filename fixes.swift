// 1. Float16.floatValue fix
var floatValue: Float {
    // Already implemented in struct
}

// 2. Add Sendable conformance
public struct CheckpointConfig: Sendable {
    let saveEveryN: Int
    let recomputeLayers: Bool
    public static let aggressive = CheckpointConfig(saveEveryN: 4, recomputeLayers: true)
    public static let balanced = CheckpointConfig(saveEveryN: 2, recomputeLayers: false)
}

// 3. TargetModule Sendable
public enum TargetModule: String, Codable, Sendable {
    case qProj, kProj, vProj, outProj
    case mambaInProj, mambaXProj, mambaOutProj
    case moeGate, moeExperts
}

// 4. Remove duplicate add() - è già in Tensor.swift

// 5. Change .loraWeights to .modelWeights in LoRALayer

// 6. In MambaMoEModel.swift: change private to internal
internal func embedTokens(_ tokenIds: [Int]) -> Tensor?
internal var layers: [any ModelLayer] = []
internal func projectToVocab(_ hidden: Tensor) -> Tensor?

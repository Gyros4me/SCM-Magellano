// Sources/MagellanoCore/Memory/MemorySnapshot.swift

import Foundation

public struct MemorySnapshot: Codable, Sendable {
    public let timestamp: Date
    public let totalUsedBytes: Int
    public let categoryBreakdown: [MemoryCategory: Int]
    
    // CORREZIONE: Sendable richiesto per actor isolation
    public enum MemoryCategory: String, Codable, CaseIterable, Sendable {
        case modelWeights, activations, optimizerStates, gradients, temporary, unknown
    }
    
    public init(timestamp: Date, totalUsedBytes: Int, categoryBreakdown: [MemoryCategory: Int]) {
        self.timestamp = timestamp
        self.totalUsedBytes = totalUsedBytes
        self.categoryBreakdown = categoryBreakdown
    }
}

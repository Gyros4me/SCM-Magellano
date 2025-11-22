// Sources/MagellanoCore/Logging/CheckpointData.swift

import Foundation

public struct CheckpointData: Codable {
    public let name: String
    public let timestamp: Date
    public let activityStack: [String]
    public let memorySnapshot: LogEntry.MemorySnapshot
    public let metadata: [String: AnyCodable]
    
    public init(
        name: String,
        timestamp: Date,
        activityStack: [String],
        memorySnapshot: LogEntry.MemorySnapshot,
        metadata: [String: AnyCodable]
    ) {
        self.name = name
        self.timestamp = timestamp
        self.activityStack = activityStack
        self.memorySnapshot = memorySnapshot
        self.metadata = metadata
    }
}

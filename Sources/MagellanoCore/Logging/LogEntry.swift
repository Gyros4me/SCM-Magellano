// Sources/MagellanoCore/Logging/LogEntry.swift

import Foundation

public struct LogEntry: Codable, Sendable {
    public let timestamp: Date
    public let level: LogLevel
    public let activity: String
    public let checkpoint: String?
    public let message: String
    public let memorySnapshot: MemorySnapshot?
    public let stackTrace: [String]?
    public let performanceMetrics: PerformanceMetrics?
    public let errorDetails: ErrorDetails?
    
    // MARK: - Nested Types (corretto per Sendable)
    public struct MemorySnapshot: Codable, Sendable {
        public let usedBytes: Int
        public let peakBytes: Int
        public let gpuBytes: Int
        public let cpuBytes: Int
        
        public init(used: Int, peak: Int, gpu: Int, cpu: Int) {
            self.usedBytes = used
            self.peakBytes = peak
            self.gpuBytes = gpu
            self.cpuBytes = cpu
        }
    }
    
    public struct PerformanceMetrics: Codable, Sendable {
        public let durationMs: Double
        public let startTime: Date
        public let endTime: Date
        public let throughput: Double?
        
        public init(durationMs: Double, startTime: Date, endTime: Date, throughput: Double? = nil) {
            self.durationMs = durationMs
            self.startTime = startTime
            self.endTime = endTime
            self.throughput = throughput
        }
    }
    
    public struct ErrorDetails: Codable, Sendable {
        public let errorType: String
        public let errorMessage: String
        public let recoverable: Bool
        public let suggestedAction: String?
        
        public init(type: String, message: String, recoverable: Bool, action: String? = nil) {
            self.errorType = type
            self.errorMessage = message
            self.recoverable = recoverable
            self.suggestedAction = action
        }
    }
    
    // MARK: - Initializers
    public init(
        level: LogLevel,
        activity: String,
        checkpoint: String? = nil,
        message: String,
        memorySnapshot: MemorySnapshot? = nil,
        stackTrace: [String]? = nil,
        performanceMetrics: PerformanceMetrics? = nil,
        errorDetails: ErrorDetails? = nil
    ) {
        self.timestamp = Date()
        self.level = level
        self.activity = activity
        self.checkpoint = checkpoint
        self.message = message
        self.memorySnapshot = memorySnapshot
        self.stackTrace = stackTrace ?? Thread.callStackSymbols
        self.performanceMetrics = performanceMetrics
        self.errorDetails = errorDetails
    }
}

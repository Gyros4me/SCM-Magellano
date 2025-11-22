// Sources/MagellanoCore/Logging/StructuredLogger.swift

import Foundation
import OSLog

public actor StructuredLogger {
    public static let shared = StructuredLogger()
    
    private let logSubsystem = "com.tim.magellano"
    private let logCategory = "Training"
    private let osLogger: Logger
    
    private let logFileURL: URL
    private let checkpointFileURL: URL
    
    private var logBuffer: [LogEntry] = []
    private let flushThreshold = 10
    
    private init() {
        let logsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Magellano")
        try? FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
        
        let timestamp = ISO8601DateFormatter().string(from: Date())
        self.logFileURL = logsDir.appendingPathComponent("magellano_\(timestamp).jsonl")
        self.checkpointFileURL = logsDir.appendingPathComponent("checkpoints.json")
        
        self.osLogger = Logger(subsystem: logSubsystem, category: logCategory)
        
        print("📝 Logger inizializzato. Log file: \(logFileURL.path)")
        print("🗂️  Checkpoint file: \(checkpointFileURL.path)")
    }
    
    // MARK: - Public API
    public func trace(_ activity: String, _ message: String) async {
        await log(level: .trace, activity: activity, message: message)
    }
    
    public func debug(_ activity: String, _ message: String) async {
        await log(level: .debug, activity: activity, message: message)
    }
    
    public func info(_ activity: String, checkpoint: String, _ message: String) async {
        let entry = LogEntry(
            level: .info,
            activity: activity,
            checkpoint: checkpoint,
            message: message,
            memorySnapshot: await captureMemory()
        )
        logAndSave(entry)
    }
    
    public func warning(_ activity: String, _ message: String, error: Error? = nil) async {
        let errorDetails = error.map { e in
            LogEntry.ErrorDetails(
                type: String(describing: type(of: e)),
                message: e.localizedDescription,
                recoverable: true,
                action: "Riprova o ignora"
            )
        }
        let entry = LogEntry(
            level: .warning,
            activity: activity,
            message: message,
            memorySnapshot: await captureMemory(),
            errorDetails: errorDetails
        )
        logAndSave(entry)
    }
    
    public func error(_ activity: String, _ message: String, error: Error) async {
        let errorDetails = LogEntry.ErrorDetails(
            type: String(describing: type(of: error)),
            message: error.localizedDescription,
            recoverable: false,
            action: "Termina e analizza log"
        )
        let entry = LogEntry(
            level: .error,
            activity: activity,
            message: message,
            memorySnapshot: await captureMemory(),
            errorDetails: errorDetails
        )
        logAndSave(entry)
    }
    
    public func critical(_ activity: String, _ message: String, error: Error) async {
        let errorDetails = LogEntry.ErrorDetails(
            type: String(describing: type(of: error)),
            message: error.localizedDescription,
            recoverable: false,
            action: "Salva checkpoint e arresta"
        )
        let entry = LogEntry(
            level: .critical,
            activity: activity,
            message: message,
            memorySnapshot: await captureMemory(),
            errorDetails: errorDetails
        )
        logAndSave(entry)
        await flush()
    }
    
    // MARK: - Performance Logging
    public func measure<T>(
        _ activity: String,
        checkpoint: String,
        expectedThroughput: Double? = nil,
        _ block: () async throws -> T
    ) async rethrows -> T {
        let start = Date()
        let startMem = await captureMemory()
        
        let result = try await block()
        
        let end = Date()
        let durationMs = end.timeIntervalSince(start) * 1000
        let endMem = await captureMemory()
        
        let perfMetrics = LogEntry.PerformanceMetrics(
            durationMs: durationMs,
            startTime: start,
            endTime: Date(),
            throughput: calculateThroughput(durationMs: durationMs, memDelta: endMem.usedBytes - startMem.usedBytes)
        )
        
        let entry = LogEntry(
            level: .info,
            activity: activity,
            checkpoint: checkpoint,
            message: "Completed in \(String(format: "%.2f", durationMs))ms",
            memorySnapshot: endMem,
            performanceMetrics: perfMetrics
        )
        logAndSave(entry)
        
        return result
    }
    
    // MARK: - Checkpoint Management
    public func saveCheckpoint(name: String, metadata: [String: AnyCodable]) async {
        let checkpointData = CheckpointData(
            name: name,
            timestamp: Date(),
            activityStack: [],
            memorySnapshot: await captureMemory(),
            metadata: metadata
        )
        
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            
            var checkpoints = try await loadCheckpoints()
            checkpoints.append(checkpointData)
            
            let data = try encoder.encode(checkpoints)
            try data.write(to: checkpointFileURL, options: .atomic)
            
            await info("Checkpoint", checkpoint: name, "Salvato su disco")
        } catch {
            await self.error("Checkpoint", "Failed to save checkpoint", error: error)  // FIX: self.error
        }
    }
    
    public func loadLastCheckpoint() async -> CheckpointData? {
        do {
            let checkpoints = try await loadCheckpoints()
            return checkpoints.last
        } catch {
            await self.error("Checkpoint", "Failed to load checkpoint", error: error)  // FIX: self.error
            return nil
        }
    }
    
    private func loadCheckpoints() async throws -> [CheckpointData] {
        guard FileManager.default.fileExists(atPath: checkpointFileURL.path) else {
            return []
        }
        
        let data = try Data(contentsOf: checkpointFileURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode([CheckpointData].self, from: data)
    }
    
    // MARK: - Private Helpers
    private func log(level: LogLevel, activity: String, message: String) async {
        let entry = LogEntry(
            level: level,
            activity: activity,
            message: message,
            memorySnapshot: await captureMemory()
        )
        
        osLogger.log(level: level.osLogType, "\(activity): \(message)")
        
        logBuffer.append(entry)
        
        if logBuffer.count >= flushThreshold {
            await flush()
        }
    }
    
    private func logAndSave(_ entry: LogEntry) {
        logBuffer.append(entry)
        
        if entry.level == .info || entry.level == .error || entry.level == .critical {
            Task { await flush() }
        }
    }
    
    private func flush() async {
        guard !logBuffer.isEmpty else { return }
        
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = .withoutEscapingSlashes
            
            let data = try encoder.encode(logBuffer)
            
            if let fileHandle = try? FileHandle(forWritingTo: logFileURL) {
                fileHandle.seekToEndOfFile()
                fileHandle.write(data)
                fileHandle.write("\n".data(using: .utf8)!)
                try fileHandle.close()
            } else {
                try data.write(to: logFileURL)
            }
            
            logBuffer.removeAll()
        } catch {
            print("❌ CRITICAL: Impossibile scrivere log: \(error)")
            fputs("LOGGING FAILURE: \(error)\n", stderr)
        }
    }
    
    private func captureMemory() async -> LogEntry.MemorySnapshot {
        let aggregator = MemoryAggregator.shared
        return LogEntry.MemorySnapshot(
            used: await aggregator.currentMemory,
            peak: await aggregator.peak,
            gpu: 0,
            cpu: 0
        )
    }
    
    private func calculateThroughput(durationMs: Double, memDelta: Int) -> Double {
        0.0
    }
}

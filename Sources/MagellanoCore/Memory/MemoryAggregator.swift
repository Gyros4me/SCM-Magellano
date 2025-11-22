// Sources/MagellanoCore/Memory/MemoryAggregator.swift

import Foundation
import Metal

public actor MemoryAggregator {
    public static let shared = MemoryAggregator()
    
    private var snapshots: [MemorySnapshot] = []
    private let collector: MemoryCollector
    private let collectionInterval: Double
    
    // Init pubblico per permettere istanze multiple se necessario
    public init(device: MTLDevice? = nil, collectionInterval: Double = 0.5) {
        self.collector = MemoryCollector(device: device ?? MTLCreateSystemDefaultDevice()!)
        self.collectionInterval = collectionInterval
        Task {
            await startPeriodicCollection()
        }
    }
    
    private func startPeriodicCollection() async {
        while !Task.isCancelled {
            let snapshot = await collector.collectSnapshot()
            addSnapshot(snapshot)
            try? await Task.sleep(nanoseconds: UInt64(collectionInterval * 1_000_000_000))
        }
    }
    
    public func captureSnapshot(tag: String) async {
        let snapshot = await collector.collectSnapshot(tag: tag)
        addSnapshot(snapshot)
    }
    
    private func addSnapshot(_ snapshot: MemorySnapshot) {
        snapshots.append(snapshot)
        if snapshots.count > 1000 {
            snapshots.removeFirst(100)
        }
    }
    
    public var currentMemory: Int {
        snapshots.last?.totalUsedBytes ?? 0
    }
    
    public var peak: Int {
        snapshots.map { $0.totalUsedBytes }.max() ?? 0
    }
    
    public func generateReport(duration: TimeInterval) -> MemoryReport {
        let relevant = snapshots.filter { $0.timestamp.timeIntervalSinceNow > -duration }
        
        return MemoryReport(
            totalUsedBytes: currentMemory,
            averageBytes: Int(relevant.map { $0.totalUsedBytes }.average() ?? 0),
            peakBytes: peak,
            growthRate: calculateGrowthRate(),
            byCategory: calculateCategoryBreakdown()
        )
    }
    
    private func calculateGrowthRate() -> Double {
        guard snapshots.count >= 2 else { return 0 }
        let first = snapshots.first!.totalUsedBytes
        let last = snapshots.last!.totalUsedBytes
        let timeDelta = snapshots.last!.timestamp.timeIntervalSince(snapshots.first!.timestamp)
        guard timeDelta > 0 else { return 0 }
        return Double(last - first) / timeDelta
    }
    
    private func calculateCategoryBreakdown() -> [String: MemoryReport.CategoryStats] {
        var breakdown: [MemorySnapshot.MemoryCategory: [Int]] = [:]
        
        for snapshot in snapshots {
            for (category, bytes) in snapshot.categoryBreakdown {
                breakdown[category, default: []].append(bytes)
            }
        }
        
        var result: [String: MemoryReport.CategoryStats] = [:]
        for (category, bytes) in breakdown {
            result[category.rawValue] = MemoryReport.CategoryStats(
                currentBytes: bytes.last ?? 0,
                averageBytes: Int(bytes.average() ?? 0),
                peakBytes: bytes.max() ?? 0
            )
        }
        
        return result
    }
}

// FIX: Aggiungi Sendable
public struct MemoryReport: Sendable {
    public let totalUsedBytes: Int
    public let averageBytes: Int
    public let peakBytes: Int
    public let growthRate: Double
    public let byCategory: [String: CategoryStats]
    
    // FIX: Aggiungi Sendable anche al nested struct
    public struct CategoryStats: Sendable {
        public let currentBytes: Int
        public let averageBytes: Int
        public let peakBytes: Int
    }
}

extension Sequence where Element == Int {
    func average() -> Double? {
        let array = Array(self)
        guard !array.isEmpty else { return nil }
        return Double(array.reduce(0, +)) / Double(array.count)
    }
}

// Sources/MagellanoCore/Memory/MemoryTracker.swift

import Foundation
import Metal
import os.log

public actor MemoryTracker {
    public static let shared = MemoryTracker()
    
    private var allocations: [ObjectIdentifier: (category: MemorySnapshot.MemoryCategory, size: Int, label: String)] = [:]
    private var categoryTotals: [MemorySnapshot.MemoryCategory: UInt64] = [:]
    private var categoryPeaks: [MemorySnapshot.MemoryCategory: UInt64] = [:]
    private let logger = Logger(subsystem: "com.scm.magellano", category: "MemoryTracker")
    
    private init() {
        logger.info("MemoryTracker initialized")
    }
    
    public func registerAllocation(_ tracked: TrackedBuffer) {
        let id = ObjectIdentifier(tracked.buffer)
        allocations[id] = (tracked.category, tracked.size, tracked.label)
        
        let currentTotal = categoryTotals[tracked.category, default: 0]
        let newTotal = currentTotal + UInt64(tracked.size)
        categoryTotals[tracked.category] = newTotal
        
        if newTotal > categoryPeaks[tracked.category, default: 0] {
            categoryPeaks[tracked.category] = newTotal
            logger.debug("Peak \(tracked.category.rawValue): \(newTotal / 1_048_576) MB")
        }
    }
    
    public func unregisterAllocation(id: ObjectIdentifier, category: MemorySnapshot.MemoryCategory, size: Int) {
        allocations.removeValue(forKey: id)
        let currentTotal = categoryTotals[category, default: 0]
        categoryTotals[category] = currentTotal - UInt64(size)
    }
    
    public func getCurrentByCategory() -> [MemorySnapshot.MemoryCategory: UInt64] {
        return categoryTotals
    }
    
    public func getPeaksByCategory() -> [MemorySnapshot.MemoryCategory: UInt64] {
        return categoryPeaks
    }
    
    public func resetPeaks() {
        categoryPeaks.removeAll()
        logger.info("Peaks reset")
    }
}

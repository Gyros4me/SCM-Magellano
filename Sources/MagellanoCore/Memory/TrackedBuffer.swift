// Sources/MagellanoCore/Memory/TrackedBuffer.swift

import Metal
import Foundation

public final class TrackedBuffer: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let category: MemorySnapshot.MemoryCategory
    public let size: Int
    public let label: String
    private let bufferId: ObjectIdentifier
    
    public init(buffer: MTLBuffer, category: MemorySnapshot.MemoryCategory, label: String = "") {
        self.buffer = buffer
        self.category = category
        self.size = buffer.length
        self.label = label
        self.bufferId = ObjectIdentifier(buffer)
        
        Task { await MemoryTracker.shared.registerAllocation(self) }
    }
    
    deinit {
        let id = bufferId
        let cat = category
        let sz = size
        Task { await MemoryTracker.shared.unregisterAllocation(id: id, category: cat, size: sz) }
    }
}

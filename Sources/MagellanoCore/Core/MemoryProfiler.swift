import Foundation
import Metal

/// Actor per tracciare l'uso della memoria durante l'esecuzione
public actor MemoryProfiler {
    public static let shared = MemoryProfiler()
    
    private var peakUsage: Int = 0
    private var currentUsage: Int = 0
    
    private init() {
        // Rimossa la chiamata problematica a updateMemoryUsage()
        // L'inizializzazione a 0 è sufficiente
    }
    
    public func trackAllocation(bytes: Int) {
        currentUsage += bytes
        if currentUsage > peakUsage {
            peakUsage = currentUsage
        }
    }
    
    public func trackDeallocation(bytes: Int) {
        currentUsage -= bytes
    }
    
    public func getCurrentUsageMB() -> Int {
        updateMemoryUsage()
        return currentUsage / (1024 * 1024)
    }
    
    public func getPeakUsageMB() -> Int {
        updateMemoryUsage()
        return max(peakUsage, currentUsage) / (1024 * 1024)
    }
    
    public func reset() {
        peakUsage = 0
        currentUsage = 0
    }
    
    private func updateMemoryUsage() {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            let usage = Int(info.resident_size)
            currentUsage = usage
            if usage > peakUsage {
                peakUsage = usage
            }
        }
    }
}

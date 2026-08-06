import Darwin
import Foundation

/// The process's real memory footprint, as macOS accounts for it.
///
/// `phys_footprint` is the measurement that counts: it is the one the system uses to decide
/// memory pressure, and it excludes file-cache pages the kernel can reclaim at any time.
/// Our writes go through `pwrite`, not through mappings, so the gigabytes passing through
/// are not counted against it — which is precisely what the project's invariant claims.
public enum MemoryFootprint {

    /// Total resident memory, mapped file pages **included**.
    ///
    /// Distinguishing the two is essential to being honest: `phys_footprint` excludes clean
    /// file-backed pages, which the kernel can reclaim at any time. But `resident.bin` is
    /// re-read on every token — its pages really are in RAM, they are simply not charged to
    /// the process. Reporting only `phys_footprint` would give a flattering, false figure.
    public static func resident() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { raw in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), raw, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
    }

    public static func current() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { raw in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), raw, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }

    /// Tracks the observed maximum, sampled at the points where we make progress.
    public final class Peak: @unchecked Sendable {
        private var maximum = 0
        private let lock = NSLock()

        private var maximumResident = 0

        public init() {}

        public func sample() {
            let footprint = MemoryFootprint.current()
            let resident = MemoryFootprint.resident()
            lock.lock()
            maximum = max(maximum, footprint)
            maximumResident = max(maximumResident, resident)
            lock.unlock()
        }

        public var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return maximum
        }

        public var residentValue: Int {
            lock.lock()
            defer { lock.unlock() }
            return maximumResident
        }
    }
}

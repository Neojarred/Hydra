import Darwin
import Foundation

/// Empreinte mémoire réelle du processus, telle que macOS la comptabilise.
///
/// `phys_footprint` est la mesure qui compte : c'est celle que le système utilise pour
/// décider de la pression mémoire, et elle exclut les pages de cache de fichiers que le
/// noyau peut reprendre à tout moment. Nos écritures passent par `pwrite`, pas par des
/// mappages, donc les gigaoctets qui transitent n'y sont pas comptés — ce qui est
/// précisément ce que l'invariant du projet affirme.
public enum MemoryFootprint {

    /// Mémoire résidente totale, pages de fichiers mappés **comprises**.
    ///
    /// Distinguer les deux est indispensable pour être honnête : `phys_footprint` exclut
    /// les pages propres adossées à un fichier, que le noyau peut reprendre à tout moment.
    /// Or `resident.bin` est relu à chaque token — ses pages sont bien en RAM, elles ne
    /// sont simplement pas imputées au processus. Ne rapporter que `phys_footprint`
    /// donnerait un chiffre flatteur et faux.
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

    /// Suit le maximum observé, échantillonné aux moments où on avance.
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

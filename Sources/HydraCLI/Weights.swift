import Foundation
import HydraCore
import HydraFormat
import HydraMetal
import Metal

/// Reads resident tensors back the way the kernels see them.
///
/// The distinction this exists to draw: a weight file can be byte-perfect and still be read
/// wrong. `resident.bin` holds every layer's tensors end to end, and a placement that is off by
/// one tensor gives kernels values that are finite, plausible, and not the ones the checkpoint
/// stored. Scanning the file finds corruption; this finds misplacement.
enum Weights {

    static func run(model: any ModelDescriptor, root: URL) throws {
        let context = try MetalContext()
        let mapping = try ModelMapping(root: root, model: model, device: context.device)

        print("resident tensors, as the runtime resolves them\n")
        var worst: (name: String, magnitude: Float)?
        var suspicious = 0

        for (name, byteCount) in model.residentTensors {
            let placement = try mapping.residentTensor(name)
            let count = byteCount / 2
            var minimum = Float.greatestFiniteMagnitude
            var maximum = -Float.greatestFiniteMagnitude
            var nonFinite = 0

            mapping.resident.withBytes { raw in
                for i in 0..<count {
                    let bits = raw.loadUnaligned(
                        fromByteOffset: placement.offset + i * 2, as: UInt16.self)
                    let value = BF16.toFloat(UInt16(littleEndian: bits))
                    guard value.isFinite else { nonFinite += 1; return }
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                }
            }

            let magnitude = max(abs(minimum), abs(maximum))
            if worst == nil || magnitude > worst!.magnitude {
                worst = (name, magnitude)
            }

            // A norm weight, a scale or a scalar lives near 1. Anything past a thousand is
            // either a misplacement or a tensor this runtime has misunderstood, and either way
            // it is what turns a finite forward pass into infinities.
            let isScale = name.hasSuffix("layer_scalar")
                || name.hasSuffix("norm.weight")
                || name.hasSuffix("router.scale")
                || name.hasSuffix("per_expert_scale")
            if nonFinite > 0 || (isScale && magnitude > 1000) {
                suspicious += 1
                print(String(
                    format: "  ✘ %-64@ min %12.4f  max %12.4f%@",
                    name as NSString, minimum, maximum,
                    nonFinite > 0 ? "  \(nonFinite) non-finite" : ""))
            }
        }

        // The per-layer scalar, printed for every layer: it multiplies the whole hidden state
        // at the end of the layer, so one wrong value takes the entire model with it.
        print("\nlayer_scalar per layer")
        for layer in 0..<model.layerCount {
            let name = "model.language_model.layers.\(layer).layer_scalar"
            guard let placement = try? mapping.residentTensor(name) else { continue }
            let value = mapping.resident.withBytes { raw -> Float in
                let bits = raw.loadUnaligned(fromByteOffset: placement.offset, as: UInt16.self)
                return BF16.toFloat(UInt16(littleEndian: bits))
            }
            print(String(format: "  layer %2d  %.6f", layer, value))
        }

        if suspicious == 0 { print("\n  ✔ no resident tensor is out of range") }
        if let worst { print(String(format: "largest magnitude: %@ at %.4f", worst.name, worst.magnitude)) }
    }
}

import Foundation

/// Qwen 3.6's multimodal rotary: three position axes instead of one.
///
/// A text-only model gives every token one position. Qwen gives every token **three**, a time, a
/// row and a column, and splits the rotary's frequencies between them. For text all three are
/// the same number, which is why the text path has been correct without any of this existing:
/// mRoPE with `t == h == w` is ordinary RoPE, exactly. Only an image makes them differ.
///
/// Both halves are transcribed from `Qwen3_5MoeTextRotaryEmbedding.apply_interleaved_mrope` and
/// `get_rope_index`.
public enum Qwen35MRoPE {

    /// The three axes, in the order the reference stacks them.
    public enum Axis: Int, Sendable, CaseIterable {
        case time = 0, height = 1, width = 2
    }

    /// Which axis a rotary frequency turns with.
    ///
    /// **`j % 3`**, and that is the whole rule. The config states it as
    /// `mrope_section: [11, 11, 10]`, which reads like three independent block sizes and is not:
    /// it is the *consequence* of interleaving three axes over 32 frequencies. The reference
    /// writes it as a strided assignment, `freqs[1]` into `slice(1, 33, 3)` and `freqs[2]` into
    /// `slice(2, 30, 3)` over a base of `freqs[0]`, which lands on exactly this.
    ///
    /// Reading `[11, 11, 10]` as "the first 11 frequencies are time, the next 11 height, the
    /// last 10 width" is the natural misreading, produces the right *counts*, and rotates
    /// entirely the wrong channels. `mrope_interleaved: true` in the config is the warning.
    public static func axis(forFrequency index: Int) -> Axis {
        Axis(rawValue: index % 3) ?? .time
    }

    /// How many frequencies each axis owns, which must reproduce the published section sizes.
    public static func sectionSizes(frequencies: Int) -> [Int] {
        var counts = [0, 0, 0]
        for j in 0..<frequencies { counts[axis(forFrequency: j).rawValue] += 1 }
        return counts
    }

    /// A run of the prompt: either ordinary tokens, or one image's merged grid.
    public enum Segment: Sendable, Equatable {
        case text(count: Int)
        /// The grid **after** the vision merge, so `height` and `width` are already divided by
        /// the spatial merge size and the product is the image's token count.
        case image(frames: Int, height: Int, width: Int)

        public var tokenCount: Int {
            switch self {
            case .text(let count): return count
            case let .image(frames, height, width): return frames * height * width
            }
        }
    }

    /// The three position ids of every token in the prompt.
    ///
    /// The subtlety is what an image does to the counter that follows it. A text run of `n`
    /// tokens advances the position by `n`, as anything would. An image advances it by
    /// **`max(height, width)`**, not by its token count: its tokens share a small square of
    /// position space rather than consuming a line of it, so a 64x64 image occupies 4096 tokens
    /// and 64 positions. Advancing by the token count instead would push everything after an
    /// image thousands of positions further along, which stays finite and quietly wrecks every
    /// distance the model has learned.
    public static func positionIDs(for segments: [Segment]) -> (t: [Int], h: [Int], w: [Int]) {
        var t: [Int] = [], h: [Int] = [], w: [Int] = []
        let total = segments.reduce(0) { $0 + $1.tokenCount }
        t.reserveCapacity(total); h.reserveCapacity(total); w.reserveCapacity(total)

        var position = 0
        for segment in segments {
            switch segment {
            case .text(let count):
                for i in 0..<count {
                    t.append(position + i); h.append(position + i); w.append(position + i)
                }
                position += count

            case let .image(frames, height, width):
                // Row-major over (frame, row, column), which is the order the merged tokens
                // leave the tower in.
                for frame in 0..<frames {
                    for row in 0..<height {
                        for column in 0..<width {
                            t.append(position + frame)
                            h.append(position + row)
                            w.append(position + column)
                        }
                    }
                }
                position += max(height, width)
            }
        }
        return (t, h, w)
    }

    /// The rotary angle of one frequency at one token, given its three positions.
    public static func angle(
        frequency index: Int, inverseFrequency: Double, t: Int, h: Int, w: Int
    ) -> Double {
        let position: Int
        switch axis(forFrequency: index) {
        case .time: position = t
        case .height: position = h
        case .width: position = w
        }
        return Double(position) * inverseFrequency
    }
}

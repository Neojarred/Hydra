/// Proposes candidate tokens by searching what has already been written.
///
/// Speculative decoding needs a **cheap** source of drafts: the gain comes from verifying
/// several tokens in one pass, so the draft must cost markedly less than the verification.
/// A second model does not fit here, the 20B is only 4.5 times cheaper than the 120B, and
/// it would need to be ten.
///
/// This source costs nothing: we look for the last occurrence of the last `n` tokens in the
/// history, and propose what followed. It is silent in open conversation and very effective
/// as soon as the answer repeats the context, summarizing, rewriting, code, questions
/// about an attached document.
///
/// A wrong draft costs only the verification, which would have happened anyway; it can
/// therefore never change the output, only the time taken to get it.
public struct NGramDrafter: Sendable {

    /// Pattern lengths tried, from the most specific to the most permissive.
    ///
    /// A long pattern is rarely wrong but rarely finds anything; a short one finds often and is
    /// often wrong. We take the first match starting from the longest: the best compromise with
    /// no notable search cost.
    public let patternLengths: [Int]
    /// Number of tokens proposed per attempt.
    public let draftLength: Int

    public init(patternLengths: [Int] = [3, 2], draftLength: Int = 4) {
        self.patternLengths = patternLengths
        self.draftLength = draftLength
    }

    /// Tokens proposed to continue `history`, or empty if nothing matches.
    ///
    /// The search starts from the end: the most recent repetition is the most likely.
    public func propose(history: [Int]) -> [Int] {
        guard !history.isEmpty else { return [] }

        for length in patternLengths where history.count > length {
            let pattern = Array(history.suffix(length))
            // We stop short of the end: the final pattern is the one we are trying to extend,
            // not a usable precedent.
            var start = history.count - length - 1
            while start >= 0 {
                if Array(history[start..<(start + length)]) == pattern {
                    let from = start + length
                    guard from < history.count else { break }
                    let upper = min(from + draftLength, history.count)
                    let draft = Array(history[from..<upper])
                    if !draft.isEmpty { return draft }
                }
                start -= 1
            }
        }
        return []
    }
}

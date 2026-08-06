/// The length of the longest common prefix of two token sequences.
///
/// Used to decide what a KV cache may keep from one conversation turn to the next. It
/// lives here rather than in the engine so the runtime can be tested against it without
/// depending on the application.
public func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
    var index = 0
    let limit = min(a.count, b.count)
    while index < limit && a[index] == b[index] { index += 1 }
    return index
}

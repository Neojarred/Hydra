import Foundation

/// An HTTP client that **consumes a response as it streams**, never accumulating it.
///
/// The repacker's first version split every range into 4 MiB sub-requests to bound the
/// heap. Measured on the real repository, that choice was expensive: one 64 MiB request
/// reaches 33.5 MB/s, eight 4 MiB requests in series fall to 5.2 MB/s. Hugging Face answers
/// with a 302 to a signed CDN, and the signature is tied to the range requested, the
/// resolved URL is therefore not reusable, and each request pays redirect and TLS again.
///
/// The right solution gives both: **a single request for a large range**, whose response is
/// handed back in pieces as it arrives. `URLSession` delivers blocks of a few tens of KiB,
/// each written then released before the next. The heap stays bounded by one block,
/// whatever the range's size.
final class StreamingHTTPClient: NSObject, URLSessionDataDelegate, @unchecked Sendable {

    private var session: URLSession!
    private let lock = NSLock()

    private final class Pending {
        let sink: (Data) throws -> Void
        var continuation: CheckedContinuation<Void, Error>?
        var failure: Error?
        var statusCode: Int?
        var finished = false

        init(sink: @escaping (Data) throws -> Void) { self.sink = sink }
    }

    private var pending: [Int: Pending] = [:]

    enum StreamError: Error, CustomStringConvertible {
        case badStatus(Int, url: String)
        case rangeNotHonored(url: String)
        case shortStream(expected: Int, got: Int, url: String)

        var description: String {
            switch self {
            case let .badStatus(code, url): return "HTTP \(code) sur \(url)"
            case .rangeNotHonored(let url):
                return "the server ignored Range on \(url), unbounded read refused"
            case let .shortStream(e, g, url):
                return "truncated stream on \(url): \(g) bytes received, \(e) expected"
            }
        }
    }

    init(maximumConnectionsPerHost: Int = 4, timeout: TimeInterval = 120) {
        super.init()
        let configuration = URLSessionConfiguration.default
        configuration.httpMaximumConnectionsPerHost = maximumConnectionsPerHost
        configuration.timeoutIntervalForRequest = timeout
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        // Nothing passing through here should be cached: these are gigabytes of weights
        // already being written to their final location.
        configuration.urlCache = nil
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    deinit { session?.finishTasksAndInvalidate() }

    /// Streams `range`, calling `sink` for each block received, in order.
    /// `sink` is invoked from the session's delegate queue, hence serialized.
    func stream(
        url: URL,
        range: Range<Int>,
        userAgent: String,
        sink: @escaping (Data) throws -> Void
    ) async throws {
        let received = Counter()
        var request = URLRequest(url: url)
        request.setValue(
            "bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")

        let task = session.dataTask(with: request)
        let record = Pending(sink: { data in
            received.add(data.count)
            try sink(data)
        })
        lock.withLock { pending[task.taskIdentifier] = record }

        enum Start { case wait, alreadyFinished(Error?) }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, Error>) in
                let start: Start = lock.withLock {
                    // The task may have finished before the continuation was installed.
                    if record.finished { return .alreadyFinished(record.failure) }
                    record.continuation = continuation
                    return .wait
                }
                switch start {
                case .wait:
                    task.resume()
                case .alreadyFinished(let failure):
                    if let failure { continuation.resume(throwing: failure) }
                    else { continuation.resume() }
                }
            }
        } onCancel: {
            task.cancel()
        }

        guard received.value == range.count else {
            throw StreamError.shortStream(
                expected: range.count, got: received.value, url: url.absoluteString)
        }
    }

    // MARK: - URLSessionDataDelegate

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? -1
        lock.lock()
        let record = pending[dataTask.taskIdentifier]
        record?.statusCode = status
        lock.unlock()

        guard status == 206 else {
            let url = response.url?.absoluteString ?? "?"
            lock.lock()
            record?.failure = status == 200
                ? StreamError.rangeNotHonored(url: url)
                : StreamError.badStatus(status, url: url)
            lock.unlock()
            completionHandler(.cancel)
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        let record = pending[dataTask.taskIdentifier]
        lock.unlock()
        guard let record, record.failure == nil else { return }
        do {
            try record.sink(data)
        } catch {
            lock.lock()
            record.failure = error
            lock.unlock()
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        guard let record = pending.removeValue(forKey: task.taskIdentifier) else {
            lock.unlock()
            return
        }
        record.finished = true
        // An application error (status, write) takes precedence over the cancellation it caused.
        let failure = record.failure ?? error
        let continuation = record.continuation
        record.continuation = nil
        lock.unlock()

        if let continuation {
            if let failure { continuation.resume(throwing: failure) }
            else { continuation.resume() }
        }
    }
}

/// A counter shared between the calling async context and the delegate queue.
public final class Counter: @unchecked Sendable {
    private var storage = 0
    private let lock = NSLock()

    public init() {}

    public func add(_ n: Int) {
        lock.lock()
        storage += n
        lock.unlock()
    }

    public var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

import Foundation

/// Client HTTP qui **consomme une réponse au fil de l'eau**, sans jamais l'accumuler.
///
/// La première version du repacker découpait chaque plage en sous-requêtes de 4 Mio pour
/// borner le tas. Mesuré sur le vrai dépôt, ce choix coûtait cher : une requête de 64 Mio
/// atteint 33,5 Mo/s, huit requêtes de 4 Mio en série tombent à 5,2 Mo/s. Hugging Face
/// répond un 302 vers un CDN signé, et la signature est liée à la plage demandée — l'URL
/// résolue n'est donc pas réutilisable, chaque requête repaie redirection et poignée TLS.
///
/// La bonne solution donne les deux : **une seule requête pour une grande plage**, dont la
/// réponse est remise par morceaux à mesure qu'elle arrive. `URLSession` livre des blocs de
/// quelques dizaines de kio, chacun écrit puis relâché avant le suivant. Le tas reste borné
/// par un bloc, indépendamment de la taille de la plage.
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
                return "le serveur a ignoré Range sur \(url) — lecture non bornée refusée"
            case let .shortStream(e, g, url):
                return "flux tronqué sur \(url) : \(g) octets reçus, \(e) attendus"
            }
        }
    }

    init(maximumConnectionsPerHost: Int = 4, timeout: TimeInterval = 120) {
        super.init()
        let configuration = URLSessionConfiguration.default
        configuration.httpMaximumConnectionsPerHost = maximumConnectionsPerHost
        configuration.timeoutIntervalForRequest = timeout
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        // Rien de ce qui transite ici ne doit être mis en cache : ce sont des gigaoctets
        // de poids qu'on écrit déjà à leur emplacement définitif.
        configuration.urlCache = nil
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    deinit { session?.finishTasksAndInvalidate() }

    /// Diffuse `range` en appelant `sink` pour chaque bloc reçu, dans l'ordre.
    /// `sink` est invoqué depuis la file de délégation de la session, donc sérialisé.
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
                    // La tâche a pu se terminer avant qu'on ait installé la continuation.
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
        // Une erreur applicative (statut, écriture) prime sur l'annulation qu'elle a causée.
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

/// Compteur partagé entre le contexte asynchrone appelant et la file de délégation.
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

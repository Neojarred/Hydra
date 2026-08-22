import Foundation
import Security

/// The search provider's API key, in the keychain.
///
/// Not `UserDefaults`. A key is a bearer credential against a metered account, and a plist in
/// Application Support is readable by anything running as the user and lands in backups in the
/// clear. The keychain is the platform's answer to exactly this and costs forty lines.
///
/// It is also the reason the key belongs to a person rather than to the application. No free
/// tier in this market survives being shared between everyone who downloads a build: a
/// thousand searches a month is one user's comfortable ceiling and two hundred users' first
/// afternoon. Hydra ships the client and the user brings the key.
enum SearchKey {

    private static let service = "app.hydra.search"

    /// One provider's key. A raw value that is stable for the life of the account, since it is
    /// what the stored item is filed under.
    enum Provider: String {
        case tavily
    }

    static func load(_ provider: Provider) -> String? {
        var query = baseQuery(provider)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
            let data = item as? Data,
            let key = String(data: data, encoding: .utf8),
            !key.isEmpty
        else { return nil }
        return key
    }

    /// Stores a key, or removes it when the string is empty.
    ///
    /// Empty means removal rather than storing nothing, because that is what clearing the field
    /// in the interface means and the alternative is a stored empty string that reads as a
    /// configured account and fails on every search with a 401.
    @discardableResult
    static func save(_ key: String, for provider: Provider) -> Bool {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return remove(provider) }

        let query = baseQuery(provider)
        let attributes: [String: Any] = [kSecValueData as String: Data(trimmed.utf8)]
        let updated = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        if updated == errSecSuccess { return true }
        guard updated == errSecItemNotFound else { return false }

        var insert = query
        insert[kSecValueData as String] = Data(trimmed.utf8)
        // Available once the machine has been unlocked, not while it is locked: a background
        // relaunch has no business searching the web on the user's account.
        insert[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(insert as CFDictionary, nil) == errSecSuccess
    }

    @discardableResult
    static func remove(_ provider: Provider) -> Bool {
        let status = SecItemDelete(baseQuery(provider) as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    private static func baseQuery(_ provider: Provider) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: provider.rawValue,
        ]
    }
}

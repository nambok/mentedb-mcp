//! Secret redaction applied to everything the hooks persist or transmit.
//!
//! Turns and tool-action notes are captured verbatim from real coding
//! sessions, which routinely contain API keys, tokens, and credentials.
//! Nothing may leave the machine or land in the local store unredacted.

use std::sync::LazyLock;

use regex::Regex;

static PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    let raw: &[(&str, &str)] = &[
        // Private key blocks swallow everything between the markers, even
        // when the closing marker was truncated away.
        (
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z0-9 ]*PRIVATE KEY-----|\z)",
            "[redacted:private-key]",
        ),
        (r"\bAKIA[0-9A-Z]{16}\b", "[redacted:aws-key]"),
        (r"\bsk-[A-Za-z0-9_-]{16,}\b", "[redacted:api-key]"),
        (
            r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b",
            "[redacted:github-token]",
        ),
        (
            r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
            "[redacted:slack-token]",
        ),
        (r"\bmdb_[A-Za-z0-9]{16,}\b", "[redacted:mentedb-key]"),
        (
            r"\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
            "[redacted:jwt]",
        ),
        // Scheme plus credential only, so the rest of a captured command
        // line survives redaction.
        (
            r"(?i)\bauthorization\s*:\s*\S+(\s+\S+)?",
            "authorization: [redacted]",
        ),
        (
            r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}",
            "bearer [redacted]",
        ),
        // key=value / key: value assignments for secret-named keys.
        (
            r#"(?i)\b(api[_-]?key|access[_-]?key|secret[_-]?access[_-]?key|client[_-]?secret|secret|token|passwd|password)\b(\s*[=:]\s*)["']?[^\s"']{8,}["']?"#,
            "${1}${2}[redacted]",
        ),
    ];
    raw.iter()
        .map(|(p, r)| (Regex::new(p).expect("valid redaction pattern"), *r))
        .collect()
});

/// Replace anything that looks like a credential with a typed placeholder.
pub fn redact(text: &str) -> String {
    let mut out = std::borrow::Cow::Borrowed(text);
    for (re, replacement) in PATTERNS.iter() {
        if let std::borrow::Cow::Owned(replaced) = re.replace_all(&out, *replacement) {
            out = std::borrow::Cow::Owned(replaced);
        }
    }
    out.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_provider_tokens() {
        let text = "aws AKIAIOSFODNN7EXAMPLE openai sk-abc123def456ghi789jkl \
                    github ghp_abcdefghijklmnopqrstuvwxyz123456 slack xox_b-not-a-token";
        let out = redact(text);
        assert!(out.contains("[redacted:aws-key]"));
        assert!(out.contains("[redacted:api-key]"));
        assert!(out.contains("[redacted:github-token]"));
        assert!(!out.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!out.contains("sk-abc123def456ghi789jkl"));
    }

    #[test]
    fn redacts_headers_and_assignments() {
        let text = r#"curl -H "Authorization: Bearer abc123def456ghi789" and password=supersecret99 plus API_KEY: sk_live_whatever123"#;
        let out = redact(text);
        assert!(!out.contains("abc123def456ghi789"));
        assert!(!out.contains("supersecret99"));
        assert!(out.contains("password=[redacted]"));
    }

    #[test]
    fn redacts_pem_blocks_and_jwts() {
        let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\nlines\n-----END RSA PRIVATE KEY-----";
        assert_eq!(redact(pem), "[redacted:private-key]");
        let jwt = "token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dQw4w9WgXcQabc123";
        assert!(redact(jwt).contains("[redacted:jwt]"));
    }

    #[test]
    fn leaves_normal_text_alone() {
        let text = "Edited file: src/main.rs then ran cargo test, the tokenizer crate is fine";
        assert_eq!(redact(text), text);
    }
}

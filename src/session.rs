//! On-disk session persistence — the versioned DTO that carries one
//! conversation's durable state across process restarts.
//!
//! The session stores **conversation state only**: the message history, the
//! rolling compaction summary, and the last measured prompt size (so the
//! compaction guard stays armed on the first post-restore turn). It carries
//! no provider or model identity — `config.json` stays authoritative for
//! those, and the normalized history is provider-agnostic, so a session
//! authored under one provider restores cleanly under another. The
//! per-process tool budget ledger is deliberately not persisted: it bounds
//! cost across the whole process, not per conversation.

use crate::turn::TurnMessage;

/// The on-disk format version this build writes and accepts. A future format
/// change bumps it, and [`Session::load`] fails fast on a mismatch rather
/// than misreading an old file.
///
/// Growing the [`crate::turn::Block`] vocabulary (for example, `Thinking`) is
/// *not* a
/// format change: every version-1 file ever written still parses under this
/// build, which is what the version protects. The reverse direction — an
/// older build opening a file that contains a new variant — fails fast on
/// the unknown variant name with or without a bump, so bumping would buy
/// nothing and cost every existing file its compatibility.
pub const SESSION_VERSION: u32 = 1;

/// One conversation's durable state, exactly as [`crate::agent::Agent`]
/// holds it in memory. `deny_unknown_fields` mirrors [`crate::config::Config`]:
/// a file this build does not fully understand is an error, not a guess.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Session {
    pub version: u32,
    pub messages: Vec<TurnMessage>,
    pub compacted_summary: Option<String>,
    pub last_input_tokens: u32,
}

impl Session {
    /// Parse a session from a JSON string, rejecting any format version other
    /// than [`SESSION_VERSION`]. The error is bare; [`Session::load`] adds the
    /// file-path context.
    pub fn from_json(contents: &str) -> Result<Session, String> {
        let session: Session = serde_json::from_str(contents).map_err(|e| e.to_string())?;
        if session.version != SESSION_VERSION {
            return Err(format!(
                "unsupported session version {} (this build reads version {SESSION_VERSION})",
                session.version
            ));
        }
        Ok(session)
    }

    /// Read and parse the session file at `path`, naming the file in both the
    /// read and the parse error — the same fail-fast shape as
    /// [`crate::config::Config::load`]. Delegates to [`Session::load_from`];
    /// the named [`crate::session_store::SessionStore`], which holds `PathBuf`
    /// targets, calls that directly to skip the `&str` round-trip.
    pub fn load(path: &str) -> Result<Session, String> {
        Session::load_from(std::path::Path::new(path))
    }

    /// [`Session::load`] over a [`Path`], for callers that already hold one.
    pub fn load_from(path: &std::path::Path) -> Result<Session, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Session::from_json(&contents).map_err(|e| format!("invalid {}: {e}", path.display()))
    }

    /// Write the session to `path` atomically via [`crate::atomic_write`]:
    /// serialize to a sibling temp file, then rename it over the target. A
    /// crash or failure mid-write leaves any existing session file intact —
    /// the rename is the commit point. The temp name carries the pid and a
    /// per-process counter, so two Omega instances sharing a session file
    /// cannot clobber each other's saves on one fixed `.tmp` path.
    pub fn save(&self, path: &str) -> Result<(), String> {
        self.save_to(std::path::Path::new(path))
    }

    /// [`Session::save`] over a [`Path`], the shared write machinery the fixed
    /// `session_file` autosave and the named
    /// [`crate::session_store::SessionStore`] both commit through.
    pub fn save_to(&self, path: &std::path::Path) -> Result<(), String> {
        // Infallible by construction: Session holds no non-string map keys
        // and no fallible Serialize impls, so a failure here is a bug, not a
        // runtime condition to handle.
        let json = serde_json::to_string(self).expect("Session serializes infallibly");
        crate::atomic_write::atomic_write_text(path, &json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::{Block, Role};

    /// A session exercising every persisted shape: both roles, all four
    /// block variants (including a structured `ToolUse.input` and a thinking
    /// block whose signature must survive the round-trip), a present
    /// summary, and a nonzero guard measurement.
    fn full_session() -> Session {
        Session {
            version: SESSION_VERSION,
            messages: vec![
                TurnMessage {
                    role: Role::User,
                    content: vec![Block::Text("read foo".to_string())],
                },
                TurnMessage {
                    role: Role::Assistant,
                    content: vec![
                        Block::Thinking {
                            text: "the user wants foo".to_string(),
                            signature: "sig_1".to_string(),
                        },
                        Block::Text("reading".to_string()),
                        Block::ToolUse {
                            id: "tu_1".to_string(),
                            name: "read_file".to_string(),
                            input: serde_json::json!({"path": "foo", "n": 3}),
                        },
                    ],
                },
                TurnMessage {
                    role: Role::User,
                    content: vec![Block::ToolResult {
                        tool_use_id: "tu_1".to_string(),
                        content: "contents".to_string(),
                        is_error: false,
                    }],
                },
            ],
            compacted_summary: Some("earlier: user set up foo".to_string()),
            last_input_tokens: 1234,
        }
    }

    #[test]
    fn round_trips_through_a_real_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        let path = path.to_str().unwrap();
        let session = full_session();
        session.save(path).unwrap();
        assert_eq!(Session::load(path).unwrap(), session);
    }

    #[test]
    fn round_trips_empty_conversation_state() {
        // The first-run shape: nothing said yet, no summary, guard unarmed.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        let path = path.to_str().unwrap();
        let session = Session {
            version: SESSION_VERSION,
            messages: vec![],
            compacted_summary: None,
            last_input_tokens: 0,
        };
        session.save(path).unwrap();
        assert_eq!(Session::load(path).unwrap(), session);
    }

    #[test]
    fn save_overwrites_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        let path = path.to_str().unwrap();
        full_session().save(path).unwrap();
        let newer = Session {
            version: SESSION_VERSION,
            messages: vec![],
            compacted_summary: None,
            last_input_tokens: 9,
        };
        newer.save(path).unwrap();
        assert_eq!(Session::load(path).unwrap(), newer);
    }

    #[test]
    fn save_leaves_no_temp_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        full_session().save(path.to_str().unwrap()).unwrap();
        // Only the target remains — the pid+counter temp file was renamed away.
        let entries: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn save_survives_a_stale_fixed_tmp_from_another_process() {
        // F13 regression: the writer must not collide on a fixed `.tmp` name.
        // A stale `{path}.tmp` left by another instance — here a directory,
        // which the old fixed-name scheme would have failed the write against
        // — must not block this save, because the temp name now carries a
        // pid+counter suffix that no other process shares.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        let path = path.to_str().unwrap();
        std::fs::create_dir(format!("{path}.tmp")).unwrap();
        full_session().save(path).unwrap();
        assert_eq!(Session::load(path).unwrap(), full_session());
    }

    #[test]
    fn failed_rename_reports_both_paths() {
        // A directory at the target makes the temp write succeed and the
        // rename fail — the commit point itself erroring, surfaced through
        // Session::save.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        let path = path.to_str().unwrap();
        std::fs::create_dir(path).unwrap();
        let err = full_session().save(path).unwrap_err();
        assert!(err.contains("cannot rename"), "got: {err}");
        assert!(err.contains("session.json"), "got: {err}");
        assert!(err.contains(".tmp"), "got: {err}");
        assert!(err.contains("over"), "got: {err}");
    }

    #[test]
    fn load_missing_file_reports_path() {
        let err = Session::load("/no/such/session.json").unwrap_err();
        assert!(
            err.contains("cannot read /no/such/session.json"),
            "got: {err}"
        );
    }

    #[test]
    fn load_corrupt_file_reports_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        std::fs::write(&path, "{not json").unwrap();
        let err = Session::load(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("invalid"), "got: {err}");
        assert!(err.contains("session.json"), "got: {err}");
    }

    #[test]
    fn load_version_mismatch_reports_version_and_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        std::fs::write(
            &path,
            r#"{"version": 2, "messages": [], "compacted_summary": null, "last_input_tokens": 0}"#,
        )
        .unwrap();
        let err = Session::load(path.to_str().unwrap()).unwrap_err();
        assert!(
            err.contains("unsupported session version 2 (this build reads version 1)"),
            "got: {err}"
        );
        assert!(err.contains("session.json"), "got: {err}");
    }

    #[test]
    fn from_json_accepts_a_pre_thinking_version_1_file() {
        // The compatibility contract behind keeping SESSION_VERSION at 1 when
        // the Block vocabulary grew: a file written by a build that predates
        // Block::Thinking (the raw JSON below, not this build's serializer)
        // must still parse. The reverse — an old build reading a new file
        // with a thinking block — fails fast on the unknown variant, which a
        // version bump could not improve on.
        let session = Session::from_json(
            r#"{
                "version": 1,
                "messages": [
                    {"role": "User", "content": [{"Text": "hi"}]},
                    {"role": "Assistant", "content": [{"Text": "hello"}]}
                ],
                "compacted_summary": null,
                "last_input_tokens": 7
            }"#,
        )
        .unwrap();
        assert_eq!(session.messages.len(), 2);
        assert_eq!(
            session.messages[1].content,
            vec![Block::Text("hello".to_string())]
        );
        assert_eq!(session.last_input_tokens, 7);
    }

    #[test]
    fn from_json_rejects_unknown_fields() {
        // `deny_unknown_fields`, like Config: a file with fields this build
        // does not understand is an error, not silently dropped state.
        let err = Session::from_json(
            r#"{"version": 1, "messages": [], "compacted_summary": null, "last_input_tokens": 0, "model": "m"}"#,
        )
        .unwrap_err();
        assert!(err.contains("unknown field"), "got: {err}");
    }

    #[test]
    fn from_json_rejects_missing_fields() {
        // Every field is mandatory — a truncated file must not parse as an
        // emptier-but-valid session.
        let err = Session::from_json(r#"{"version": 1}"#).unwrap_err();
        assert!(err.contains("missing field"), "got: {err}");
    }
}

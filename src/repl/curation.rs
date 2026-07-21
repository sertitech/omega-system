//! Curation of a provider's model catalog for the REPL's two listing
//! surfaces — Tab completion and the printed `/model <provider>:` listing.
//!
//! `GET /v1/models` returns everything the vendor serves. OpenAI's catalog
//! (120 ids at the 2026-07 live audit) is mostly ids the agent cannot
//! converse with — audio, image, video, embedding, and moderation models —
//! plus dated snapshots duplicating a current alias. Curation hides both
//! classes, leaving the ~30 usable, distinct chat models; Anthropic's
//! catalog (10 chat ids) passes through untouched. It is presentation only:
//! `/model` accepts any typed id, hidden or not.

/// A curated catalog: the ids worth offering, in the incoming (newest-first)
/// order, and how many were hidden. The count keeps the printed listing
/// honest — the catalog did not shrink, the noise did.
pub struct Curated {
    pub shown: Vec<String>,
    pub hidden: usize,
}

/// Id substrings marking a model the agent cannot converse with over the
/// chat endpoint it drives. A denylist rather than an allowlist of chat
/// families, deliberately: it fails open, so a newly shipped chat family
/// appears without a code change, at the cost of curating new *non-chat*
/// marker words as vendors coin them. Every entry is grounded in the
/// 2026-07 live catalog (e.g. `whisper-1`, `gpt-4o-mini-tts`,
/// `gpt-realtime-2`, `text-embedding-3-large`, `sora-2`, `gpt-image-2`,
/// `omni-moderation-latest`, `gpt-3.5-turbo-instruct`, `babbage-002`,
/// `o4-mini-deep-research` — the last usable only via an API the agent
/// does not speak).
const NON_CHAT_MARKERS: [&str; 13] = [
    "whisper",
    "tts",
    "transcribe",
    "audio",
    "realtime",
    "image",
    "sora",
    "embedding",
    "moderation",
    "instruct",
    "babbage",
    "davinci",
    "deep-research",
];

/// Hide the ids the agent cannot use and the dated snapshots whose alias is
/// also listed, preserving the incoming order.
///
/// Snapshot collapsing is guarded, not unconditional: a dated id with no
/// listed alias (Anthropic's `claude-haiku-4-5-20251001`) is the only name
/// that model has and stays visible.
pub fn curate(ids: Vec<String>) -> Curated {
    let total = ids.len();
    let chat: Vec<String> = ids
        .into_iter()
        .filter(|id| !NON_CHAT_MARKERS.iter().any(|marker| id.contains(marker)))
        .collect();
    let shown: Vec<String> = chat
        .iter()
        .filter(|id| {
            !snapshot_alias(id).is_some_and(|alias| chat.iter().any(|other| other == alias))
        })
        .cloned()
        .collect();
    Curated {
        hidden: total - shown.len(),
        shown,
    }
}

/// The alias a dated snapshot id abbreviates to, if the id ends in a date
/// suffix: `-YYYY-MM-DD` (`gpt-5.4-2026-03-05`), `-YYYYMMDD`
/// (`claude-opus-4-5-20251101`), or `-MMDD` (`gpt-4-0613`). Longest shape
/// first, so a full date is never misread as its own tail. Whether the
/// alias is actually listed is the caller's judgment.
fn snapshot_alias(id: &str) -> Option<&str> {
    const SHAPES: [&str; 3] = ["-0000-00-00", "-00000000", "-0000"];
    SHAPES.iter().find_map(|shape| {
        let (alias, suffix) = id.split_at_checked(id.len().checked_sub(shape.len())?)?;
        let fits = suffix.bytes().zip(shape.bytes()).all(|(byte, template)| {
            if template == b'-' {
                byte == b'-'
            } else {
                byte.is_ascii_digit()
            }
        });
        (fits && !alias.is_empty()).then_some(alias)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run [`curate`] over string literals.
    fn curate_ids(ids: &[&str]) -> Curated {
        curate(ids.iter().map(|id| id.to_string()).collect())
    }

    #[test]
    fn chat_ids_pass_through_in_order() {
        let curated = curate_ids(&["claude-sonnet-5", "claude-fable-5", "claude-opus-4-8"]);
        assert_eq!(
            curated.shown,
            ["claude-sonnet-5", "claude-fable-5", "claude-opus-4-8"]
        );
        assert_eq!(curated.hidden, 0);
    }

    #[test]
    fn every_non_chat_marker_hides_a_live_catalog_id() {
        // One representative per marker, straight from the 2026-07 probe —
        // a marker no id exercises would be a dead list entry.
        let junk = [
            "whisper-1",
            "gpt-4o-mini-tts",
            "gpt-4o-transcribe-diarize",
            "gpt-audio-1.5",
            "gpt-realtime-2",
            "chatgpt-image-latest",
            "sora-2-pro",
            "text-embedding-3-large",
            "omni-moderation-latest",
            "gpt-3.5-turbo-instruct",
            "babbage-002",
            "davinci-002",
            "o4-mini-deep-research",
        ];
        assert_eq!(junk.len(), NON_CHAT_MARKERS.len());
        for (id, marker) in junk.iter().zip(NON_CHAT_MARKERS) {
            let curated = curate_ids(&[id]);
            assert!(curated.shown.is_empty(), "{id} survived");
            assert_eq!(curated.hidden, 1);
            assert!(id.contains(marker), "{id} does not exercise {marker:?}");
        }
    }

    #[test]
    fn dated_snapshots_hide_only_when_their_alias_is_listed() {
        // Both OpenAI date shapes collapse onto a listed alias; Anthropic's
        // compact-dated id has no alias in the catalog and must survive.
        let curated = curate_ids(&[
            "gpt-5.4",
            "gpt-5.4-2026-03-05",
            "gpt-4",
            "gpt-4-0613",
            "claude-opus-4-5-20251101",
        ]);
        assert_eq!(
            curated.shown,
            ["gpt-5.4", "gpt-4", "claude-opus-4-5-20251101"]
        );
        assert_eq!(curated.hidden, 2);
    }

    #[test]
    fn hidden_counts_both_classes_together() {
        // The printed listing reports one number: junk and collapsed
        // snapshots accumulate into it.
        let curated = curate_ids(&["gpt-5.5", "gpt-5.5-2026-04-23", "tts-1-hd-1106"]);
        assert_eq!(curated.shown, ["gpt-5.5"]);
        assert_eq!(curated.hidden, 2);
    }

    #[test]
    fn snapshot_alias_recognizes_each_date_shape() {
        assert_eq!(snapshot_alias("gpt-5.4-2026-03-05"), Some("gpt-5.4"));
        assert_eq!(
            snapshot_alias("claude-opus-4-5-20251101"),
            Some("claude-opus-4-5")
        );
        assert_eq!(snapshot_alias("gpt-3.5-turbo-0125"), Some("gpt-3.5-turbo"));
    }

    #[test]
    fn snapshot_alias_rejects_undated_and_degenerate_ids() {
        // No date tail; a non-digit tail; an id shorter than every shape;
        // and a bare date whose alias would be empty.
        assert_eq!(snapshot_alias("gpt-5.4"), None);
        assert_eq!(snapshot_alias("gpt-3.5-turbo-16k"), None);
        assert_eq!(snapshot_alias("o3"), None);
        assert_eq!(snapshot_alias("-20251101"), None);
    }

    #[test]
    fn snapshot_alias_survives_multibyte_ids() {
        // 'é' is two bytes. In the first id the 5-byte `-0000` shape splits
        // on a char boundary and matches; in the second the split lands
        // *inside* the 'é' — `split_at_checked` declines where a plain
        // `split_at` would panic.
        assert_eq!(snapshot_alias("é-0000"), Some("é"));
        assert_eq!(snapshot_alias("aé-000"), None);
    }
}

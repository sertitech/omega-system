//! Terminal-safety policies for untrusted text bound for the operator's screen.
//!
//! Model output, tool results, and filesystem-derived paths are all
//! indirect-injection channels: a raw ESC/BEL/CR or C1 byte could clear the
//! screen, move the cursor, or forge a prompt or tool line, and a bidi override
//! or zero-width character could visually reorder or hide text. Everything the
//! agent prints from one of those channels passes through one of the policies
//! here first.
//!
//! **The classification is shared; the policies are deliberately not.** All
//! five consult the same predicates — [`char::is_control`] (the C0 + DEL + C1
//! set) and [`is_invisible_format`] (the bidi/zero-width set) — but they differ
//! on two axes that are per-channel choices, not accidents:
//!
//! - **Replacement glyph.** Model- and stream-facing text ([`scrub_controls`])
//!   and tool-echoed paths ([`escape_control_chars`]) replace with `U+FFFD`,
//!   the Unicode replacement character. Operator-facing agent status lines
//!   ([`sanitize_for_display`], [`sanitize_multiline`]) replace with `?`.
//! - **Newline handling.** [`scrub_controls`] keeps `\n` (line structure drives
//!   the Markdown renderer); [`sanitize_multiline`] keeps `\n` as a separator
//!   while scrubbing each line; [`sanitize_for_display`] and
//!   [`escape_control_chars`] scrub `\n` too (a newline forges a fake status
//!   line or an extra tool-output row).
//!
//! Exact file-review text uses [`escape_for_review`] instead: controls and
//! invisible formats become visible escape sequences, and literal backslashes
//! are doubled so two different edits cannot collapse onto the same display.
//! The preview adds its own line structure after escaping each content line.
//!
//! Keeping every policy side by side means shared character-set changes touch
//! one file and the table test pins their differences.

/// The invisible-format characters every display policy scrubs. None render as
/// a visible glyph, yet each can hide, reorder, or smuggle text past the
/// operator and the model, which ordinary control-byte scrubbing
/// (`char::is_control`, the C0 + DEL + C1 set) leaves untouched:
/// - bidi controls — U+202A..=U+202E overrides/embeddings, U+2066..=U+2069
///   isolates, U+061C ALM, and the U+200E/U+200F marks — visually reorder the
///   text that follows (U+202E flips its direction);
/// - zero-width characters — U+200B..=U+200D and the U+2060..=U+206F
///   word-joiner/invisible-operator block — hide or reorder characters;
/// - blank-rendering fillers — the soft hyphen U+00AD, the Hangul fillers
///   U+115F/U+1160/U+3164/U+FFA0, and the BOM/ZWNBSP U+FEFF — read as empty;
/// - deprecated interlinear annotation controls U+FFF9..=U+FFFB;
/// - the Tags block U+E0000..=U+E007F, whose only modern use is smuggling
///   hidden ASCII instructions into an LLM.
///
/// Variation selectors (U+FE00..=U+FE0F, U+E0100..=U+E01EF) are deliberately
/// NOT scrubbed: U+FE0F drives emoji presentation and the supplement drives CJK
/// ideographic variation sequences, a high cost on a hot path. Shared so every
/// policy classifies identically.
pub(crate) fn is_invisible_format(c: char) -> bool {
    matches!(
        c as u32,
        0x00AD                  // SOFT HYPHEN — invisible except at a line break
            | 0x061C            // ARABIC LETTER MARK
            | 0x115F | 0x1160   // HANGUL CHOSEONG/JUNGSEONG FILLER (blank)
            | 0x200B..=0x200F   // zero-width set + LRM/RLM
            | 0x202A..=0x202E   // bidi embeddings/overrides
            | 0x2060..=0x206F   // word joiner / invisible operators
            | 0x3164            // HANGUL FILLER (blank)
            | 0xFEFF            // BOM / ZWNBSP
            | 0xFFA0            // HALFWIDTH HANGUL FILLER (blank)
            | 0xFFF9..=0xFFFB   // interlinear annotation controls (deprecated)
            | 0xE0000..=0xE007F // Tags block — ASCII smuggling into the model
    )
}

/// Replace every control character (except the `\n` the renderer consumes) and
/// every [invisible-format character](is_invisible_format) with U+FFFD. Model
/// output is an indirect-injection channel (e.g. text fetched by `web_fetch`),
/// so a raw ESC/BEL/CR or C1 byte could clear the screen, move the cursor, or
/// forge a prompt or tool line, and a bidi override or zero-width character
/// could visually reorder or hide the text. `char::is_control` covers the C0,
/// DEL, and C1 controls; `\n` alone survives because line structure drives the
/// renderer, and it is the only C0 byte the acceptance permits through. Tab is
/// scrubbed like any other control (never expanded). Differs from the
/// operator-facing [`sanitize_for_display`] only in the newline exemption and
/// the replacement glyph (U+FFFD here, `?` there). The renderer emits its own
/// SGR styling *after* this scrub, so its escapes are unaffected.
pub(crate) fn scrub_controls(text: &str) -> String {
    text.chars()
        .map(|c| {
            if (c != '\n' && c.is_control()) || is_invisible_format(c) {
                '\u{FFFD}'
            } else {
                c
            }
        })
        .collect()
}

/// Strip control characters (ANSI escapes, newlines) and invisible Unicode
/// formatting (bidi overrides, zero-width characters) from a string before
/// displaying it in the terminal. Prevents prompt spoofing and visual
/// reordering from model-controlled input. Shares its character classification
/// with the streamed-text [`scrub_controls`] — both scrub the same controls and
/// [invisible-format set](is_invisible_format), differing only in the newline
/// exemption (this scrubs every control, the renderer keeps `\n`) and the
/// replacement glyph (`?` here, U+FFFD there).
pub(crate) fn sanitize_for_display(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_control() || is_invisible_format(c) {
                '?'
            } else {
                c
            }
        })
        .collect()
}

/// Sanitize a multi-line string for display, preserving line structure:
/// each line is scrubbed like [`sanitize_for_display`], the newlines
/// between them survive. For tool errors, which are read in full when
/// something breaks and are often several lines of stderr.
pub(crate) fn sanitize_multiline(s: &str) -> String {
    s.lines()
        .map(sanitize_for_display)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Replace every control character (newlines, tabs, ESC, etc.) and every
/// invisible-format character — bidi controls, zero-width characters, and the
/// BOM/ZWNBSP U+FEFF, the set classified by [`is_invisible_format`] — with
/// U+FFFD so a crafted filename cannot spoof a tool's output. A newline-bearing
/// name would otherwise forge extra `list_directory` entries or fake
/// `path:line:` rows in `search_files`, and a bidi override or zero-width
/// character could visually reorder or hide a path component. One policy, shared
/// by both file tools that echo filesystem-derived paths back to the model.
pub(crate) fn escape_control_chars(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_control() || is_invisible_format(c) {
                '\u{FFFD}'
            } else {
                c
            }
        })
        .collect()
}

/// Render an exact review line: controls and invisible formats use visible Rust
/// escapes, and literal backslashes are doubled so escaped and literal bytes
/// cannot look identical in an approval preview. Line structure is added by
/// the preview renderer after escaping each individual line.
pub(crate) fn escape_for_review(s: &str) -> String {
    let mut escaped = String::new();
    for c in s.chars() {
        if c.is_control() || is_invisible_format(c) || c == '\\' {
            escaped.extend(c.escape_default());
        } else {
            escaped.push(c);
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    const FFFD: char = '\u{FFFD}';

    #[test]
    fn review_escapes_keep_literal_and_control_text_distinct() {
        assert_eq!(escape_for_review("\n\r\t\\n"), "\\n\\r\\t\\\\n");
        assert_eq!(escape_for_review("a\u{202e}b"), "a\\u{202e}b");
    }

    /// The shared contract for every display policy in one table: each row is a
    /// character class, each policy a column. Control (C0/DEL/C1) and
    /// invisible-format characters are replaced by every policy, differing only
    /// in the glyph; ordinary characters (including the non-control U+00A0 and
    /// the range neighbours the classifier must *not* catch) pass through
    /// everywhere; `\n` is the axis the four diverge on.
    #[test]
    fn each_policy_pins_its_glyph_and_newline_rule() {
        // Every policy replaces these, differing only in the glyph. Spans the
        // C0 controls, DEL, both C1 bounds, and the invisible-format set
        // (zero-width, bidi override/isolate, the U+2060..=U+206F high bound,
        // the BOM, and the Arabic letter mark).
        let scrubbed = [
            '\x1b',
            '\x07',
            '\r',
            '\t',
            '\x00',
            '\x7f',
            '\u{0080}',
            '\u{009f}',
            '\u{200b}',
            '\u{202e}',
            '\u{2066}',
            '\u{206f}',
            '\u{feff}',
            '\u{061c}',
            '\u{00ad}',
            '\u{115f}',
            '\u{1160}',
            '\u{3164}',
            '\u{ffa0}',
            '\u{fff9}',
            '\u{fffb}',
            '\u{e0000}',
            '\u{e0041}',
            '\u{e007f}',
        ];
        for c in scrubbed {
            let s = format!("x{c}y");
            // U+FFFD channels: streamed model text and tool-echoed paths.
            assert_eq!(scrub_controls(&s), format!("x{FFFD}y"));
            assert_eq!(escape_control_chars(&s), format!("x{FFFD}y"));
            // `?` channels: operator-facing agent status lines.
            assert_eq!(sanitize_for_display(&s), "x?y");
            assert_eq!(sanitize_multiline(&s), "x?y");
            assert_eq!(escape_for_review(&s), format!("x{}y", c.escape_default()));
            // The shared classification flags every one.
            assert!(c.is_control() || is_invisible_format(c));
        }

        // Every policy passes these through untouched: ordinary text, the
        // non-control U+00A0, the range neighbours the classifier must not catch
        // (U+200A/U+2070 around the zero-width block, and one either side of each
        // added filler/range), and — pinned as a deliberate scope exclusion — the
        // variation selectors U+FE0F and U+E0100.
        let kept = [
            'a',
            ' ',
            'é',
            '日',
            '🦀',
            '\u{a0}',
            '\u{200a}',
            '\u{2070}',
            '\u{00ac}',
            '\u{00ae}',
            '\u{115e}',
            '\u{1161}',
            '\u{3163}',
            '\u{3165}',
            '\u{ff9f}',
            '\u{ffa1}',
            '\u{fff8}',
            '\u{fffc}',
            '\u{e0080}',
            '\u{fe0f}',
            '\u{e0100}',
        ];
        for c in kept {
            let s = format!("x{c}y");
            assert_eq!(scrub_controls(&s), s);
            assert_eq!(escape_control_chars(&s), s);
            assert_eq!(sanitize_for_display(&s), s);
            assert_eq!(sanitize_multiline(&s), s);
            assert_eq!(escape_for_review(&s), s);
            assert!(!c.is_control() && !is_invisible_format(c));
        }

        // Newline is the divergence axis:
        //   scrub_controls keeps it (line structure drives the renderer),
        //   escape_control_chars replaces it (a newline forges tool rows),
        //   sanitize_for_display replaces it (a single operator line),
        //   sanitize_multiline keeps it as a separator, scrubbing each line.
        assert_eq!(scrub_controls("a\nb"), "a\nb");
        assert_eq!(escape_control_chars("a\nb"), format!("a{FFFD}b"));
        assert_eq!(sanitize_for_display("a\nb"), "a?b");
        assert_eq!(
            sanitize_multiline("plain\n\x1b[31mred\nlast"),
            "plain\n?[31mred\nlast"
        );
        // The multiline splitter's edge inputs: a single line, and empty.
        assert_eq!(sanitize_multiline("one line"), "one line");
        assert_eq!(sanitize_multiline(""), "");
    }
}

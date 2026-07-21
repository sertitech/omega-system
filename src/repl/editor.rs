//! The raw-mode line editor's hermetic core: one keystroke loop over an
//! injected byte source and render sink. The live shell around it —
//! `termios` raw-mode toggling and the real stdin/stdout — lives in
//! [`super::raw_live`]; everything observable here (editing, completion,
//! rendering) is driven in tests by feeding byte scripts.
//!
//! The editor keeps a real cursor rather than assuming it stays at the end:
//! Left/Right and Ctrl-B/Ctrl-F move by
//! character, Home/End and Ctrl-A/Ctrl-E jump the line, Ctrl-/Alt-arrows
//! (and readline's Alt-b/Alt-f) move by whitespace-delimited word, and
//! printable input and Backspace edit *at* the cursor. History recall
//! (Up/Down) still replaces the whole line and puts the cursor at the end —
//! recall semantics are untouched. Escape sequences the editor does not act
//! on are consumed and discarded whole: in raw mode an unhandled key arrives
//! as multiple bytes of garbage if merely ignored.
//!
//! Edits re-render the line in place (carriage return, erase to end of
//! line, prompt, buffer, then a cursor-left hop to the edit point). Two
//! accepted approximations, unchanged in spirit from the original editor:
//! a line wrapped past one terminal row garbles on redraw (completion
//! targets short command lines), and the cursor hop counts *characters*,
//! so double-width glyphs (CJK, emoji) left of the tail offset the visual
//! cursor — full width-awareness is not a goal.

use super::PROMPT;
use super::completion::Completion;
use std::io::{ErrorKind, Read, Write};

const CTRL_A: u8 = 0x01;
const CTRL_B: u8 = 0x02;
const CTRL_C: u8 = 0x03;
const CTRL_D: u8 = 0x04;
const CTRL_E: u8 = 0x05;
const CTRL_F: u8 = 0x06;
const CTRL_G: u8 = 0x07;
const BACKSPACE: u8 = 0x08;
const CTRL_R: u8 = 0x12;
const TAB: u8 = b'\t';
const ESC: u8 = 0x1b;
const DEL: u8 = 0x7f;

/// Edit one line: consume keystrokes from `input` until Enter, Ctrl-D on an
/// empty line, or the source runs dry, echoing to `out` as the line takes
/// shape. The finished line lands appended to `buf` — with its newline,
/// mirroring `BufRead::read_line` so this can back [`super::LineSource`] —
/// and the byte count returns (0 = end of input). `complete` supplies Tab
/// candidates for the buffer as typed so far.
///
/// `history` is the Up/Down recall store, owned by the caller so it
/// outlives the call: Enter appends the submitted line (blank lines and a
/// repeat of the newest entry are not stored), and Up/Down replace the
/// whole line with the previous/next entry through the in-place redraw,
/// cursor at the end. Stepping down past the newest entry restores the line
/// as it stood when recall began (cursor likewise at the end); edits made
/// *while* recalled belong to the line, not the store, so stepping away
/// discards them. Cancelled (Ctrl-C) and EOF-cut lines are never stored.
///
/// Ctrl-C cancels the line: the typed text is discarded and a bare newline
/// is yielded, so the caller re-prompts rather than exiting. Ctrl-D on a
/// non-empty line is ignored (delete-forward is out of scope). A source
/// that dries up mid-line yields the partial line without a newline —
/// cooked `read_line`'s EOF contract.
///
/// Trait objects, not generics, on purpose: keystrokes arrive at human
/// speed, and one instantiation keeps the whole editor inside a single
/// coverage unit — with generic params, the gate scores the best-covered
/// instantiation alone, and the error paths driven by dedicated failing
/// test doubles would read as uncovered (the 6a llvm-cov lesson).
pub(super) fn edit_line(
    input: &mut dyn Read,
    out: &mut dyn Write,
    buf: &mut String,
    complete: &mut dyn FnMut(&str) -> Completion,
    history: &mut Vec<String>,
    after_cr: &mut bool,
) -> std::io::Result<usize> {
    let mut line = String::new();
    // Byte offset of the cursor within `line`, always on a char boundary.
    let mut cursor = 0usize;
    // Recall state, scoped to this line: the history index currently shown
    // (`None` = editing the live draft) and the draft saved at first Up.
    let mut recall: Option<usize> = None;
    let mut draft = String::new();
    loop {
        let byte = next_byte(input)?;
        // Swallow the LF of a CRLF pair. Enter finishes a line on `\r`, so a
        // CRLF paste (or a source speaking `\r\n`) would submit the CR's line
        // and then submit an empty one on the trailing LF. `after_cr` — the
        // caller's cross-call flag — is set by the CR submit below and cleared
        // by this `take` on the next byte, so only an LF *immediately* after a
        // CR is dropped: a lone LF still submits, and CR CR submits twice.
        if std::mem::take(after_cr) && byte == Some(b'\n') {
            continue;
        }
        match byte {
            None => return finish(buf, line),
            Some(b'\r') | Some(b'\n') => {
                *after_cr = byte == Some(b'\r');
                echo(out, b"\r\n")?;
                remember(history, &line);
                line.push('\n');
                return finish(buf, line);
            }
            Some(CTRL_D) if line.is_empty() => {
                echo(out, b"\r\n")?;
                return Ok(0);
            }
            Some(CTRL_D) => {}
            Some(CTRL_C) => {
                echo(out, b"^C\r\n")?;
                line.clear();
                line.push('\n');
                return finish(buf, line);
            }
            Some(CTRL_A) => move_cursor(out, &line, &mut cursor, Motion::Home)?,
            Some(CTRL_E) => move_cursor(out, &line, &mut cursor, Motion::End)?,
            Some(CTRL_B) => move_cursor(out, &line, &mut cursor, Motion::Left)?,
            Some(CTRL_F) => move_cursor(out, &line, &mut cursor, Motion::Right)?,
            Some(BACKSPACE) | Some(DEL) => {
                if cursor > 0 {
                    let prev = prev_boundary(&line, cursor);
                    line.remove(prev);
                    cursor = prev;
                    redraw(out, PROMPT, &line, cursor)?;
                }
            }
            Some(TAB) => {
                let Completion { start, candidates } = complete(&line);
                match candidates.as_slice() {
                    [] => {}
                    [only] => {
                        // Completion considers the whole line as typed and
                        // leaves the cursor at the end — it targets
                        // end-of-line typing, wherever the cursor sat.
                        line.truncate(start);
                        line.push_str(only);
                        cursor = line.len();
                        redraw(out, PROMPT, &line, cursor)?;
                    }
                    several => {
                        // List the choices on their own line, then re-render
                        // the untouched input line beneath them.
                        echo(out, format!("\r\n{}\r\n", several.join("  ")).as_bytes())?;
                        redraw(out, PROMPT, &line, cursor)?;
                    }
                }
            }
            Some(CTRL_R) => {
                // Reverse incremental search over the recall store, readline-
                // style. It runs its own keystroke loop and reports what the
                // editor should do next: submit the match, adopt it into the
                // edit buffer (optionally trailed by a motion the escape
                // sequence carried), or fall back to the untouched pre-search
                // line.
                match reverse_i_search(input, out, history, &line)? {
                    SearchOutcome::Submit { line: matched, cr } => {
                        *after_cr = cr;
                        redraw(out, PROMPT, &matched, matched.len())?;
                        echo(out, b"\r\n")?;
                        remember(history, &matched);
                        let mut submitted = matched;
                        submitted.push('\n');
                        return finish(buf, submitted);
                    }
                    SearchOutcome::Accept {
                        line: matched,
                        motion,
                    } => {
                        // The match becomes a fresh draft: recall starts over
                        // from it, and a motion the escape sequence carried
                        // lands on it exactly as it would on freshly typed text.
                        line = matched;
                        cursor = line.len();
                        recall = None;
                        draft.clear();
                        redraw(out, PROMPT, &line, cursor)?;
                        if let Some(motion) = motion {
                            move_cursor(out, &line, &mut cursor, motion)?;
                        }
                    }
                    SearchOutcome::Cancel => redraw(out, PROMPT, &line, cursor)?,
                }
            }
            Some(ESC) => match read_escape_sequence(input)? {
                Some(Key::Up) => {
                    // First Up recalls the newest entry; each further Up
                    // steps older. At the oldest (or with no history at
                    // all) the key is a quiet no-op.
                    let target = match recall {
                        None => history.len().checked_sub(1),
                        Some(shown) => shown.checked_sub(1),
                    };
                    if let Some(index) = target {
                        if recall.is_none() {
                            draft = std::mem::take(&mut line);
                        }
                        recall = Some(index);
                        line.clear();
                        line.push_str(&history[index]);
                        cursor = line.len();
                        redraw(out, PROMPT, &line, cursor)?;
                    }
                }
                Some(Key::Down) => {
                    // Steps newer through the entries; past the newest it
                    // restores the saved draft. Without active recall the
                    // key is a quiet no-op.
                    if let Some(shown) = recall {
                        if let Some(entry) = history.get(shown + 1) {
                            recall = Some(shown + 1);
                            line.clear();
                            line.push_str(entry);
                        } else {
                            recall = None;
                            line = std::mem::take(&mut draft);
                        }
                        cursor = line.len();
                        redraw(out, PROMPT, &line, cursor)?;
                    }
                }
                Some(Key::Motion(motion)) => move_cursor(out, &line, &mut cursor, motion)?,
                None => {}
            },
            // Remaining control bytes have no binding — discarded.
            Some(byte) if byte < 0x20 => {}
            // Printable ASCII: insert at the cursor.
            Some(byte) if byte < 0x80 => insert_char(out, &mut line, &mut cursor, byte as char)?,
            Some(lead) => {
                if let Some(ch) = read_utf8(input, lead)? {
                    insert_char(out, &mut line, &mut cursor, ch)?;
                }
            }
        }
    }
}

/// Append the finished line to the caller's buffer, `read_line`-style.
fn finish(buf: &mut String, line: String) -> std::io::Result<usize> {
    buf.push_str(&line);
    Ok(line.len())
}

/// Store a submitted line for Up/Down recall. Blank lines carry nothing
/// worth recalling, and a repeat of the newest entry is not double-stored —
/// resubmitting a recalled line must not silt the history up with copies.
/// Non-consecutive duplicates are kept: each still marks a distinct point
/// in the stepping order.
fn remember(history: &mut Vec<String>, line: &str) {
    if !line.trim().is_empty() && history.last().map(String::as_str) != Some(line) {
        history.push(line.to_string());
    }
}

/// Insert `ch` at the cursor. At the end of the line this is the fast path —
/// append and echo, no redraw; mid-line the whole line re-renders with the
/// cursor hopped back to just after the insertion.
fn insert_char(
    out: &mut dyn Write,
    line: &mut String,
    cursor: &mut usize,
    ch: char,
) -> std::io::Result<()> {
    if *cursor == line.len() {
        line.push(ch);
        *cursor = line.len();
        echo(out, ch.encode_utf8(&mut [0u8; 4]).as_bytes())
    } else {
        line.insert(*cursor, ch);
        *cursor += ch.len_utf8();
        redraw(out, PROMPT, line, *cursor)
    }
}

/// The keys the editor acts on: history recall plus the cursor motions.
enum Key {
    Up,
    Down,
    Motion(Motion),
}

/// A cursor movement. Character steps, line jumps, and whitespace-delimited
/// word hops — each mapped from both its escape sequence(s) and (where one
/// exists) its control-byte chord.
enum Motion {
    Left,
    Right,
    Home,
    End,
    WordLeft,
    WordRight,
}

/// Apply a motion: compute the target offset and, when it differs, move and
/// re-render. A motion that cannot move (Left at the start, Right at the
/// end, WordLeft at the start …) is a quiet no-op — no redraw, no flicker.
fn move_cursor(
    out: &mut dyn Write,
    line: &str,
    cursor: &mut usize,
    motion: Motion,
) -> std::io::Result<()> {
    let target = match motion {
        Motion::Left => prev_boundary(line, *cursor),
        Motion::Right => next_boundary(line, *cursor),
        Motion::Home => 0,
        Motion::End => line.len(),
        Motion::WordLeft => word_left(line, *cursor),
        Motion::WordRight => word_right(line, *cursor),
    };
    if target != *cursor {
        *cursor = target;
        redraw(out, PROMPT, line, *cursor)?;
    }
    Ok(())
}

/// What a completed reverse search tells [`edit_line`] to do next.
enum SearchOutcome {
    /// Enter accepted the match — submit it as the finished line. `cr` carries
    /// whether the terminator was a bare `\r`, so the caller keeps the same
    /// CRLF-swallow contract as a normally-typed Enter.
    Submit { line: String, cr: bool },
    /// A complete escape sequence ended the search: adopt the match into the
    /// edit buffer, then apply the `motion` it decoded to (if any) — arrows,
    /// Home/End, and word hops land on the freshly adopted line. Non-motion
    /// sequences (Up/Down, or anything unbound) still accept, with no motion.
    Accept {
        line: String,
        motion: Option<Motion>,
    },
    /// Ctrl-G, Ctrl-C, or a dry source abandoned the search — resume the
    /// pre-search line untouched.
    Cancel,
}

/// Run a reverse-incremental search over `entries`, readline-style, starting
/// from `start_line` (the line as it stood when Ctrl-R was pressed, and the
/// candidate shown while the query is empty or matches nothing).
///
/// The query grows and shrinks a plain substring, matched newest→oldest over
/// the same slice the arrows recall; a further Ctrl-R steps to the next older
/// match. A keystroke that would leave the query matching nothing is refused —
/// the query is left as it was and a `failed` marker is shown — so the search
/// never strands the user on an empty result. Returns once the user commits
/// (Enter), redirects (an escape sequence), or backs out (Ctrl-G/Ctrl-C/EOF).
fn reverse_i_search(
    input: &mut dyn Read,
    out: &mut dyn Write,
    entries: &[String],
    start_line: &str,
) -> std::io::Result<SearchOutcome> {
    let mut query = String::new();
    // Index into `entries` of the current match; `None` while the query is
    // empty (the candidate is then `start_line`).
    let mut matched: Option<usize> = None;
    let mut failed = false;
    render_search(out, &query, candidate(entries, matched, start_line), failed)?;
    loop {
        match next_byte(input)? {
            None => return Ok(SearchOutcome::Cancel),
            Some(byte) => match byte {
                b'\r' | b'\n' => {
                    let matched = candidate(entries, matched, start_line).to_string();
                    return Ok(SearchOutcome::Submit {
                        line: matched,
                        cr: byte == b'\r',
                    });
                }
                CTRL_G | CTRL_C => return Ok(SearchOutcome::Cancel),
                ESC => {
                    // Any complete escape sequence ends the search and accepts;
                    // only a motion it decoded to travels on to the adopted line
                    // (Up/Down recall and unbound sequences carry none).
                    let motion = match read_escape_sequence(input)? {
                        Some(Key::Motion(motion)) => Some(motion),
                        _ => None,
                    };
                    let matched = candidate(entries, matched, start_line).to_string();
                    return Ok(SearchOutcome::Accept {
                        line: matched,
                        motion,
                    });
                }
                CTRL_R => {
                    // Step to the next older match: search the entries strictly
                    // older than the current one (or, with none yet shown, from
                    // the newest). No older match leaves the current one and
                    // flags `failed`.
                    let upper = match matched {
                        Some(i) => i.checked_sub(1),
                        None => entries.len().checked_sub(1),
                    };
                    match upper.and_then(|u| search_back(entries, &query, u)) {
                        Some(i) => {
                            matched = Some(i);
                            failed = false;
                        }
                        None => failed = true,
                    }
                }
                BACKSPACE | DEL => {
                    // Shrink the query and re-search from the newest. An empty
                    // query is left empty (the candidate falls back to
                    // `start_line`); backspace on it is a quiet no-op.
                    if query.pop().is_some() {
                        matched = if query.is_empty() {
                            None
                        } else {
                            newest_match(entries, &query)
                        };
                        failed = false;
                    }
                }
                b if b < 0x20 => continue,
                b if b < 0x80 => extend(&mut query, &mut matched, &mut failed, entries, b as char),
                lead => {
                    if let Some(ch) = read_utf8(input, lead)? {
                        extend(&mut query, &mut matched, &mut failed, entries, ch);
                    }
                }
            },
        }
        render_search(out, &query, candidate(entries, matched, start_line), failed)?;
    }
}

/// The string a search renders and would accept: the matched entry, or
/// `start_line` while nothing is matched (empty query, or a query the last
/// keystroke failed to extend).
fn candidate<'a>(entries: &'a [String], matched: Option<usize>, start_line: &'a str) -> &'a str {
    match matched {
        Some(i) => entries[i].as_str(),
        None => start_line,
    }
}

/// Extend the query by `ch`, re-searching newest→oldest. A match commits the
/// character; no match anywhere refuses it — the query and current match are
/// left untouched and `failed` is raised, so the failing keystroke is dropped
/// rather than stranding the search on nothing.
fn extend(
    query: &mut String,
    matched: &mut Option<usize>,
    failed: &mut bool,
    entries: &[String],
    ch: char,
) {
    let mut extended = query.clone();
    extended.push(ch);
    match newest_match(entries, &extended) {
        Some(i) => {
            *query = extended;
            *matched = Some(i);
            *failed = false;
        }
        None => *failed = true,
    }
}

/// The newest entry containing `query` as a substring, or `None`.
fn newest_match(entries: &[String], query: &str) -> Option<usize> {
    entries
        .len()
        .checked_sub(1)
        .and_then(|upper| search_back(entries, query, upper))
}

/// The greatest index in `0..=upper` whose entry contains `query` — the
/// newest match at or before `upper`. `upper` must be a valid index.
fn search_back(entries: &[String], query: &str, upper: usize) -> Option<usize> {
    (0..=upper).rev().find(|&i| entries[i].contains(query))
}

/// Render the reverse-search row through the shared [`redraw`] seam: the
/// prompt is readline's `(reverse-i-search)\`query': ` (its `failed` form when
/// nothing matches), the candidate is the "line", cursor parked at its end.
fn render_search(
    out: &mut dyn Write,
    query: &str,
    candidate: &str,
    failed: bool,
) -> std::io::Result<()> {
    let label = if failed {
        "(failed reverse-i-search)"
    } else {
        "(reverse-i-search)"
    };
    let prompt = format!("{label}`{query}': ");
    redraw(out, &prompt, candidate, candidate.len())
}

/// The byte offset of the character before `cursor` (0 at the start).
fn prev_boundary(line: &str, cursor: usize) -> usize {
    line[..cursor]
        .char_indices()
        .next_back()
        .map_or(0, |(i, _)| i)
}

/// The byte offset just past the character at `cursor` (unchanged at the
/// end).
fn next_boundary(line: &str, cursor: usize) -> usize {
    line[cursor..]
        .chars()
        .next()
        .map_or(cursor, |ch| cursor + ch.len_utf8())
}

/// One word back: skip whitespace leftward, then the word it lands on —
/// readline's backward-word over whitespace-delimited words. The trim
/// helpers return shrinking prefixes of `line[..cursor]`, so their lengths
/// are valid char boundaries by construction.
fn word_left(line: &str, cursor: usize) -> usize {
    let head = line[..cursor].trim_end();
    head.trim_end_matches(|ch: char| !ch.is_whitespace()).len()
}

/// One word forward: skip whitespace rightward, then to the end of the word
/// it lands on.
fn word_right(line: &str, cursor: usize) -> usize {
    let tail = &line[cursor..];
    let after_ws = tail.len() - tail.trim_start().len();
    let rest = &tail[after_ws..];
    let word = rest.len()
        - rest
            .trim_start_matches(|ch: char| !ch.is_whitespace())
            .len();
    cursor + after_ws + word
}

/// Read one byte, retrying interrupted reads (a terminal read returns
/// `EINTR` on window resize). `None` when the source is dry.
fn next_byte(input: &mut dyn Read) -> std::io::Result<Option<u8>> {
    let mut byte = [0u8; 1];
    loop {
        match input.read(&mut byte) {
            Ok(0) => return Ok(None),
            Ok(_) => return Ok(Some(byte[0])),
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

/// Ceiling on collected CSI parameter bytes. The longest sequence the
/// editor recognizes carries three (`1;5`); anything past eight is noise
/// from a key this editor will never bind, discarded once its final byte
/// arrives.
const MAX_CSI_PARAMS: usize = 8;

/// Consume the remainder of an escape sequence whose lead `ESC` was already
/// read, recognizing the keys the editor binds and discarding everything
/// else whole.
///
/// `ESC [` (CSI) runs through its parameter bytes to a final byte in
/// `0x40..=0x7e`; [`csi_key`] maps the recognized (parameters, final) pairs
/// — plain arrows, Home/End in their `H`/`F` and `1~`/`7~`/`4~`/`8~`
/// spellings, and the Ctrl-/Alt-modified arrows (`1;5`/`1;3`) as word
/// motion. `ESC O` (SS3 — application-mode cursor keys, F1–F4) recognizes
/// `H`/`F` as Home/End and discards the rest. `ESC b`/`ESC f` are
/// readline's Alt-b/Alt-f word hops; any other Alt-chord byte is discarded
/// together with the ESC. A *bare* Escape press therefore blocks until the
/// next keystroke and consumes it: telling a lone ESC apart from the start
/// of a sequence needs a timed read (readline's ESC timeout), which the
/// plain byte-stream seam deliberately lacks — accepted for a key this
/// REPL assigns no meaning to. A source that dries up after the ESC just
/// drops it.
fn read_escape_sequence(input: &mut dyn Read) -> std::io::Result<Option<Key>> {
    match next_byte(input)? {
        Some(b'[') => {
            let mut params = Vec::new();
            let mut overlong = false;
            while let Some(byte) = next_byte(input)? {
                if (0x40..=0x7e).contains(&byte) {
                    return Ok(if overlong {
                        None
                    } else {
                        csi_key(&params, byte)
                    });
                }
                if params.len() < MAX_CSI_PARAMS {
                    params.push(byte);
                } else {
                    overlong = true;
                }
            }
            Ok(None)
        }
        Some(b'O') => Ok(match next_byte(input)? {
            Some(b'H') => Some(Key::Motion(Motion::Home)),
            Some(b'F') => Some(Key::Motion(Motion::End)),
            _ => None,
        }),
        Some(b'b') => Ok(Some(Key::Motion(Motion::WordLeft))),
        Some(b'f') => Ok(Some(Key::Motion(Motion::WordRight))),
        _ => Ok(None),
    }
}

/// Map a complete CSI sequence to the key it names, `None` for everything
/// the editor doesn't bind (modified arrows other than word motion, Delete,
/// function keys, …).
fn csi_key(params: &[u8], final_byte: u8) -> Option<Key> {
    match (params, final_byte) {
        ([], b'A') => Some(Key::Up),
        ([], b'B') => Some(Key::Down),
        ([], b'C') => Some(Key::Motion(Motion::Right)),
        ([], b'D') => Some(Key::Motion(Motion::Left)),
        ([], b'H') => Some(Key::Motion(Motion::Home)),
        ([], b'F') => Some(Key::Motion(Motion::End)),
        // Ctrl-arrow (1;5) and Alt-arrow (1;3): word motion.
        ([b'1', b';', b'5'] | [b'1', b';', b'3'], b'C') => Some(Key::Motion(Motion::WordRight)),
        ([b'1', b';', b'5'] | [b'1', b';', b'3'], b'D') => Some(Key::Motion(Motion::WordLeft)),
        // Home/End in their numbered spellings (vt220 1~/4~, rxvt 7~/8~).
        ([b'1'] | [b'7'], b'~') => Some(Key::Motion(Motion::Home)),
        ([b'4'] | [b'8'], b'~') => Some(Key::Motion(Motion::End)),
        _ => None,
    }
}

/// Collect the continuation bytes of a UTF-8 sequence led by `lead` and
/// decode the character. Malformed input — a stray continuation byte, an
/// invalid lead, a sequence that fails validation, a source drying up
/// mid-sequence — decodes to `None` and is dropped: keystrokes are not file
/// contents, so there is nothing for the project's lossy-decode policy to
/// preserve; the bytes consumed with a rejected sequence are discarded too.
fn read_utf8(input: &mut dyn Read, lead: u8) -> std::io::Result<Option<char>> {
    let len = match lead {
        0xc2..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf4 => 4,
        _ => return Ok(None),
    };
    let mut seq = [lead, 0, 0, 0];
    for slot in seq.iter_mut().take(len).skip(1) {
        match next_byte(input)? {
            Some(byte) => *slot = byte,
            None => return Ok(None),
        }
    }
    Ok(std::str::from_utf8(&seq[..len])
        .ok()
        .and_then(|s| s.chars().next()))
}

/// Write and flush: every keystroke's feedback must be visible before the
/// next blocking read.
fn echo(out: &mut dyn Write, bytes: &[u8]) -> std::io::Result<()> {
    out.write_all(bytes)?;
    out.flush()
}

/// Re-render an edit row in place: carriage return, erase to end of line
/// (`CSI K`), `prompt`, `line` — then hop the terminal cursor left to the
/// edit point (`CSI n D`), counting the characters right of `cursor`. The
/// hop is what makes mid-line editing visible; a cursor at the end needs
/// none, which keeps the end-of-line render byte-identical.
///
/// `prompt` is a parameter, not the hardcoded [`PROMPT`], so reverse search
/// can render its `(reverse-i-search)\`query': ` header through the very same
/// seam (the candidate is the "line", cursor at its end) — the normal prompt
/// still renders byte-for-byte as before.
fn redraw(out: &mut dyn Write, prompt: &str, line: &str, cursor: usize) -> std::io::Result<()> {
    write!(out, "\r\x1b[K{prompt}{line}")?;
    let tail = line[cursor..].chars().count();
    if tail > 0 {
        write!(out, "\x1b[{tail}D")?;
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A completer with nothing to offer — the shared default for scripts
    /// whose Tab (if any) must produce no candidates. A named fn rather than
    /// per-test closures: a closure a script never Tabs into would be a
    /// permanently-dead line under the coverage gate.
    fn no_candidates(line: &str) -> Completion {
        Completion {
            start: line.len(),
            candidates: Vec::new(),
        }
    }

    /// Drive [`edit_line`] over a byte script with `complete` and the recall
    /// `history` injected, returning `(result, finished_buf, rendered_output)`.
    fn edit_full(
        script: &[u8],
        complete: &mut dyn FnMut(&str) -> Completion,
        history: &mut Vec<String>,
    ) -> (std::io::Result<usize>, String, String) {
        let mut input = script;
        let mut out = Vec::new();
        let mut buf = String::new();
        let mut after_cr = false;
        let result = edit_line(
            &mut input,
            &mut out,
            &mut buf,
            complete,
            history,
            &mut after_cr,
        );
        (result, buf, String::from_utf8(out).unwrap())
    }

    /// Drive [`edit_line`] repeatedly over one byte source — the REPL's
    /// per-line loop — threading the cross-call `after_cr` flag, and collect
    /// every submitted line. A paste arrives as one stream the REPL reads a
    /// line at a time, so this is the only way to observe CRLF handling across
    /// the call boundary. Stops when the source dries up with nothing buffered.
    fn edit_lines(script: &[u8]) -> Vec<String> {
        let mut input = script;
        let mut out = Vec::new();
        let mut history = Vec::new();
        let mut after_cr = false;
        let mut submitted = Vec::new();
        loop {
            let mut buf = String::new();
            let n = edit_line(
                &mut input,
                &mut out,
                &mut buf,
                &mut no_candidates,
                &mut history,
                &mut after_cr,
            )
            .unwrap();
            if n == 0 && buf.is_empty() {
                break;
            }
            submitted.push(buf);
        }
        submitted
    }

    /// [`edit_full`] with a throwaway history.
    fn edit_with(
        script: &[u8],
        complete: &mut dyn FnMut(&str) -> Completion,
    ) -> (std::io::Result<usize>, String, String) {
        edit_full(script, complete, &mut Vec::new())
    }

    /// [`edit_full`] under the no-candidate completer, `history` injected —
    /// the recall tests' seam. Entries are `&str` for literal convenience.
    fn edit_recalling(script: &[u8], entries: &[&str]) -> (std::io::Result<usize>, String, String) {
        let mut history = entries.iter().map(|e| e.to_string()).collect();
        edit_full(script, &mut no_candidates, &mut history)
    }

    /// [`edit_with`] under the no-candidate completer.
    fn edit(script: &[u8]) -> (std::io::Result<usize>, String, String) {
        edit_with(script, &mut no_candidates)
    }

    #[test]
    fn typed_line_echoes_and_finishes_on_enter() {
        let (result, buf, out) = edit(b"hi\r");
        assert_eq!(result.unwrap(), 3);
        assert_eq!(buf, "hi\n");
        assert_eq!(out, "hi\r\n");
    }

    #[test]
    fn newline_finishes_like_carriage_return() {
        // Raw-mode Enter sends \r; \n finishes too so a source that already
        // speaks cooked newlines (a script, a paste) behaves identically.
        let (result, buf, _) = edit(b"ok\n");
        assert_eq!(result.unwrap(), 3);
        assert_eq!(buf, "ok\n");
    }

    #[test]
    fn crlf_paste_submits_once_per_line() {
        // A pasted `\r\n`-terminated block: the CR finishes each line and the
        // trailing LF is swallowed, so no spurious blank line lands between
        // them (finding F12). Two lines in, exactly two lines out.
        let submitted = edit_lines(b"a\r\nb\r\n");
        assert_eq!(submitted, ["a\n", "b\n"]);
    }

    #[test]
    fn lone_lf_and_cr_cr_still_submit() {
        // A CR not paired with a following LF still terminates: `a\r\rb\r`
        // yields "a", the empty line the second CR opens, then "b" — CR CR is
        // two submits, not one. The swallow only ever drops an LF right after
        // a CR, so a lone LF (see `newline_finishes_like_carriage_return`)
        // also still submits.
        let submitted = edit_lines(b"a\r\rb\r");
        assert_eq!(submitted, ["a\n", "\n", "b\n"]);
    }

    #[test]
    fn dry_source_with_nothing_typed_is_eof() {
        let (result, buf, out) = edit(b"");
        assert_eq!(result.unwrap(), 0);
        assert_eq!(buf, "");
        assert_eq!(out, "");
    }

    #[test]
    fn dry_source_mid_line_yields_the_partial_line() {
        // Cooked read_line's EOF contract: the typed text, no newline.
        let (result, buf, _) = edit(b"hi");
        assert_eq!(result.unwrap(), 2);
        assert_eq!(buf, "hi");
    }

    #[test]
    fn ctrl_d_on_empty_line_is_eof() {
        let (result, buf, out) = edit(b"\x04never read");
        assert_eq!(result.unwrap(), 0);
        assert_eq!(buf, "");
        // The caller's next output starts on a fresh row.
        assert_eq!(out, "\r\n");
    }

    #[test]
    fn ctrl_d_mid_line_is_ignored() {
        let (result, buf, _) = edit(b"hi\x04!\r");
        assert_eq!(result.unwrap(), 4);
        assert_eq!(buf, "hi!\n");
    }

    #[test]
    fn ctrl_c_cancels_the_line() {
        // The typed text is discarded; a bare newline comes back so the
        // caller re-prompts instead of treating the cancel as EOF.
        let (result, buf, out) = edit(b"abc\x03");
        assert_eq!(result.unwrap(), 1);
        assert_eq!(buf, "\n");
        assert_eq!(out, "abc^C\r\n");
    }

    #[test]
    fn backspace_removes_the_last_char_and_redraws() {
        let (result, buf, out) = edit(b"ab\x7fc\r");
        assert_eq!(result.unwrap(), 3);
        assert_eq!(buf, "ac\n");
        // The redraw re-renders prompt + remaining buffer in place.
        assert!(out.contains("\r\x1b[K> a"));
    }

    #[test]
    fn backspace_byte_and_del_byte_both_erase() {
        let (_, buf, _) = edit(b"ab\x08c\r");
        assert_eq!(buf, "ac\n");
    }

    #[test]
    fn backspace_on_an_empty_line_does_nothing() {
        let (result, buf, out) = edit(b"\x7fa\r");
        assert_eq!(result.unwrap(), 2);
        assert_eq!(buf, "a\n");
        // No redraw fired for the empty pop.
        assert!(!out.contains("\x1b[K"));
    }

    #[test]
    fn multibyte_input_appends_whole_chars() {
        // é (2 bytes), ✓ (3 bytes), 🦀 (4 bytes) — echoed byte-for-byte.
        let script = "é✓🦀\r".as_bytes();
        let (result, buf, out) = edit(script);
        assert_eq!(result.unwrap(), "é✓🦀\n".len());
        assert_eq!(buf, "é✓🦀\n");
        assert_eq!(out, "é✓🦀\r\n");
    }

    #[test]
    fn backspace_removes_a_whole_multibyte_char() {
        let (_, buf, _) = edit("é\u{7f}x\r".as_bytes());
        assert_eq!(buf, "x\n");
    }

    #[test]
    fn invalid_utf8_is_discarded() {
        // A stray continuation byte (0x80), an overlong-encoding lead
        // (0xc0), and a lead whose continuation is malformed (0xc3 then '(')
        // are all dropped — the '(' is consumed with its rejected sequence.
        let (_, buf, _) = edit(b"a\x80\xc0b\xc3(c\r");
        assert_eq!(buf, "abc\n");
    }

    #[test]
    fn dry_source_mid_utf8_sequence_drops_the_partial_char() {
        let (result, buf, _) = edit(b"a\xc3");
        assert_eq!(result.unwrap(), 1);
        assert_eq!(buf, "a");
    }

    #[test]
    fn unbound_escape_sequences_are_consumed_and_discarded() {
        // Shift-Tab (ESC [ Z), Delete (ESC [ 3 ~), a modified arrow the
        // editor doesn't bind (Ctrl-Up: ESC [ 1 ; 5 A), an unbound SS3 key
        // (ESC O P — F1), and an Alt-chord (ESC x) each vanish without
        // leaving stray bytes in the line.
        let (_, buf, _) = edit(b"a\x1b[Zb\x1b[3~c\x1b[1;5Ad\x1bOPe\x1bxf\r");
        assert_eq!(buf, "abcdef\n");
    }

    #[test]
    fn overlong_csi_parameters_disqualify_the_sequence() {
        // A parameter run past the cap is consumed to its final byte and
        // discarded — even when that final byte would otherwise spell a
        // bound key (D = Left). The cursor stays put: 'c' appends at the
        // end rather than landing inside "ab".
        let (_, buf, out) = edit(b"ab\x1b[123456789;5Dc\r");
        assert_eq!(buf, "abc\n");
        assert!(!out.contains("\x1b[K"), "no motion redraw must fire");
    }

    #[test]
    fn dry_source_inside_an_escape_sequence_ends_the_line() {
        // A lone trailing ESC, and a CSI the source dries up inside — both
        // leave the typed text intact.
        let (_, buf, _) = edit(b"a\x1b");
        assert_eq!(buf, "a");
        let (_, buf, _) = edit(b"b\x1b[");
        assert_eq!(buf, "b");
    }

    #[test]
    fn unbound_control_bytes_are_discarded() {
        // ^K and ^Z have no binding; the bell byte is input noise too.
        let (_, buf, _) = edit(b"a\x0bb\x1ac\x07\r");
        assert_eq!(buf, "abc\n");
    }

    #[test]
    fn tab_with_no_candidates_does_nothing() {
        let (result, buf, out) = edit(b"x\t!\r");
        assert_eq!(result.unwrap(), 3);
        assert_eq!(buf, "x!\n");
        assert!(!out.contains("\x1b[K"));
    }

    #[test]
    fn tab_with_a_unique_candidate_completes_and_redraws() {
        // The completer sees the buffer as typed; its candidate replaces
        // line[start..].
        let mut complete = |line: &str| {
            assert_eq!(line, "/mo");
            Completion {
                start: 0,
                candidates: vec!["/model".to_string()],
            }
        };
        let (result, buf, out) = edit_with(b"/mo\t\r", &mut complete);
        assert_eq!(result.unwrap(), 7);
        assert_eq!(buf, "/model\n");
        assert!(out.contains("\r\x1b[K> /model"));
    }

    #[test]
    fn tab_completion_replaces_only_the_token_after_start() {
        // A mid-line start keeps the prefix: the model id completes after
        // the provider colon.
        let mut complete = |_: &str| Completion {
            start: 17,
            candidates: vec!["claude-sonnet-4-6".to_string()],
        };
        let (_, buf, _) = edit_with(b"/model anthropic:cl\t\r", &mut complete);
        assert_eq!(buf, "/model anthropic:claude-sonnet-4-6\n");
    }

    #[test]
    fn tab_with_several_candidates_lists_them_and_redraws() {
        let mut complete = |_: &str| Completion {
            start: 0,
            candidates: vec!["/model".to_string(), "/quit".to_string()],
        };
        let (result, buf, out) = edit_with(b"/\t\r", &mut complete);
        // The line itself is untouched — the operator disambiguates.
        assert_eq!(result.unwrap(), 2);
        assert_eq!(buf, "/\n");
        // Candidates print on their own row, then the line re-renders.
        assert!(out.contains("\r\n/model  /quit\r\n"));
        assert!(out.contains("\r\x1b[K> /"));
    }

    #[test]
    fn tab_completion_moves_the_cursor_to_the_end() {
        // Completion is whole-line, wherever the cursor sat: after a Left,
        // the unique candidate still replaces the token and typing resumes
        // at the end.
        let mut complete = |line: &str| {
            assert_eq!(line, "/mo");
            Completion {
                start: 0,
                candidates: vec!["/model".to_string()],
            }
        };
        let (_, buf, _) = edit_with(b"/mo\x1b[D\t!\r", &mut complete);
        assert_eq!(buf, "/model!\n");
    }

    #[test]
    fn interrupted_reads_are_retried() {
        // EINTR (a window resize mid-read) must not end the line.
        struct Interrupting<'a> {
            interrupted: bool,
            rest: &'a [u8],
        }
        impl Read for Interrupting<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if !self.interrupted {
                    self.interrupted = true;
                    return Err(std::io::Error::from(ErrorKind::Interrupted));
                }
                self.rest.read(out)
            }
        }
        let mut input = Interrupting {
            interrupted: false,
            rest: b"ok\r",
        };
        let mut buf = String::new();
        let result = edit_line(
            &mut input,
            &mut Vec::new(),
            &mut buf,
            &mut no_candidates,
            &mut Vec::new(),
            &mut false,
        );
        assert_eq!(result.unwrap(), 3);
        assert_eq!(buf, "ok\n");
    }

    #[test]
    fn read_errors_propagate() {
        struct FailingReader;
        impl Read for FailingReader {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("terminal vanished"))
            }
        }
        let mut buf = String::new();
        let result = edit_line(
            &mut FailingReader,
            &mut Vec::new(),
            &mut buf,
            &mut no_candidates,
            &mut Vec::new(),
            &mut false,
        );
        assert_eq!(result.unwrap_err().to_string(), "terminal vanished");
    }

    #[test]
    fn write_errors_propagate() {
        // Fails at flush, not write, so both trait methods execute — a
        // write-failure double would leave its flush body permanently dead.
        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::other("stdout gone"))
            }
        }
        let mut buf = String::new();
        let result = edit_line(
            &mut b"a".as_slice(),
            &mut FailingWriter,
            &mut buf,
            &mut no_candidates,
            &mut Vec::new(),
            &mut false,
        );
        assert_eq!(result.unwrap_err().to_string(), "stdout gone");
    }

    // ── cursor motion & mid-line editing ──

    #[test]
    fn left_then_typing_inserts_at_the_cursor() {
        let (result, buf, out) = edit(b"ac\x1b[Db\r");
        assert_eq!(result.unwrap(), 4);
        assert_eq!(buf, "abc\n");
        // Both the motion and the insertion redraw with a one-char hop back
        // to the edit point.
        assert!(out.contains("\r\x1b[K> ac\x1b[1D"));
        assert!(out.contains("\r\x1b[K> abc\x1b[1D"));
    }

    #[test]
    fn left_at_the_start_is_a_quiet_no_op() {
        let (_, buf, out) = edit(b"\x1b[Da\r");
        assert_eq!(buf, "a\n");
        assert!(!out.contains("\x1b[K"));
    }

    #[test]
    fn right_moves_back_over_a_left() {
        // Left then Right returns to the end; the next char appends on the
        // fast path.
        let (_, buf, _) = edit(b"ab\x1b[D\x1b[Cc\r");
        assert_eq!(buf, "abc\n");
    }

    #[test]
    fn right_at_the_end_is_a_quiet_no_op() {
        let (_, buf, out) = edit(b"a\x1b[Cb\r");
        assert_eq!(buf, "ab\n");
        assert!(!out.contains("\x1b[K"));
    }

    #[test]
    fn home_and_end_keys_jump_the_line() {
        // ESC [ H to the start (insert lands first), ESC [ F back to the
        // end (insert lands last).
        let (_, buf, _) = edit(b"bc\x1b[Ha\x1b[Fd\r");
        assert_eq!(buf, "abcd\n");
    }

    #[test]
    fn home_and_end_numbered_spellings_jump_too() {
        // vt220's 1~/4~ and rxvt's 7~/8~.
        let (_, buf, _) = edit(b"bc\x1b[1~a\x1b[4~d\r");
        assert_eq!(buf, "abcd\n");
        let (_, buf, _) = edit(b"bc\x1b[7~a\x1b[8~d\r");
        assert_eq!(buf, "abcd\n");
    }

    #[test]
    fn ss3_home_and_end_jump_too() {
        // Application-mode Home/End (ESC O H / ESC O F).
        let (_, buf, _) = edit(b"bc\x1bOHa\x1bOFd\r");
        assert_eq!(buf, "abcd\n");
    }

    #[test]
    fn ctrl_a_and_ctrl_e_jump_the_line() {
        let (_, buf, _) = edit(b"bc\x01a\x05d\r");
        assert_eq!(buf, "abcd\n");
    }

    #[test]
    fn ctrl_b_and_ctrl_f_step_by_character() {
        // a·c, Ctrl-B before 'c', insert 'b', Ctrl-F past 'c', append 'd'.
        let (_, buf, _) = edit(b"ac\x02b\x06d\r");
        assert_eq!(buf, "abcd\n");
    }

    #[test]
    fn home_at_the_start_is_a_quiet_no_op() {
        let (_, buf, out) = edit(b"\x01a\r");
        assert_eq!(buf, "a\n");
        assert!(!out.contains("\x1b[K"));
    }

    #[test]
    fn backspace_mid_line_removes_before_the_cursor() {
        // Cursor left of 'c': backspace removes 'b', not 'c'.
        let (_, buf, _) = edit(b"abc\x1b[D\x7f\r");
        assert_eq!(buf, "ac\n");
    }

    #[test]
    fn backspace_at_the_start_of_a_full_line_does_nothing() {
        let (_, buf, out) = edit(b"ab\x01\x7fc\r");
        assert_eq!(buf, "cab\n");
        // The Home redraw fired; no further redraw for the dead backspace.
        assert_eq!(out.matches("\x1b[K").count(), 2);
    }

    #[test]
    fn word_left_hops_to_the_word_start() {
        // Ctrl-Left (1;5D), Alt-Left (1;3D), and Alt-b all land at the
        // start of "two"; the insert marks where the cursor stopped.
        for script in [
            b"one two\x1b[1;5DX\r".as_slice(),
            b"one two\x1b[1;3DX\r",
            b"one two\x1bbX\r",
        ] {
            let (_, buf, _) = edit(script);
            assert_eq!(buf, "one Xtwo\n");
        }
    }

    #[test]
    fn word_left_skips_trailing_whitespace_first() {
        let (_, buf, _) = edit(b"ab  \x1b[1;5DX\r");
        assert_eq!(buf, "Xab  \n");
    }

    #[test]
    fn word_right_hops_past_the_word_end() {
        // From Home, Ctrl-Right (1;5C), Alt-Right (1;3C), and Alt-f all
        // stop just after "one".
        for script in [
            b"one two\x01\x1b[1;5C!\r".as_slice(),
            b"one two\x01\x1b[1;3C!\r",
            b"one two\x01\x1bf!\r",
        ] {
            let (_, buf, _) = edit(script);
            assert_eq!(buf, "one! two\n");
        }
    }

    #[test]
    fn word_right_skips_leading_whitespace_first() {
        // From the start of "  ab", word-right lands after "ab".
        let (_, buf, _) = edit(b"  ab\x01\x1b[1;5C!\r");
        assert_eq!(buf, "  ab!\n");
    }

    #[test]
    fn word_motion_at_the_line_edges_is_a_quiet_no_op() {
        let (_, buf, out) = edit(b"ab\x1b[1;5Cc\r");
        assert_eq!(buf, "abc\n");
        assert!(
            !out.contains("\x1b[K"),
            "word-right at the end must not redraw"
        );
        let (_, buf, out) = edit(b"\x1b[1;5Da\r");
        assert_eq!(buf, "a\n");
        assert!(
            !out.contains("\x1b[K"),
            "word-left at the start must not redraw"
        );
    }

    #[test]
    fn cursor_steps_whole_multibyte_chars() {
        // Two Lefts cross 'x' then 'é' as whole characters; the insert
        // lands before both.
        let (_, buf, _) = edit("éx\u{1b}[D\u{1b}[Da\r".as_bytes());
        assert_eq!(buf, "aéx\n");
    }

    #[test]
    fn cursor_hop_counts_characters_not_bytes() {
        // With the two-byte é right of the cursor, the hop is one column.
        let (_, _, out) = edit("é\u{01}a\r".as_bytes());
        assert!(out.contains("\x1b[1D"), "got: {out:?}");
        assert!(!out.contains("\x1b[2D"), "got: {out:?}");
    }

    #[test]
    fn enter_mid_line_submits_the_whole_line() {
        let mut history = Vec::new();
        let (_, buf, _) = edit_full(b"abc\x01\r", &mut no_candidates, &mut history);
        assert_eq!(buf, "abc\n");
        assert_eq!(history, ["abc"]);
    }

    // ── Up/Down history recall ──

    #[test]
    fn up_recalls_the_previous_line() {
        let (result, buf, out) = edit_recalling(b"\x1b[A\r", &["first"]);
        assert_eq!(result.unwrap(), 6);
        assert_eq!(buf, "first\n");
        // The recall renders through the in-place redraw.
        assert!(out.contains("\r\x1b[K> first"));
    }

    #[test]
    fn repeated_up_steps_further_back() {
        let (_, buf, _) = edit_recalling(b"\x1b[A\x1b[A\r", &["one", "two"]);
        assert_eq!(buf, "one\n");
    }

    #[test]
    fn up_at_the_oldest_entry_stays_put() {
        let (_, buf, out) = edit_recalling(b"\x1b[A\x1b[A\r", &["only"]);
        assert_eq!(buf, "only\n");
        // The second Up neither moved nor redrew.
        assert_eq!(out.matches("\x1b[K").count(), 1);
    }

    #[test]
    fn down_steps_forward_through_the_entries() {
        let (_, buf, _) = edit_recalling(b"\x1b[A\x1b[A\x1b[B\r", &["one", "two"]);
        assert_eq!(buf, "two\n");
    }

    #[test]
    fn down_past_the_newest_entry_restores_the_draft() {
        // "dra" is in progress when Up recalls; Down steps back past the
        // newest entry and the partial line comes back, rendered in place.
        let (_, buf, out) = edit_recalling(b"dra\x1b[A\x1b[B\r", &["old"]);
        assert_eq!(buf, "dra\n");
        assert!(out.contains("\r\x1b[K> old"));
        assert!(out.contains("\r\x1b[K> dra"));
    }

    #[test]
    fn recall_places_the_cursor_at_the_end() {
        // Typing right after a recall appends — recall semantics are
        // untouched by the cursor model.
        let (_, buf, _) = edit_recalling(b"\x1b[A!\r", &["cmd"]);
        assert_eq!(buf, "cmd!\n");
    }

    #[test]
    fn draft_restore_places_the_cursor_at_the_end() {
        // Mid-line cursor when recall starts; stepping back to the draft
        // resumes typing at its end.
        let (_, buf, _) = edit_recalling(b"dr\x1b[A\x1b[Baft\r", &["old"]);
        assert_eq!(buf, "draft\n");
    }

    #[test]
    fn up_with_empty_history_is_a_no_op() {
        let (_, buf, out) = edit(b"a\x1b[Ab\r");
        assert_eq!(buf, "ab\n");
        assert!(!out.contains("\x1b[K"));
    }

    #[test]
    fn down_without_active_recall_is_a_no_op() {
        let (_, buf, out) = edit_recalling(b"a\x1b[Bb\r", &["never shown"]);
        assert_eq!(buf, "ab\n");
        assert!(!out.contains("\x1b[K"));
    }

    #[test]
    fn submitted_lines_accumulate_across_invocations_and_recall() {
        // Two edit_line calls over one history — the session-scoped store —
        // then a third recalls the newest entry.
        let mut history = Vec::new();
        edit_full(b"first\r", &mut no_candidates, &mut history)
            .0
            .unwrap();
        edit_full(b"second\r", &mut no_candidates, &mut history)
            .0
            .unwrap();
        assert_eq!(history, ["first", "second"]);
        let (_, buf, _) = edit_full(b"\x1b[A\r", &mut no_candidates, &mut history);
        assert_eq!(buf, "second\n");
    }

    #[test]
    fn consecutive_duplicates_are_stored_once() {
        // A resubmitted newest entry is suppressed; a non-consecutive
        // duplicate is a distinct step and is kept.
        let mut history = Vec::new();
        for script in [b"same\r".as_slice(), b"same\r", b"other\r", b"same\r"] {
            edit_full(script, &mut no_candidates, &mut history)
                .0
                .unwrap();
        }
        assert_eq!(history, ["same", "other", "same"]);
    }

    #[test]
    fn blank_lines_are_not_stored() {
        let mut history = Vec::new();
        edit_full(b"\r", &mut no_candidates, &mut history)
            .0
            .unwrap();
        edit_full(b"   \r", &mut no_candidates, &mut history)
            .0
            .unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn ctrl_c_cancels_a_recalled_line_without_storing() {
        let mut history = vec!["kept".to_string()];
        let (_, buf, _) = edit_full(b"\x1b[A\x03", &mut no_candidates, &mut history);
        assert_eq!(buf, "\n");
        assert_eq!(history, ["kept"]);
    }

    #[test]
    fn backspace_edits_a_recalled_line() {
        let (_, buf, _) = edit_recalling(b"\x1b[A\x7f\r", &["abc"]);
        assert_eq!(buf, "ab\n");
    }

    #[test]
    fn tab_completes_a_recalled_line() {
        let mut complete = |line: &str| {
            assert_eq!(line, "/mo");
            Completion {
                start: 0,
                candidates: vec!["/model".to_string()],
            }
        };
        let mut history = vec!["/mo".to_string()];
        let (_, buf, _) = edit_full(b"\x1b[A\t\r", &mut complete, &mut history);
        assert_eq!(buf, "/model\n");
    }

    #[test]
    fn mid_line_edit_of_a_recalled_line_belongs_to_the_line() {
        // Recall, hop a word left, edit — the store keeps the original.
        let mut history = vec!["run tests".to_string()];
        let (_, buf, _) = edit_full(b"\x1b[A\x1b[1;5Dall \r", &mut no_candidates, &mut history);
        assert_eq!(buf, "run all tests\n");
        assert_eq!(history, ["run tests", "run all tests"]);
    }

    // ── Ctrl-R reverse incremental search ──

    #[test]
    fn ctrl_r_search_submits_the_match_and_pins_the_search_prompt() {
        // Ctrl-R, a query, Enter: the substring match is submitted whole.
        let (result, buf, out) = edit_recalling(b"\x12ls\r", &["ls -la"]);
        assert_eq!(result.unwrap(), 7);
        assert_eq!(buf, "ls -la\n");
        // The search row renders through the generalized redraw seam: the
        // readline header is the "prompt", the candidate the "line".
        assert!(
            out.contains("\r\x1b[K(reverse-i-search)`ls': ls -la"),
            "got: {out:?}"
        );
        // The empty-query entry render shows the header with no candidate yet.
        assert!(
            out.contains("\r\x1b[K(reverse-i-search)`': "),
            "got: {out:?}"
        );
        // On accept the row flips back to the normal prompt before the newline.
        assert!(out.contains("\r\x1b[K> ls -la\r\n"), "got: {out:?}");
    }

    #[test]
    fn ctrl_r_submits_on_a_bare_newline_too() {
        // A `\n` terminator accepts the match exactly like `\r`.
        let (_, buf, _) = edit_recalling(b"\x12fo\n", &["foo"]);
        assert_eq!(buf, "foo\n");
    }

    #[test]
    fn ctrl_r_steps_to_older_matches_then_fails_past_the_oldest() {
        // "git" matches both git commands newest-first; each further Ctrl-R
        // steps older, and a step past the oldest match flags `failed` while
        // leaving the current candidate in place.
        let (_, buf, out) = edit_recalling(
            b"\x12git\x12\x12\r",
            &["git status", "git commit", "grep foo"],
        );
        assert_eq!(buf, "git status\n");
        assert!(out.contains("(failed reverse-i-search)"), "got: {out:?}");
    }

    #[test]
    fn ctrl_r_backspace_shrinks_and_researches_from_the_newest() {
        // "app" only matches "apple"; backspacing to "ap" re-matches the newer
        // "grape", proving the shrink re-searched rather than kept the index.
        let (_, buf, _) = edit_recalling(b"\x12app\x7f\r", &["apple", "grape"]);
        assert_eq!(buf, "grape\n");
    }

    #[test]
    fn ctrl_r_backspace_to_empty_then_backspace_again_is_a_no_op() {
        // Shrinking to an empty query drops the match (candidate falls back to
        // the pre-search line); a backspace on the empty query does nothing,
        // and typing re-searches from scratch.
        let (_, buf, _) = edit_recalling(b"\x12a\x7f\x7fb\r", &["abc"]);
        assert_eq!(buf, "abc\n");
    }

    #[test]
    fn ctrl_r_accept_via_arrow_adopts_the_match_and_applies_the_motion() {
        // A Left arrow ends the search, adopts "grep foo" into the buffer, and
        // moves one char left; the inserted 'X' lands before the final 'o'.
        let (_, buf, out) = edit_recalling(b"\x12gr\x1b[DX\r", &["grep foo"]);
        assert_eq!(buf, "grep foXo\n");
        // The adopted line renders under the normal prompt, then the motion
        // hops the cursor one column left.
        assert!(out.contains("\r\x1b[K> grep foo"), "got: {out:?}");
        assert!(out.contains("\x1b[1D"), "got: {out:?}");
    }

    #[test]
    fn ctrl_r_accept_via_unbound_escape_adopts_without_a_motion() {
        // A complete-but-unbound sequence (Shift-Tab) still ends the search and
        // adopts the match; no motion follows, so typing appends at the end.
        let (_, buf, _) = edit_recalling(b"\x12ru\x1b[Z!\r", &["run"]);
        assert_eq!(buf, "run!\n");
    }

    #[test]
    fn ctrl_r_cancel_via_ctrl_g_restores_the_pre_search_line() {
        // Ctrl-G backs out of the search to the line as it stood, cursor intact,
        // and editing resumes — unlike a top-level Ctrl-C, the line survives.
        let (_, buf, out) = edit_recalling(b"abc\x12c\x07d\r", &["cmd"]);
        assert_eq!(buf, "abcd\n");
        assert!(out.contains("\r\x1b[K> abc"), "got: {out:?}");
    }

    #[test]
    fn ctrl_r_cancel_via_ctrl_c_restores_the_pre_search_line() {
        // Ctrl-C inside search cancels the search only, not the whole line.
        let (_, buf, _) = edit_recalling(b"abc\x12c\x03d\r", &["cmd"]);
        assert_eq!(buf, "abcd\n");
    }

    #[test]
    fn ctrl_r_then_eof_abandons_the_search() {
        // A source that dries up inside the search backs out to the pre-search
        // line, which the dry main loop then yields as a partial line.
        let (result, buf, _) = edit_recalling(b"hi\x12", &["history"]);
        assert_eq!(result.unwrap(), 2);
        assert_eq!(buf, "hi");
    }

    #[test]
    fn ctrl_r_no_match_shows_failed_and_refuses_to_extend() {
        // 'z' matches nothing: the query is left empty and `failed` shown, so
        // the following 'h' extends "" (not "z") and matches "hello".
        let (_, buf, out) = edit_recalling(b"\x12zh\r", &["hello"]);
        assert_eq!(buf, "hello\n");
        assert!(out.contains("(failed reverse-i-search)"), "got: {out:?}");
    }

    #[test]
    fn ctrl_r_on_empty_history_finds_nothing() {
        // Nothing to match: an extension fails, and a step has no newest entry
        // to seek from. Both submit the (empty) pre-search line.
        let (_, buf, _) = edit(b"\x12x\r");
        assert_eq!(buf, "\n");
        let (_, buf, out) = edit(b"\x12\x12\r");
        assert_eq!(buf, "\n");
        assert!(out.contains("(reverse-i-search)"), "got: {out:?}");
    }

    #[test]
    fn ctrl_r_ignores_unbound_control_bytes() {
        // A Tab inside search has no binding and is dropped; the query is
        // built from the printable bytes alone.
        let (_, buf, _) = edit_recalling(b"\x12\tc\r", &["cmd"]);
        assert_eq!(buf, "cmd\n");
    }

    #[test]
    fn ctrl_r_search_handles_utf8_in_query_and_candidate() {
        // Multi-byte input extends the query; the matched entry carries its own
        // multi-byte glyphs through unharmed.
        let (_, buf, _) = edit_recalling("\u{12}café\r".as_bytes(), &["héllo café"]);
        assert_eq!(buf, "héllo café\n");
    }

    #[test]
    fn tab_completion_works_after_an_accepted_search() {
        // Accept "/mo" via a (no-op) Right arrow, then Tab completes it — the
        // editor is back in its normal mode, seam and all.
        let mut complete = |line: &str| {
            assert_eq!(line, "/mo");
            Completion {
                start: 0,
                candidates: vec!["/model".to_string()],
            }
        };
        let mut history = vec!["/mo".to_string()];
        let (_, buf, _) = edit_full(b"\x12/mo\x1b[C\t\r", &mut complete, &mut history);
        assert_eq!(buf, "/model\n");
    }

    #[test]
    fn cursor_motion_works_after_an_accepted_search() {
        // Accept "hello", then Ctrl-A/Ctrl-E jump the adopted line's edges and
        // the inserts land at each end.
        let (_, buf, _) = edit_recalling(b"\x12he\x1b[C\x01X\x05Y\r", &["hello"]);
        assert_eq!(buf, "XhelloY\n");
    }

    #[test]
    fn up_down_recall_works_after_an_accepted_search() {
        // Accepting a match starts recall over: the adopted line becomes the
        // draft, and Up walks the store from the newest as usual.
        let (_, buf, _) = edit_recalling(b"\x12be\x1b[C\x1b[A\x1b[A\r", &["alpha", "beta"]);
        assert_eq!(buf, "alpha\n");
    }
}

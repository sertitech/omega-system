//! Minimal Markdown → ANSI renderer for streamed model text.
//!
//! Models emit GitHub-flavored Markdown by habit; on a TTY the raw markers
//! (`**`, `##`, backticks) read as noise. This module renders a deliberately
//! small subset — headings, bold, italics, inline code, list bullets, fenced
//! code blocks, pipe tables — to SGR escape sequences. Everything else passes
//! through verbatim, and unmatched markers stay literal, so imperfect
//! Markdown never corrupts the text.
//!
//! Rendering streams: every byte is written the moment its styling is
//! decided, not when its line completes. Prose, heading text, bullet items,
//! and fenced code reach the terminal as they arrive; only bytes whose
//! meaning depends on bytes not yet seen are held back. A potential span
//! opener (`` ` ``, `*`, `_`) holds from the marker until its closer arrives
//! or the line ends — a span cannot be styled until it provably closes, and
//! models don't hard-wrap, so waiting for the newline would mean waiting for
//! the whole paragraph. Line shapes (heading, bullet, fence, table row) are
//! decided from a line's first unambiguous characters. Whatever is still
//! held when the newline (or the stream's tail) arrives renders through the
//! same complete-line rules the incremental scanner mirrors, so a split
//! marker (`**bo` … `ld**`) renders exactly like a one-push line.
//!
//! Tables are the one construct that still buffers whole: a pipe row means
//! nothing until the next line proves it a header (separator row) or not,
//! and column widths need every row, so a confirmed table renders only at
//! its first non-row line. Malformed or oversized tables flush as ordinary
//! lines — fail-open, like unmatched markers.

use crate::display::scrub_controls;
use std::io::Write;

const BOLD: &str = "\x1b[1m";
const ITALIC: &str = "\x1b[3m";
const HEADING: &str = "\x1b[1;4m";
const CODE: &str = "\x1b[36m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Cap on the raw bytes buffered for one table. A table renders only when a
/// non-row line (or the end of the stream) closes it, so without a bound a
/// pathological unterminated run of pipe rows would buffer forever. 64 KiB
/// is hundreds of realistic rows — beyond any table a model plausibly emits
/// — while keeping worst-case memory trivial. A row that would push the
/// buffer past the cap flushes the table (aligned, if already confirmed) and
/// itself renders as an ordinary line: fail-open, never truncation.
const TABLE_MAX_BYTES: usize = 64 * 1024;

/// Cap on the raw bytes held for one unresolved inline span. A span opener
/// (`` ` ``, `*`, `_`) holds from its marker until the closer or the line
/// ends; without a bound, an opener that never closes would make every later
/// delta re-scan an ever-growing tail — O(n²). 64 KiB is far past any single
/// line a model plausibly emits (models don't hard-wrap, but one paragraph is
/// still only kilobytes), so crossing it means the opener will not close:
/// flush the held bytes literally and start fresh — fail-open, like an
/// oversized table, never truncation.
const INLINE_MAX_BYTES: usize = 64 * 1024;

/// What the renderer knows about the line in flight.
#[derive(Clone, Copy, Default)]
enum Line {
    /// At a line start, shape undecided: nothing written yet; `pending`
    /// holds the still-ambiguous prefix.
    #[default]
    Start,
    /// A `|`-opened line: a table-row candidate, buffered whole — a row
    /// means nothing until its own line (and the line after it) completes.
    Table,
    /// Inline-styled content — plain prose, a heading's text, a bullet's
    /// item — streaming span by span. `base` is the enclosing style each
    /// span restores, `reset` closes the line's own style at its end, and
    /// `prev` is the character just before `pending`, feeding the italic
    /// flanking rules.
    Inline {
        base: &'static str,
        reset: bool,
        prev: Option<char>,
    },
    /// A fence marker or fenced-content line: styled at classification,
    /// streaming verbatim, closed with a reset at the line's end.
    Fence,
}

/// A line shape, decided from the first unambiguous characters.
enum Shape {
    Plain,
    /// Heading: skip this many bytes (the hashes and their space).
    Heading(usize),
    /// Bullet: the indent's byte length; marker and space are replaced.
    Bullet(usize),
    /// A ``` line: dimmed whole, toggling fence state.
    Marker,
    /// Any other line inside a fence: code-styled, verbatim.
    Code,
    /// A `|`-opened line: routed to the table machinery.
    Row,
}

/// Streaming renderer. Feed chunks with [`push`](Self::push); flush the
/// held tail with [`finish`](Self::finish). Fence state persists across
/// pushes, so a code block spanning many deltas stays a code block.
#[derive(Default)]
pub struct MarkdownRenderer {
    /// Bytes of the current line not yet written: at a line start the
    /// unclassified prefix, mid-line an unresolved opener's holdback, for
    /// a table candidate the whole row so far.
    pending: String,
    /// The current line's decided shape.
    line: Line,
    /// Whether the last write left the cursor mid-line — an abandoned
    /// stream must close that line before anything else prints.
    dirty: bool,
    /// Whether the cursor sits inside a ``` fence.
    in_fence: bool,
    /// Buffered raw table rows: one pipe row awaiting its separator, or a
    /// confirmed table (header + separator at least) awaiting its end.
    table: Vec<String>,
    /// Total bytes in `table`, checked against [`TABLE_MAX_BYTES`].
    table_bytes: usize,
}

impl MarkdownRenderer {
    /// Feed a chunk of streamed text: everything already decidable renders
    /// immediately; span openers hold until they resolve, table rows until
    /// the table closes.
    pub fn push(&mut self, chunk: &str, out: &mut dyn Write) {
        // Scrub on the way in, so every downstream path — plain-text
        // passthrough, code spans, fenced blocks — sees only clean bytes and
        // no model-supplied control byte can reach `out`.
        self.pending.push_str(&scrub_controls(chunk));
        let buf = self.drain();
        self.write(&buf, out);
    }

    /// Flush everything still held: an unresolved span renders by the
    /// complete-line rules, a pending table flushes (its rows arrived
    /// newline-terminated and keep their newlines), and the line's style
    /// closes — all without adding a newline; the caller decides how the
    /// stream's tail terminates.
    pub fn finish(&mut self, out: &mut dyn Write) {
        let pending = std::mem::take(&mut self.pending);
        let mut buf = String::new();
        match self.line {
            Line::Start | Line::Table => {
                // Nothing of this line has been written: the complete-line
                // path renders it whole (possibly extending the table).
                let mut lines = Vec::new();
                if !pending.is_empty() {
                    lines = self.process_line(&pending);
                }
                lines.append(&mut self.flush_table());
                if let Some(last) = lines.pop() {
                    for line in &lines {
                        buf.push_str(line);
                        buf.push('\n');
                    }
                    buf.push_str(&last);
                    if pending.is_empty() {
                        buf.push('\n');
                    }
                }
            }
            Line::Inline { base, reset, prev } => {
                buf.push_str(&inline_from(&pending, base, prev));
                if reset {
                    buf.push_str(RESET);
                }
            }
            Line::Fence => {
                // Fence bytes stream eagerly, so nothing is pending here —
                // only the line's style needs closing.
                buf.push_str(RESET);
            }
        }
        self.line = Line::Start;
        self.write(&buf, out);
        // The tail is the caller's line now — finish never owes a newline.
        self.dirty = false;
    }

    /// Close the line an abandoned stream leaves open. Text streams ahead
    /// of its newline, so an error or cancellation can land mid-line —
    /// possibly mid-style; one reset-and-newline keeps whatever prints
    /// next clean. Held-back bytes are dropped: the turn is being
    /// abandoned, not completed.
    pub fn interrupt(&mut self, out: &mut dyn Write) {
        if self.dirty {
            let _ = out.write_all(RESET.as_bytes());
            let _ = out.write_all(b"\n");
            self.dirty = false;
        }
    }

    /// Write a drained batch, tracking whether it left the cursor mid-line.
    fn write(&mut self, buf: &str, out: &mut dyn Write) {
        if buf.is_empty() {
            return;
        }
        self.dirty = !buf.ends_with('\n');
        let _ = out.write_all(buf.as_bytes());
    }

    /// Render everything `pending` can already decide, advancing the line
    /// state machine across as many lines as the buffer holds.
    fn drain(&mut self) -> String {
        let mut buf = String::new();
        loop {
            match self.line {
                Line::Start => {
                    let Some(shape) = classify(&self.pending, self.in_fence) else {
                        return buf;
                    };
                    // Any non-row shape ends a pending table.
                    if !matches!(shape, Shape::Row) {
                        for row in self.flush_table() {
                            buf.push_str(&row);
                            buf.push('\n');
                        }
                    }
                    self.line = match shape {
                        Shape::Plain => Line::Inline {
                            base: "",
                            reset: false,
                            prev: None,
                        },
                        Shape::Heading(skip) => {
                            buf.push_str(HEADING);
                            self.pending.drain(..skip);
                            Line::Inline {
                                base: HEADING,
                                reset: true,
                                prev: None,
                            }
                        }
                        Shape::Bullet(indent) => {
                            buf.push_str(&self.pending[..indent]);
                            buf.push_str("• ");
                            self.pending.drain(..indent + 2);
                            Line::Inline {
                                base: "",
                                reset: false,
                                prev: None,
                            }
                        }
                        Shape::Marker => {
                            buf.push_str(DIM);
                            self.in_fence = !self.in_fence;
                            Line::Fence
                        }
                        Shape::Code => {
                            buf.push_str(CODE);
                            Line::Fence
                        }
                        Shape::Row => Line::Table,
                    };
                }
                Line::Table => {
                    let Some(pos) = self.pending.find('\n') else {
                        return buf;
                    };
                    let row = self.pending[..pos].to_string();
                    self.pending.drain(..=pos);
                    for line in self.process_line(&row) {
                        buf.push_str(&line);
                        buf.push('\n');
                    }
                    self.line = Line::Start;
                }
                Line::Inline { base, reset, prev } => {
                    let Some(pos) = self.pending.find('\n') else {
                        let consumed = scan_spans(&self.pending, prev, base, &mut buf);
                        if consumed > 0 {
                            let last = self.pending[..consumed].chars().next_back();
                            self.pending.drain(..consumed);
                            self.line = Line::Inline {
                                base,
                                reset,
                                prev: last,
                            };
                        }
                        // An opener still unresolved past the cap will not
                        // close: flush the held tail literally rather than
                        // re-scan an unbounded buffer on every delta —
                        // fail-open, like an oversized table.
                        if self.pending.len() > INLINE_MAX_BYTES {
                            buf.push_str(&self.pending);
                            let last = self.pending.chars().next_back();
                            self.pending.clear();
                            self.line = Line::Inline {
                                base,
                                reset,
                                prev: last,
                            };
                        }
                        return buf;
                    };
                    // The line end is in the buffer: the complete-line
                    // rules settle whatever is held.
                    buf.push_str(&inline_from(&self.pending[..pos], base, prev));
                    if reset {
                        buf.push_str(RESET);
                    }
                    buf.push('\n');
                    self.pending.drain(..=pos);
                    self.line = Line::Start;
                }
                Line::Fence => {
                    let Some(pos) = self.pending.find('\n') else {
                        buf.push_str(&self.pending);
                        self.pending.clear();
                        return buf;
                    };
                    buf.push_str(&self.pending[..pos]);
                    buf.push_str(RESET);
                    buf.push('\n');
                    self.pending.drain(..=pos);
                    self.line = Line::Start;
                }
            }
        }
    }

    /// Route one complete line through the table machinery: zero output
    /// lines (row buffered), several (a table flushed), or one (anything
    /// else).
    fn process_line(&mut self, line: &str) -> Vec<String> {
        if self.in_fence || self.table.is_empty() {
            if !self.in_fence && is_table_row(line) && self.try_buffer(line) {
                return Vec::new();
            }
            return vec![self.render_line(line)];
        }
        // One buffered row is only a candidate header — the next line must
        // be the separator to confirm it; from two on, any pipe row extends.
        let extends = if self.table.len() == 1 {
            is_separator_row(line)
        } else {
            is_table_row(line)
        };
        if extends && self.try_buffer(line) {
            return Vec::new();
        }
        let mut lines = self.flush_table();
        lines.extend(self.process_line(line));
        lines
    }

    /// Buffer a table row if [`TABLE_MAX_BYTES`] allows; `false` sends the
    /// caller down the ordinary-line path.
    fn try_buffer(&mut self, line: &str) -> bool {
        if self.table_bytes + line.len() > TABLE_MAX_BYTES {
            return false;
        }
        self.table_bytes += line.len();
        self.table.push(line.to_string());
        true
    }

    /// Emit whatever the table buffer holds: a confirmed table (header +
    /// separator at least) renders aligned; a lone candidate header renders
    /// as the ordinary line it turned out to be.
    fn flush_table(&mut self) -> Vec<String> {
        self.table_bytes = 0;
        let rows = std::mem::take(&mut self.table);
        match rows.len() {
            0 => Vec::new(),
            1 => vec![self.render_line(&rows[0])],
            _ => render_table(&rows),
        }
    }

    /// Render one complete line reaching the buffered paths: fenced content
    /// prints as code, everything else as inline spans. Line shapes that
    /// stream (headings, bullets, fence markers) are decided in [`classify`]
    /// and never arrive here — only `|`-shaped lines and the ambiguous
    /// prefixes `finish` cuts short (hash runs, short backtick runs, lone
    /// list markers, whitespace).
    fn render_line(&self, line: &str) -> String {
        if self.in_fence {
            return format!("{CODE}{line}{RESET}");
        }
        inline(line, "")
    }
}

/// Decide the shape of the line opening `text`, or `None` while its prefix
/// is still ambiguous — a hash run that may yet become a heading, a short
/// backtick run that may yet become a fence, whitespace that may lead
/// anything. A newline in `text` completes the line, so classification is
/// then definitive.
fn classify(text: &str, in_fence: bool) -> Option<Shape> {
    let (line, complete) = match text.find('\n') {
        Some(pos) => (&text[..pos], true),
        None => (text, false),
    };
    if in_fence {
        let body = line.trim_start();
        if body.starts_with("```") {
            return Some(Shape::Marker);
        }
        // A short backtick run (or bare whitespace) could still grow into
        // the closing marker.
        if !complete && body.bytes().all(|b| b == b'`') {
            return None;
        }
        return Some(Shape::Code);
    }
    if line.starts_with('#') {
        let hashes = line.bytes().take_while(|&b| b == b'#').count();
        if hashes == line.len() && !complete && hashes <= 6 {
            return None; // the next byte may be the heading's space
        }
        if hashes <= 6 && line[hashes..].starts_with(' ') {
            return Some(Shape::Heading(hashes + 1));
        }
        return Some(Shape::Plain);
    }
    let body = line.trim_start();
    let indent = line.len() - body.len();
    match body.as_bytes().first() {
        // Pure whitespace so far: plain once the line completes.
        None => complete.then_some(Shape::Plain),
        Some(b'|') => Some(Shape::Row),
        Some(b'`') => {
            let ticks = body.bytes().take_while(|&b| b == b'`').count();
            if ticks >= 3 {
                return Some(Shape::Marker);
            }
            if ticks == body.len() && !complete {
                return None; // one or two backticks at the buffer's edge
            }
            Some(Shape::Plain)
        }
        Some(b'-' | b'*') => {
            if body.len() == 1 && !complete {
                return None; // the next byte decides bullet vs prose
            }
            if body.as_bytes().get(1) == Some(&b' ') {
                return Some(Shape::Bullet(indent));
            }
            Some(Shape::Plain)
        }
        Some(_) => Some(Shape::Plain),
    }
}

/// One decision of the incremental scanner at a span opener.
enum Probe {
    /// The span closes in the buffer: its rendered form and raw byte length.
    Span(String, usize),
    /// The marker is provably literal — no later byte can change that.
    Literal,
    /// Undecidable until more of the line arrives: hold from here.
    Wait,
}

/// Emit every inline span of `text` (a line still without its newline) that
/// is already decidable, appending rendered output to `buf`. Returns the
/// raw bytes consumed; everything from the first undecidable opener stays
/// held. Mirrors [`inline_from`] decision for decision — anything it cannot
/// match early it defers to that complete-line fallback, so the two paths
/// cannot disagree, only differ in when they emit.
fn scan_spans(text: &str, prev: Option<char>, base: &str, buf: &mut String) -> usize {
    let mut i = 0;
    let mut lit = 0;
    while i < text.len() {
        let c = text[i..].chars().next().unwrap();
        if c != '`' && c != '*' && c != '_' {
            i += c.len_utf8();
            continue;
        }
        match probe(text, i, prev, base) {
            Probe::Span(rendered, consumed) => {
                buf.push_str(&text[lit..i]);
                buf.push_str(&rendered);
                i += consumed;
                lit = i;
            }
            Probe::Literal => i += 1, // all three markers are one byte
            Probe::Wait => {
                buf.push_str(&text[lit..i]);
                return i;
            }
        }
    }
    buf.push_str(&text[lit..]);
    text.len()
}

/// Try to decide the span opening at byte `i` of `text` from the bytes seen
/// so far. Positive decisions are final by construction: a code or bold
/// span closes at its first possible closer, an italic span lives or dies
/// on its first closing marker — nothing a later byte can revise. Anything
/// else waits for more of the line.
fn probe(text: &str, i: usize, prev0: Option<char>, base: &str) -> Probe {
    let rest = &text[i..];
    if let Some(tail) = rest.strip_prefix('`') {
        return match tail.find('`') {
            Some(end) => Probe::Span(format!("{CODE}{}{RESET}{base}", &tail[..end]), end + 2),
            None => Probe::Wait,
        };
    }
    if let Some(tail) = rest.strip_prefix("**") {
        let Some(mut end) = tail.find("**") else {
            return Probe::Wait;
        };
        // The closing `**` slides right over further `*`s (see
        // [`inline_from`]); the slide settles only once a byte beyond the
        // star run is visible.
        while tail[end + 2..].starts_with('*') {
            end += 1;
        }
        if end + 2 == tail.len() {
            return Probe::Wait;
        }
        // The flanking guard MUST run after the slide-settling check above: a `*`
        // arriving next can extend the closer and pull a trailing space into
        // `content`. Whitespace-flanked or empty bold stays literal, the same
        // rule the italic path enforces.
        let content = &tail[..end];
        if content.is_empty()
            || content.starts_with(char::is_whitespace)
            || content.ends_with(char::is_whitespace)
        {
            return Probe::Literal;
        }
        return Probe::Span(
            format!("{BOLD}{}{RESET}{base}", inline(content, BOLD)),
            end + 4,
        );
    }
    if rest == "*" {
        return Probe::Wait; // the next byte decides bold vs italic
    }
    let marker = rest.chars().next().unwrap();
    let prev = if i == 0 {
        prev0
    } else {
        text[..i].chars().next_back()
    };
    if prev == Some(marker) || (marker == '_' && prev.is_some_and(char::is_alphanumeric)) {
        return Probe::Literal;
    }
    let tail = &rest[1..];
    let Some(end) = tail.find(marker) else {
        return Probe::Wait;
    };
    let content = &tail[..end];
    if content.is_empty()
        || content.starts_with(char::is_whitespace)
        || content.ends_with(char::is_whitespace)
    {
        return Probe::Literal;
    }
    if marker == '_' {
        match tail[end + 1..].chars().next() {
            None => return Probe::Wait, // the word-boundary rule needs the next byte
            Some(next) if next.is_alphanumeric() => return Probe::Literal,
            Some(_) => {}
        }
    }
    Probe::Span(
        format!("{ITALIC}{}{RESET}{base}", inline(content, ITALIC)),
        end + 2,
    )
}

/// Render the inline spans of one complete line: `` `code` ``, `**bold**`,
/// and `*italic*`/`_italic_`, first match wins in that order. `base` is the
/// enclosing style to restore after each span's reset, so a span inside a
/// heading or an outer bold run hands control back to that style instead of
/// to plain text. Unmatched markers stay literal. `prev0` is the character
/// immediately before `text` on its line — the streaming scanner hands off
/// mid-line, and the italic flanking rules need the true neighbor; `None`
/// at a line or span start.
fn inline_from(text: &str, base: &str, prev0: Option<char>) -> String {
    let mut rendered = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        if let Some(tail) = rest.strip_prefix('`')
            && let Some(end) = tail.find('`')
        {
            rendered.push_str(CODE);
            rendered.push_str(&tail[..end]);
            rendered.push_str(RESET);
            rendered.push_str(base);
            i += end + 2;
            continue;
        }
        if let Some(tail) = rest.strip_prefix("**")
            && let Some(mut end) = tail.find("**")
        {
            // A closing `**` followed by more `*`s slides right, so
            // `**bold *nested***` and `***both***` keep the inner span's
            // closer as content instead of splitting it.
            while tail[end + 2..].starts_with('*') {
                end += 1;
            }
            let content = &tail[..end];
            // Bold needs the same empty/whitespace-flanking guard the italic
            // path has (see `italic_end`), so `2 ** 3 ** 4` stays literal instead
            // of bolding " 3 ". When it fails, fall through to the italic/literal
            // path below.
            if !content.is_empty()
                && !content.starts_with(char::is_whitespace)
                && !content.ends_with(char::is_whitespace)
            {
                rendered.push_str(BOLD);
                rendered.push_str(&inline(content, BOLD));
                rendered.push_str(RESET);
                rendered.push_str(base);
                i += end + 4;
                continue;
            }
        }
        if let Some(end) = italic_end(text, i, prev0) {
            rendered.push_str(ITALIC);
            rendered.push_str(&inline(&text[i + 1..i + 1 + end], ITALIC));
            rendered.push_str(RESET);
            rendered.push_str(base);
            i += end + 2;
            continue;
        }
        let ch = rest.chars().next().unwrap();
        rendered.push(ch);
        i += ch.len_utf8();
    }
    rendered
}

/// [`inline_from`] at a line or span start: no preceding character.
fn inline(text: &str, base: &str) -> String {
    inline_from(text, base, None)
}

/// `*text*` / `_text_` opening at byte `i` of `text`: the span content's
/// byte length, or `None` when no italic span opens here. Pragmatic rules,
/// first-match-wins like the other spans:
/// - the content is non-empty and not whitespace-flanked, so `2 * 3 * 4`
///   stays literal;
/// - a doubled marker never opens — `**` is bold's, `__` stays literal;
/// - `_` needs a word boundary outside each marker, so `snake_case_names`
///   stay literal (standard renderers don't italicize intra-word
///   underscores);
/// - only the first closing marker is considered; if it fails the rules the
///   opener stays literal (fail-open, no longer search).
///
/// `prev0` stands in for the character before `text` when `i` is 0.
fn italic_end(text: &str, i: usize, prev0: Option<char>) -> Option<usize> {
    let rest = &text[i..];
    let marker = rest.chars().next().filter(|&c| c == '*' || c == '_')?;
    let prev = if i == 0 {
        prev0
    } else {
        text[..i].chars().next_back()
    };
    if prev == Some(marker) || (marker == '_' && prev.is_some_and(char::is_alphanumeric)) {
        return None;
    }
    let tail = &rest[1..];
    let end = tail.find(marker)?;
    let content = &tail[..end];
    let next = tail[end + 1..].chars().next();
    if content.is_empty()
        || content.starts_with(char::is_whitespace)
        || content.ends_with(char::is_whitespace)
        || (marker == '_' && next.is_some_and(char::is_alphanumeric))
    {
        return None;
    }
    Some(end)
}

/// A line shaped like a table row: `|`-delimited on both edges after
/// trimming. Escaped or code-span pipes are not special-cased — the same
/// pragmatism as the rest of the renderer.
fn is_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 2 && trimmed.starts_with('|') && trimmed.ends_with('|')
}

/// `| --- | :--: |` — the row that turns the pipe row above it into a table
/// header: every cell is dashes with optional alignment colons.
fn is_separator_row(line: &str) -> bool {
    is_table_row(line)
        && cells(line).iter().all(|cell| {
            let dashes = cell.strip_prefix(':').unwrap_or(cell);
            let dashes = dashes.strip_suffix(':').unwrap_or(dashes);
            !dashes.is_empty() && dashes.bytes().all(|b| b == b'-')
        })
}

/// The trimmed cell texts of a pipe row: `| a | b |` → `["a", "b"]`.
/// Callers guarantee [`is_table_row`], so the edge pipes exist.
fn cells(line: &str) -> Vec<&str> {
    let trimmed = line.trim();
    trimmed[1..trimmed.len() - 1]
        .split('|')
        .map(str::trim)
        .collect()
}

/// Column alignment, from the separator row's colons.
#[derive(Clone, Copy)]
enum Align {
    Left,
    Center,
    Right,
}

/// `---`/`:--` → left, `--:` → right, `:-:` → center.
fn alignment(cell: &str) -> Align {
    match (cell.starts_with(':'), cell.ends_with(':')) {
        (true, true) => Align::Center,
        (false, true) => Align::Right,
        _ => Align::Left,
    }
}

/// On-screen character count of a rendered cell: SGR escape sequences
/// contribute nothing. Chars, not grapheme clusters or terminal cells — the
/// same pragmatism as the rest of the renderer.
fn visible_width(text: &str) -> usize {
    let mut width = 0;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for escape in chars.by_ref() {
                if escape == 'm' {
                    break;
                }
            }
        } else {
            width += 1;
        }
    }
    width
}

/// Pad a rendered cell to `width` visible columns; the escape-free width
/// decides the padding.
fn pad(cell: &str, width: usize, align: Align) -> String {
    let gap = width - visible_width(cell);
    let (left, right) = match align {
        Align::Left => (0, gap),
        Align::Right => (gap, 0),
        Align::Center => (gap / 2, gap - gap / 2),
    };
    format!("{}{cell}{}", " ".repeat(left), " ".repeat(right))
}

/// Render a buffered table — header, separator, body rows — as aligned
/// columns with dimmed pipes, a bold header, and a dimmed rule. Column
/// widths come from the widest rendered cell; alignment follows the
/// separator's colons. Ragged rows are tolerated, never dropped: short rows
/// pad with empty cells, long rows widen the table.
fn render_table(rows: &[String]) -> Vec<String> {
    let aligns: Vec<Align> = cells(&rows[1]).into_iter().map(alignment).collect();
    let head: Vec<String> = cells(&rows[0])
        .into_iter()
        .map(|cell| format!("{BOLD}{}{RESET}", inline(cell, BOLD)))
        .collect();
    let body: Vec<Vec<String>> = rows[2..]
        .iter()
        .map(|row| {
            cells(row)
                .into_iter()
                .map(|cell| inline(cell, ""))
                .collect()
        })
        .collect();
    let columns = std::iter::once(&head)
        .chain(&body)
        .map(Vec::len)
        .max()
        .unwrap();
    let widths: Vec<usize> = (0..columns)
        .map(|col| {
            std::iter::once(&head)
                .chain(&body)
                .filter_map(|row| row.get(col))
                .map(|cell| visible_width(cell))
                .max()
                .unwrap_or(0)
        })
        .collect();
    let render_row = |row: &[String]| {
        let mut line = String::new();
        for (col, width) in widths.iter().enumerate() {
            let cell = row.get(col).map_or("", String::as_str);
            let align = aligns.get(col).copied().unwrap_or(Align::Left);
            line.push_str(&format!("{DIM}|{RESET} {} ", pad(cell, *width, align)));
        }
        line + DIM + "|" + RESET
    };
    let rule: String = widths
        .iter()
        .map(|w| format!("|{}", "-".repeat(w + 2)))
        .collect();
    let mut out = vec![render_row(&head), format!("{DIM}{rule}|{RESET}")];
    out.extend(body.iter().map(|row| render_row(row)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a full stream through a fresh renderer: every chunk pushed in
    /// order, then the tail flushed.
    fn render(chunks: &[&str]) -> String {
        let mut r = MarkdownRenderer::default();
        let mut out = Vec::new();
        for chunk in chunks {
            r.push(chunk, &mut out);
        }
        r.finish(&mut out);
        String::from_utf8(out).unwrap()
    }

    /// Like [`render`], but capturing each push's output separately — the
    /// streaming behavior itself, not just the final bytes. The last
    /// element is what `finish` flushed.
    fn staged(chunks: &[&str]) -> Vec<String> {
        let mut r = MarkdownRenderer::default();
        let mut outs = Vec::new();
        for chunk in chunks {
            let mut out = Vec::new();
            r.push(chunk, &mut out);
            outs.push(String::from_utf8(out).unwrap());
        }
        let mut out = Vec::new();
        r.finish(&mut out);
        outs.push(String::from_utf8(out).unwrap());
        outs
    }

    #[test]
    fn plain_text_passes_through() {
        assert_eq!(render(&["hello world\n"]), "hello world\n");
    }

    #[test]
    fn multibyte_text_passes_through() {
        assert_eq!(render(&["héllo → wörld ✓\n"]), "héllo → wörld ✓\n");
    }

    #[test]
    fn bold_renders_even_when_split_across_chunks() {
        assert_eq!(render(&["**bo", "ld** x\n"]), "\x1b[1mbold\x1b[0m x\n");
    }

    #[test]
    fn one_push_can_carry_several_lines() {
        assert_eq!(render(&["a\nb\nc"]), "a\nb\nc");
    }

    #[test]
    fn inline_code_renders_cyan() {
        assert_eq!(
            render(&["see `foo()` here\n"]),
            "see \x1b[36mfoo()\x1b[0m here\n"
        );
    }

    #[test]
    fn markers_inside_code_span_stay_literal() {
        assert_eq!(
            render(&["`**not bold**`\n"]),
            "\x1b[36m**not bold**\x1b[0m\n"
        );
    }

    #[test]
    fn asterisks_inside_code_span_stay_literal() {
        assert_eq!(render(&["`a *b* c`\n"]), "\x1b[36ma *b* c\x1b[0m\n");
    }

    #[test]
    fn unmatched_backtick_stays_literal() {
        assert_eq!(render(&["a ` b\n"]), "a ` b\n");
    }

    #[test]
    fn unmatched_bold_marker_stays_literal() {
        assert_eq!(render(&["a ** b\n"]), "a ** b\n");
    }

    #[test]
    fn code_span_inside_bold_restores_bold() {
        assert_eq!(
            render(&["**a `c` b**\n"]),
            "\x1b[1ma \x1b[36mc\x1b[0m\x1b[1m b\x1b[0m\n"
        );
    }

    #[test]
    fn star_italic_renders() {
        assert_eq!(render(&["*it* x\n"]), "\x1b[3mit\x1b[0m x\n");
    }

    #[test]
    fn underscore_italic_renders() {
        assert_eq!(render(&["a _b_ c\n"]), "a \x1b[3mb\x1b[0m c\n");
    }

    #[test]
    fn italic_renders_even_when_split_across_chunks() {
        assert_eq!(render(&["*ita", "lic* x\n"]), "\x1b[3mitalic\x1b[0m x\n");
    }

    #[test]
    fn bold_containing_nested_italic() {
        assert_eq!(
            render(&["**bold with *nested***\n"]),
            "\x1b[1mbold with \x1b[3mnested\x1b[0m\x1b[1m\x1b[0m\n"
        );
    }

    #[test]
    fn triple_star_renders_bold_italic() {
        assert_eq!(
            render(&["***both***\n"]),
            "\x1b[1m\x1b[3mboth\x1b[0m\x1b[1m\x1b[0m\n"
        );
    }

    #[test]
    fn underscore_italic_nests_inside_star_italic() {
        assert_eq!(
            render(&["*_text_*\n"]),
            "\x1b[3m\x1b[3mtext\x1b[0m\x1b[3m\x1b[0m\n"
        );
    }

    #[test]
    fn star_flanked_by_spaces_stays_literal() {
        assert_eq!(render(&["2 * 3 * 4\n"]), "2 * 3 * 4\n");
    }

    #[test]
    fn bold_flanked_by_spaces_stays_literal() {
        // Whole-line path (`inline_from`): `**` flanked by spaces is arithmetic,
        // not bold — mirrors `star_flanked_by_spaces_stays_literal` for italic.
        assert_eq!(render(&["2 ** 3 ** 4\n"]), "2 ** 3 ** 4\n");
    }

    #[test]
    fn bold_whitespace_flanked_closer_releases_the_hold() {
        // Streaming path (`probe`): once the closer settles and the content is
        // whitespace-flanked, the hold releases to literal instead of bolding.
        assert_eq!(staged(&["2 ** 3 *", "* 4"]), ["2 ", "** 3 ", "** 4"]);
    }

    #[test]
    fn whitespace_before_first_closer_keeps_opener_literal() {
        assert_eq!(render(&["*foo *bar*\n"]), "*foo \x1b[3mbar\x1b[0m\n");
    }

    #[test]
    fn unclosed_star_stays_literal() {
        assert_eq!(render(&["a *b\n"]), "a *b\n");
    }

    #[test]
    fn intra_word_underscores_stay_literal() {
        assert_eq!(render(&["snake_case_name\n"]), "snake_case_name\n");
    }

    #[test]
    fn underscore_closing_into_a_word_stays_literal() {
        assert_eq!(render(&["_foo_bar\n"]), "_foo_bar\n");
    }

    #[test]
    fn doubled_underscore_stays_literal() {
        assert_eq!(render(&["__x__\n"]), "__x__\n");
    }

    #[test]
    fn heading_drops_hashes_and_styles_the_title() {
        assert_eq!(render(&["## Title\n"]), "\x1b[1;4mTitle\x1b[0m\n");
    }

    #[test]
    fn code_span_inside_heading_restores_heading_style() {
        assert_eq!(
            render(&["# The `foo` module\n"]),
            "\x1b[1;4mThe \x1b[36mfoo\x1b[0m\x1b[1;4m module\x1b[0m\n"
        );
    }

    #[test]
    fn hash_without_space_is_not_a_heading() {
        assert_eq!(render(&["#hashtag\n"]), "#hashtag\n");
    }

    #[test]
    fn seven_hashes_is_not_a_heading() {
        assert_eq!(render(&["####### deep\n"]), "####### deep\n");
    }

    #[test]
    fn dash_bullet_becomes_a_dot_and_keeps_indent() {
        assert_eq!(render(&["  - item\n"]), "  • item\n");
    }

    #[test]
    fn star_bullet_becomes_a_dot() {
        assert_eq!(render(&["* item\n"]), "• item\n");
    }

    #[test]
    fn bullet_body_still_renders_inline_spans() {
        assert_eq!(
            render(&["- **key** point\n"]),
            "• \x1b[1mkey\x1b[0m point\n"
        );
    }

    #[test]
    fn fence_dims_markers_and_leaves_content_as_code() {
        assert_eq!(
            render(&["```rust\nlet **x** = 1;\n```\nafter **b**\n"]),
            "\x1b[2m```rust\x1b[0m\n\
             \x1b[36mlet **x** = 1;\x1b[0m\n\
             \x1b[2m```\x1b[0m\n\
             after \x1b[1mb\x1b[0m\n"
        );
    }

    #[test]
    fn fence_state_survives_chunk_boundaries() {
        assert_eq!(
            render(&["```\n", "raw **text**\n", "```\n"]),
            "\x1b[2m```\x1b[0m\n\x1b[36mraw **text**\x1b[0m\n\x1b[2m```\x1b[0m\n"
        );
    }

    #[test]
    fn finish_flushes_partial_line_without_newline() {
        assert_eq!(render(&["**tail**"]), "\x1b[1mtail\x1b[0m");
    }

    #[test]
    fn finish_with_empty_buffer_writes_nothing() {
        assert_eq!(render(&["done\n"]), "done\n");
        let mut r = MarkdownRenderer::default();
        let mut out = Vec::new();
        r.finish(&mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn well_formed_table_aligns_columns() {
        assert_eq!(
            render(&["| Name | Qty |\n| :--- | ---: |\n| foo | 1 |\n| barbaz | 22 |\n"]),
            "\x1b[2m|\x1b[0m \x1b[1mName\x1b[0m   \x1b[2m|\x1b[0m \x1b[1mQty\x1b[0m \x1b[2m|\x1b[0m\n\
             \x1b[2m|--------|-----|\x1b[0m\n\
             \x1b[2m|\x1b[0m foo    \x1b[2m|\x1b[0m   1 \x1b[2m|\x1b[0m\n\
             \x1b[2m|\x1b[0m barbaz \x1b[2m|\x1b[0m  22 \x1b[2m|\x1b[0m\n"
        );
    }

    #[test]
    fn centered_column_pads_both_sides() {
        assert_eq!(
            render(&["| h |\n| :-: |\n| xxxxx |\n"]),
            "\x1b[2m|\x1b[0m   \x1b[1mh\x1b[0m   \x1b[2m|\x1b[0m\n\
             \x1b[2m|-------|\x1b[0m\n\
             \x1b[2m|\x1b[0m xxxxx \x1b[2m|\x1b[0m\n"
        );
    }

    #[test]
    fn cell_width_ignores_escape_sequences() {
        assert_eq!(
            render(&["| A | B |\n| --- | --- |\n| **bold** | y |\n"]),
            "\x1b[2m|\x1b[0m \x1b[1mA\x1b[0m    \x1b[2m|\x1b[0m \x1b[1mB\x1b[0m \x1b[2m|\x1b[0m\n\
             \x1b[2m|------|---|\x1b[0m\n\
             \x1b[2m|\x1b[0m \x1b[1mbold\x1b[0m \x1b[2m|\x1b[0m y \x1b[2m|\x1b[0m\n"
        );
    }

    #[test]
    fn header_without_separator_falls_back_raw() {
        assert_eq!(render(&["| a | b |\nplain\n"]), "| a | b |\nplain\n");
    }

    #[test]
    fn two_pipe_rows_without_separator_fall_back_raw() {
        assert_eq!(render(&["| a |\n| b |\n"]), "| a |\n| b |\n");
    }

    #[test]
    fn lone_pipe_row_at_end_of_stream_stays_raw_without_newline() {
        assert_eq!(render(&["| a |"]), "| a |");
    }

    #[test]
    fn single_row_table_renders_header_and_rule() {
        assert_eq!(
            render(&["| a | b |\n| --- | --- |\n"]),
            "\x1b[2m|\x1b[0m \x1b[1ma\x1b[0m \x1b[2m|\x1b[0m \x1b[1mb\x1b[0m \x1b[2m|\x1b[0m\n\
             \x1b[2m|---|---|\x1b[0m\n"
        );
    }

    #[test]
    fn table_flushes_when_prose_resumes() {
        assert_eq!(
            render(&["| a |\n| --- |\ntext\n"]),
            "\x1b[2m|\x1b[0m \x1b[1ma\x1b[0m \x1b[2m|\x1b[0m\n\
             \x1b[2m|---|\x1b[0m\n\
             text\n"
        );
    }

    #[test]
    fn table_at_end_of_stream_keeps_the_missing_newline_missing() {
        assert_eq!(
            render(&["| a |\n|---|\n| b |"]),
            "\x1b[2m|\x1b[0m \x1b[1ma\x1b[0m \x1b[2m|\x1b[0m\n\
             \x1b[2m|---|\x1b[0m\n\
             \x1b[2m|\x1b[0m b \x1b[2m|\x1b[0m"
        );
    }

    #[test]
    fn ragged_rows_pad_and_widen_instead_of_dropping() {
        assert_eq!(
            render(&["| a |\n|---|\n| x | y |\n"]),
            "\x1b[2m|\x1b[0m \x1b[1ma\x1b[0m \x1b[2m|\x1b[0m   \x1b[2m|\x1b[0m\n\
             \x1b[2m|---|---|\x1b[0m\n\
             \x1b[2m|\x1b[0m x \x1b[2m|\x1b[0m y \x1b[2m|\x1b[0m\n"
        );
    }

    #[test]
    fn pipe_rows_inside_a_fence_stay_code() {
        assert_eq!(
            render(&["```\n| a |\n| --- |\n```\n"]),
            "\x1b[2m```\x1b[0m\n\
             \x1b[36m| a |\x1b[0m\n\
             \x1b[36m| --- |\x1b[0m\n\
             \x1b[2m```\x1b[0m\n"
        );
    }

    #[test]
    fn row_wider_than_the_buffer_cap_passes_through_raw() {
        let giant = format!("|{}|\n", "y".repeat(TABLE_MAX_BYTES));
        assert_eq!(render(&[&giant]), giant);
    }

    #[test]
    fn overflowing_table_flushes_aligned_then_falls_back_raw() {
        let giant = format!("|{}|", "y".repeat(TABLE_MAX_BYTES));
        let input = format!("| a |\n|---|\n{giant}\n");
        assert_eq!(
            render(&[&input]),
            format!(
                "\x1b[2m|\x1b[0m \x1b[1ma\x1b[0m \x1b[2m|\x1b[0m\n\
                 \x1b[2m|---|\x1b[0m\n\
                 {giant}\n"
            )
        );
    }

    #[test]
    fn unterminated_span_past_the_cap_flushes_literally() {
        // An opener with no closer holds forever without the cap; past the
        // bound it flushes the held bytes verbatim — no styling, no loss.
        let giant = format!("*{}", "a".repeat(INLINE_MAX_BYTES));
        assert_eq!(render(&[&giant]), giant);
    }

    #[test]
    fn large_span_that_closes_within_the_cap_still_renders_styled() {
        // A well-formed span whose held tail stays under the bound resolves
        // normally: the opener holds across the split, then styles once the
        // closer arrives — the cap never fires on realistic content.
        let body = "a".repeat(INLINE_MAX_BYTES / 2);
        assert_eq!(
            render(&[&format!("**{body}"), "** x\n"]),
            format!("\x1b[1m{body}\x1b[0m x\n")
        );
    }

    #[test]
    fn text_after_a_capped_flush_keeps_rendering() {
        // After the over-cap opener flushes literally, a later `*` is just
        // literal text, but a fresh, well-formed span still styles normally.
        let giant = format!("*{}", "a".repeat(INLINE_MAX_BYTES));
        assert_eq!(
            render(&[&giant, "* and `code`\n"]),
            format!("{giant}* and \x1b[36mcode\x1b[0m\n")
        );
    }

    // ── streaming: what each push emits, before any newline ──

    #[test]
    fn prose_streams_before_its_newline() {
        assert_eq!(staged(&["hello wo", "rld"]), ["hello wo", "rld", ""]);
    }

    #[test]
    fn held_bold_opener_emits_nothing_until_it_resolves() {
        assert_eq!(
            staged(&["a **bo", "ld", "** x"]),
            ["a ", "", "\x1b[1mbold\x1b[0m x", ""]
        );
    }

    #[test]
    fn code_span_resolves_at_its_closer_not_the_newline() {
        assert_eq!(
            staged(&["see `fo", "o` here"]),
            ["see ", "\x1b[36mfoo\x1b[0m here", ""]
        );
    }

    #[test]
    fn failed_first_closer_releases_the_hold_immediately() {
        // Only the first closing marker is ever considered: " 3 " is
        // whitespace-flanked, so the opener is provably literal the moment
        // that closer arrives — not at the newline.
        assert_eq!(staged(&["2 * 3 *", " 4"]), ["2 * 3 ", "", "* 4"]);
    }

    #[test]
    fn intra_word_underscores_stream_without_holding() {
        assert_eq!(staged(&["snake_ca", "se_x"]), ["snake_ca", "se_x", ""]);
    }

    #[test]
    fn doubled_underscores_stream_as_literals() {
        assert_eq!(staged(&["__x", "__"]), ["__x", "__", ""]);
    }

    #[test]
    fn heading_styles_and_streams_before_the_newline() {
        assert_eq!(
            staged(&["## Ti", "tle\n"]),
            ["\x1b[1;4mTi", "tle\x1b[0m\n", ""]
        );
    }

    #[test]
    fn bullet_replaces_its_marker_and_streams() {
        assert_eq!(staged(&["  - ite", "m one"]), ["  • ite", "m one", ""]);
    }

    #[test]
    fn fenced_code_streams_verbatim() {
        assert_eq!(
            staged(&["```rust\nlet x", " = 1;"]),
            ["\x1b[2m```rust\x1b[0m\n\x1b[36mlet x", " = 1;", "\x1b[0m"]
        );
    }

    #[test]
    fn ambiguous_line_openers_hold_until_the_deciding_byte() {
        // "#" may become a heading, "-" a bullet, "``" a fence: each holds
        // until decided, then the line streams.
        assert_eq!(staged(&["#", "# Ok"]), ["", "\x1b[1;4mOk", "\x1b[0m"]);
        assert_eq!(staged(&["-", "x"]), ["", "-x", ""]);
        assert_eq!(staged(&["``", "` "]), ["", "\x1b[2m``` ", "\x1b[0m"]);
    }

    #[test]
    fn whitespace_only_prefix_holds_then_renders_plain() {
        assert_eq!(staged(&["  ", " \n"]), ["", "   \n", ""]);
    }

    #[test]
    fn short_backtick_run_inside_a_fence_waits_for_the_marker_decision() {
        assert_eq!(
            staged(&["```\n", "``", "`\n"]),
            ["\x1b[2m```\x1b[0m\n", "", "\x1b[2m```\x1b[0m\n", ""]
        );
    }

    #[test]
    fn partial_backticks_inside_a_fence_flush_as_code_at_the_end() {
        // `finish` cuts a fence line short while it is still an ambiguous
        // backtick run: it renders as the code it turned out to be.
        assert_eq!(
            render(&["```\n", "``"]),
            "\x1b[2m```\x1b[0m\n\x1b[36m``\x1b[0m"
        );
    }

    #[test]
    fn table_rows_still_buffer_until_the_table_closes() {
        let outs = staged(&["| a |\n| --- |\n", "done\n"]);
        assert_eq!(outs[0], "");
        assert_eq!(
            outs[1],
            "\x1b[2m|\x1b[0m \x1b[1ma\x1b[0m \x1b[2m|\x1b[0m\n\x1b[2m|---|\x1b[0m\ndone\n"
        );
    }

    #[test]
    fn consumed_spans_feed_the_flanking_rules_as_line_context() {
        // After `*a*` resolves, the closing `*` is the previous character:
        // the next `*` cannot open italics, exactly as the complete line
        // would decide it.
        assert_eq!(
            staged(&["*a*", "*b", "*\n"]),
            ["\x1b[3ma\x1b[0m", "*b", "*\n", ""]
        );
    }

    #[test]
    fn split_slide_still_renders_the_triple_star_form() {
        // The closing `**` sits at the buffer's edge: one more `*` may
        // still slide it, so the span waits.
        assert_eq!(
            staged(&["***both**", "*\n"]),
            ["", "\x1b[1m\x1b[3mboth\x1b[0m\x1b[1m\x1b[0m\n", ""]
        );
    }

    #[test]
    fn slide_resolves_in_buffer_once_the_star_run_ends() {
        // `**bold *nested***` and a byte beyond the star run, no newline:
        // the slid closer is settled, so the span renders immediately.
        assert_eq!(
            staged(&["**bold *nested*** x"]),
            ["\x1b[1mbold \x1b[3mnested\x1b[0m\x1b[1m\x1b[0m x", ""]
        );
    }

    #[test]
    fn underscore_word_boundary_needs_the_byte_after_the_closer() {
        assert_eq!(staged(&["_ab_", "c"]), ["", "_ab_c", ""]);
        assert_eq!(staged(&["_ab_", " x"]), ["", "\x1b[3mab\x1b[0m x", ""]);
    }

    #[test]
    fn finish_closes_a_heading_tail_without_newline() {
        assert_eq!(render(&["# Ti", "tle"]), "\x1b[1;4mTitle\x1b[0m");
    }

    // ── interrupt: closing the line an abandoned stream leaves open ──

    #[test]
    fn interrupt_closes_a_half_printed_line() {
        let mut r = MarkdownRenderer::default();
        let mut out = Vec::new();
        r.push("# Title without newline", &mut out);
        r.interrupt(&mut out);
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.ends_with("\x1b[0m\n"));
        // A second interrupt owes nothing.
        let mut again = Vec::new();
        r.interrupt(&mut again);
        assert!(again.is_empty());
    }

    #[test]
    fn interrupt_after_a_complete_line_writes_nothing() {
        let mut r = MarkdownRenderer::default();
        let mut out = Vec::new();
        r.push("done\n", &mut out);
        r.interrupt(&mut out);
        assert_eq!(String::from_utf8(out).unwrap(), "done\n");
    }

    #[test]
    fn interrupt_with_only_held_bytes_writes_nothing() {
        // Nothing reached the terminal: the hold is dropped silently.
        let mut r = MarkdownRenderer::default();
        let mut out = Vec::new();
        r.push("**held", &mut out);
        r.interrupt(&mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn interrupt_after_finish_writes_nothing() {
        // `finish` hands the tail to the caller — the line is theirs now.
        let mut r = MarkdownRenderer::default();
        let mut out = Vec::new();
        r.push("tail", &mut out);
        r.finish(&mut out);
        r.interrupt(&mut out);
        assert_eq!(String::from_utf8(out).unwrap(), "tail");
    }

    // ── control-byte scrubbing: model text is an injection channel ──
    //
    // The `scrub_controls` policy itself is pinned by the table test in
    // `crate::display`; these exercise it through the streaming renderer.

    #[test]
    fn plain_text_control_bytes_are_replaced() {
        assert_eq!(
            render(&["a\x1bb\x07c\rd\n"]),
            "a\u{FFFD}b\u{FFFD}c\u{FFFD}d\n"
        );
    }

    #[test]
    fn c1_controls_in_plain_text_are_replaced() {
        assert_eq!(render(&["x\u{0080}y\u{009f}z\n"]), "x\u{FFFD}y\u{FFFD}z\n");
    }

    #[test]
    fn control_bytes_inside_a_code_span_are_replaced() {
        assert_eq!(render(&["`a\x1bb`\n"]), "\x1b[36ma\u{FFFD}b\x1b[0m\n");
    }

    #[test]
    fn control_bytes_inside_a_fence_are_replaced() {
        assert_eq!(
            render(&["```\n", "l\x1bt\n", "```\n"]),
            "\x1b[2m```\x1b[0m\n\x1b[36ml\u{FFFD}t\x1b[0m\n\x1b[2m```\x1b[0m\n"
        );
    }

    #[test]
    fn injected_escape_is_scrubbed_while_emitted_styling_survives() {
        // The model's own ESC becomes U+FFFD, but the renderer's bold SGR
        // codes (its own output, added after the scrub) reach the terminal
        // intact.
        assert_eq!(render(&["**a\x1bb**\n"]), "\x1b[1ma\u{FFFD}b\x1b[0m\n");
    }

    #[test]
    fn newline_survives_among_scrubbed_controls() {
        assert_eq!(render(&["a\x07\nb\n"]), "a\u{FFFD}\nb\n");
    }

    #[test]
    fn invisible_format_chars_in_plain_text_are_replaced() {
        // A bidi override (U+202E) reverses displayed direction, a zero-width
        // space (U+200B) hides between glyphs, and U+FEFF is a zero-width
        // no-break space: all three are visual-spoofing vectors and scrub to
        // U+FFFD just like control bytes.
        assert_eq!(
            render(&["a\u{202e}b\u{200b}c\u{feff}d\n"]),
            "a\u{FFFD}b\u{FFFD}c\u{FFFD}d\n"
        );
    }

    #[test]
    fn invisible_format_chars_inside_a_code_span_are_replaced() {
        // The single-entry scrub in `push` runs before classification, so a
        // code span sees only clean bytes — the bidi override cannot survive
        // by hiding inside a backtick span.
        assert_eq!(render(&["`a\u{202e}b`\n"]), "\x1b[36ma\u{FFFD}b\x1b[0m\n");
    }

    #[test]
    fn invisible_format_scrubbed_while_emitted_styling_survives() {
        // The model's bidi override becomes U+FFFD, but the renderer's own
        // bold SGR codes (emitted after the scrub) reach the terminal intact.
        assert_eq!(render(&["**a\u{202e}b**\n"]), "\x1b[1ma\u{FFFD}b\x1b[0m\n");
    }
}

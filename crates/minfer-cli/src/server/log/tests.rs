//! The logger's contract: the level parser, the line shape, and the thread-local
//! capture seam. The request lifecycle itself is driven in
//! `src/server/batch/tests/logging.rs`, where a toy model exists.

use super::*;

#[test]
fn levels_parse_exactly_and_refuse_a_typo() {
    for (raw, want) in [
        ("off", Level::Off),
        ("error", Level::Error),
        ("warn", Level::Warn),
        ("info", Level::Info),
        ("debug", Level::Debug),
        ("  INFO ", Level::Info),
    ] {
        assert_eq!(Level::parse(raw).unwrap(), want, "{raw:?}");
    }
    let err = Level::parse("verbose").unwrap_err();
    assert!(
        err.contains("verbose"),
        "the refusal names the value: {err}"
    );
    assert!(err.contains("warn"), "the refusal names the levels: {err}");
    // The threshold order is the one the gate reads: Off < Error < … < Debug.
    assert!(Level::Off < Level::Error);
    assert!(Level::Error < Level::Warn);
    assert!(Level::Warn < Level::Info);
    assert!(Level::Info < Level::Debug);
}

#[test]
fn an_event_is_one_parseable_line_and_values_are_quoted() {
    let (_, lines) = capture(Level::Info, || {
        event(
            Level::Info,
            "request.admitted",
            &[("id", "chatcmpl-abc"), ("slot", "1"), ("queued", "0")],
        );
        event(
            Level::Info,
            "request.finished",
            &[("reason", "length limit")],
        );
        msg(Level::Warn, "[server] keeps its text");
        debug("debug line");
    });
    assert_eq!(
        lines.len(),
        3,
        "debug is below the info threshold: {lines:?}"
    );
    assert_eq!(
        lines[0],
        "level=info event=request.admitted id=chatcmpl-abc slot=1 queued=0"
    );
    // A value with a space cannot split the line.
    assert_eq!(
        lines[1],
        "level=info event=request.finished reason=\"length limit\""
    );
    // A free-form diagnostic keeps its text, quoted.
    assert_eq!(lines[2], "level=warn msg=\"[server] keeps its text\"");
}

#[test]
fn off_prints_nothing() {
    let (_, lines) = capture(Level::Off, || {
        error("boom");
        warn("warning");
        info("info");
        debug("debug");
        event(Level::Error, "request.finished", &[("id", "x")]);
    });
    assert!(lines.is_empty(), "off must print nothing: {lines:?}");
}

#[test]
fn the_threshold_gates_each_level() {
    let (_, warn_lines) = capture(Level::Warn, || {
        error("e");
        warn("w");
        info("i");
        debug("d");
    });
    assert_eq!(warn_lines.len(), 2, "{warn_lines:?}");
    assert!(warn_lines[0].starts_with("level=error"));
    assert!(warn_lines[1].starts_with("level=warn"));

    let (_, all) = capture(Level::Debug, || {
        error("e");
        warn("w");
        info("i");
        debug("d");
    });
    assert_eq!(all.len(), 4, "{all:?}");
}

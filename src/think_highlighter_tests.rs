//! `#[cfg(test)] mod think_highlighter_tests` for `src/main.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

fn run(color: bool, exit: &'static [u8], chunks: &[&[u8]]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut hi = ThinkHighlighter::new(color, exit, |b| out.extend_from_slice(b));
    for c in chunks {
        hi.feed(c);
    }
    hi.finish();
    out
}

#[test]
fn colors_think_block_gray() {
    let out = run(true, b"\x1b[32m", &[b"<think>abc</think>def"]);
    let s = String::from_utf8(out).unwrap();
    // whole think block (tags + content) stays gray; answer turns green after
    assert_eq!(s, "\x1b[90m<think>abc\x1b[90m</think>\x1b[32mdef");
}

#[test]
fn markers_split_across_chunks() {
    // <think> split 3 ways, </think> split 2 ways, trailing answer split.
    let out = run(
        true,
        b"\x1b[0m",
        &[b"<thi", b"nk>one two</th", b"ink>answer ", b"tail"],
    );
    let s = String::from_utf8(out).unwrap();
    assert_eq!(
        s,
        "\x1b[90m<think>one two\x1b[90m</think>\x1b[0manswer tail"
    );
}

#[test]
fn multiple_think_blocks() {
    let out = run(true, b"\x1b[0m", &[b"<think>a</think><think>b</think>c"]);
    let s = String::from_utf8(out).unwrap();
    assert_eq!(
        s,
        "\x1b[90m<think>a\x1b[90m</think>\x1b[0m\x1b[90m<think>b\x1b[90m</think>\x1b[0mc"
    );
}

#[test]
fn no_marker_passthrough() {
    let out = run(true, b"\x1b[0m", &[b"plain text", b" more"]);
    assert_eq!(out, b"plain text more");
}

#[test]
fn color_off_passthrough() {
    let chunks: Vec<&[u8]> = vec![b"<think>", b"secret</think>", b"visible"];
    let out = run(false, b"\x1b[0m", &chunks);
    assert_eq!(out, b"<think>secret</think>visible");
}

#[test]
fn unclosed_think_flushed_raw() {
    let out = run(true, b"\x1b[0m", &[b"<think>never closed"]);
    let s = String::from_utf8(out).unwrap();
    // gray switched on, content emitted, and finish() restores the exit
    // color so no gray leaks into the trailing stats lines
    assert_eq!(s, "\x1b[90m<think>never closed\x1b[0m");
}

#[test]
fn unclosed_think_color_off_passthrough() {
    let out = run(false, b"\x1b[0m", &[b"<think>never closed"]);
    assert_eq!(out, b"<think>never closed");
}

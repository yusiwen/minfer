//! `#[cfg(test)] mod tests` for `src/device_entry.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// The narrowed guard's acceptance: a **second thread** is refused, the
/// message names the holder, the mechanism (the unbound context stream's
/// scratches) and the issue that narrowed the scope, and the entry is
/// released again when the holder drops.
///
/// Mutation evidence (rule 3 of the gate contract): deleting the
/// `Some(holder) if holder.thread != me` arm — i.e. letting every thread in —
/// makes this test fail at `must be refused`.
///
/// Both halves live in **one** test on purpose: the guard is a process global,
/// so two tests running in parallel (libtest's default) would refuse each
/// other — the very property being asserted.
#[test]
fn the_legacy_unbound_path_is_exclusive_across_threads_and_re_entrant_on_one() {
    // ── the refusal half ────────────────────────────────────────────────
    let held = enter("the first thread's layer pass").expect("the path starts free");

    let (tx, rx) = std::sync::mpsc::channel::<Option<String>>();
    let other = std::thread::spawn(move || {
        // `.err()` drops the token on the second thread either way, so a bug
        // that let it in would not leak the entry past this test.
        let reason = enter("a second thread's layer pass")
            .err()
            .map(|e| e.to_string());
        tx.send(reason).expect("the observer is alive");
    });
    let reason = rx
        .recv()
        .expect("the observer answered")
        .expect("a second thread must be refused");
    other.join().expect("the observer thread");

    assert!(
        reason.contains("a second thread's layer pass"),
        "the refusal must name what was refused: {reason}"
    );
    assert!(
        reason.contains("the first thread's layer pass"),
        "the refusal must name the operation already inside: {reason}"
    );
    // #188 narrowed the scope: the message must name the unbound context
    // stream as the mechanism, not "CudaState is a process-wide singleton"
    // (that reading is gone with per-instance streams).
    assert!(
        reason.contains("unbound"),
        "the refusal must state the narrowed mechanism: {reason}"
    );
    assert!(
        reason.contains("layer_gpu"),
        "the refusal must name the only path that still needs it: {reason}"
    );
    assert!(
        reason.contains("#188"),
        "the refusal must point at the narrowing issue: {reason}"
    );

    // ── the re-entrancy half ────────────────────────────────────────────
    // The same thread nests freely …
    let inner = enter("a nested same-thread entry").expect("re-entrant on the owner");
    drop(inner);
    // … and dropping the inner entry must not release the outer one.
    let (tx, rx) = std::sync::mpsc::channel::<bool>();
    std::thread::spawn(move || {
        tx.send(enter("foreign while the outer entry is held").is_err())
            .expect("alive");
    })
    .join()
    .expect("thread");
    assert!(
        rx.recv().expect("answered"),
        "dropping the inner entry must not release the outer one"
    );

    // ── release ─────────────────────────────────────────────────────────
    drop(held);
    let (tx, rx) = std::sync::mpsc::channel::<bool>();
    std::thread::spawn(move || {
        tx.send(enter("after the release").is_ok())
            .expect("the observer is alive");
    })
    .join()
    .expect("the observer thread");
    assert!(
        rx.recv().expect("the observer answered"),
        "a released device path must be enterable again"
    );
}

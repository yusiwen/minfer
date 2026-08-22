# vendored `block` crate (v0.1.6, MIT)

This is a vendored copy of [rust-block v0.1.6](https://github.com/SSheldon/rust-block)
with **one fix**, wired in via `[patch.crates-io]` in the root `Cargo.toml`.

## Why

`block` is a hard dependency of `metal` (used for command-buffer completion
handlers, e.g. `MpsCommandBuffer::submit` in `src/metal.rs`). The published
0.1.6 (2019, upstream unmaintained — master has the same code) declares

```rust
enum Class { }
extern { static _NSConcreteStackBlock: Class; }
```

`Class` is uninhabited, so `_NSConcreteStackBlock` is a *static of uninhabited
type*, which rustc is phasing out (rust-lang/rust#74840): it currently warns
(`future-incompat`) and will become a **hard error**, breaking any build that
pulls in `block` (i.e. all macOS builds).

## The fix

```rust
#[repr(C)]
struct Class { _priv: [u8; 0] }   // opaque ZST — same isa-pointer semantics
```

`Class` is only used as the type of the extern static and the `isa` pointer; it
is never constructed or dereferenced. Two changes vs. upstream:

1. `enum Class {}` (uninhabited) → inhabited `repr(C)` ZST, so the extern
   static is valid and the future-incompat `uninhabited_static` lint
   (rust#74840) is silenced.
2. `#[repr(C)]` + an FFI-safe field, so the extern block passes
   `improper_ctypes` (upstream's original declaration relied on cargo's lint
   cap for registry dependencies, which does not apply to this path patch).

Explicit `extern "C"` ABIs were also added to the extern block and fn-pointer
types (the `extern`-without-ABI deprecation, a rust 2024 change).

## Updating

If upstream ever publishes a fixed version, drop this directory and the
`[patch.crates-io]` entry.

---

Original upstream README:

Rust interface for Apple's C language extension of blocks.

For more information on the specifics of the block implementation, see
Clang's documentation: http://clang.llvm.org/docs/Block-ABI-Apple.html

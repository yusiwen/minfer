# Metal Backend: objc 0.2 → objc2 Migration Plan

**Status:** ✅ Implemented (2026-08-25). The plan was executed and committed: `metal`/`block`+`vendor/block` removed, `src/metal.rs` + `src/graph/metal_backend.rs` + the `tests/*_isolation.rs` migrated to objc2-metal/block2. All `cargo test` targets pass (164 tests incl. Metal-vs-CPU + kernel isolation) and end-to-end Metal inference produces correct output. This doc is kept as the record of what changed and the pitfalls found (keep the `commandBuffer()` vs Metal-4 `newCommandBuffer` distinction, `NSUInteger=usize`, `NonNull` contents, `unsafe` wrapping, `MTLBarrierScope`).  
**Estimated effort:** 4–6 working days (senior Rust + Metal dev)  
**Scope:** `src/metal.rs` (2226 LOC), `src/graph/metal_backend.rs` (1420 LOC), `build.rs`, `Cargo.toml`, tests

---

## 1. Background

minfer's Metal backend uses the `metal` 0.28 crate, which is built on the legacy `objc` 0.2 ecosystem (frozen at 0.2.7, 2019). A full migration to `objc2-metal` 0.3.x + `block2` 0.6.x replaces:

- **`metal` 0.28** → **`objc2-metal`** (the `objc2`-based Metal bindings, actively maintained)
- **`block` 0.1.6** → **`block2`** (the `objc2`-based block runtime wrapper)
- **`objc` 0.2.x** (internal to `metal`) → **`objc2`** (modern, typed, ARC-based)

See [`docs/METAL_OBJC-ECOSYSTEM.md`](../METAL_OBJC-ECOSYSTEM.md) for the full ecosystem analysis.

---

## 2. Key API Differences (Old → New)

### 2.1 Device

| objc 0.2 (`metal` 0.28) | objc2-metal 0.3.x | Notes |
|---|---|---|
| `metal::Device::system_default() → Option<Device>` | `MTLCreateSystemDefaultDevice() → Option<Retained<ProtocolObject<dyn MTLDevice>>>` | Free function; objc2-metal has **no** `Device::system_default()` convenience — use the free fn |
| `device.name() → &str` | `device.name() → Retained<NSString>`; render via `.to_string()`/`.as_str()`, fall back to `"unknown"` | ⚠️ New returns an **owned `Retained<NSString>`** (not an `Option`) |
| `device.max_threadgroup_memory_length() → u64` | `device.maxThreadgroupMemoryLength() → NSUInteger` | Same property, snake→camel; type stays `u64`. ⚠️ **Not** `maxThreadExecutionLength()` — that is a different property (see Known Gotcha #4) |
| `device.new_compute_pipeline_state_with_function(&f) → Result<Pipeline, Error>` | `device.newComputePipelineStateWithFunction_error(&function)` (**one arg**) → `Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, Retained<NSError>>` | Error is `Retained<NSError>`, **not** a generic `ObjCError`; the trailing `None` error-slot argument is **elided** by objc2 codegen |

### 2.2 Commands & Encoders

| objc 0.2 | objc2-metal | Notes |
|---|---|---|
| `queue.new_command_buffer() → &CommandBufferRef` (autoreleased) | `queue.commandBuffer() → Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>` | ⚠️ camelCase `commandBuffer()`; returns **`Option`** (not `Result`). `newCommandBuffer` on the device is a separate **Metal-4 API** returning `MTL4CommandBuffer` — not the queue path |
| `cmd_buf.new_compute_command_encoder() → &ComputeCommandEncoderRef` | `cmd_buf.computeCommandEncoder() → Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>` | ⚠️ camelCase; **`Option`** — not an owned struct, and `unsafe` is **not** required here |
| `cmd_buf.new_blit_command_encoder() → &BlitCommandEncoderRef` | `cmd_buf.blitCommandEncoder() → Option<Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>>` | ⚠️ camelCase; **`Option`** |
| `enc.set_compute_pipeline_state(&pl)` | `encoder.setComputePipelineState(&state)` | camelCase; takes `&ProtocolObject<dyn MTLComputePipelineState>` |
| `enc.set_buffer(idx, Some(&buf), off)` | `encoder.setBuffer_offset_atIndex(Some(&buf), off, idx)` — **unsafe** | camelCase; arg order is `(buffer, offset, index)` |
| `enc.dispatch_thread_groups(MTLSize, MTLSize)` | `encoder.dispatchThreadgroups_threadsPerThreadgroup(size, threadgroup)` | camelCase |
| `enc.end_encoding()` | `encoder.endEncoding()` | camelCase |
| `cmd_buf.add_completed_handler(&blk)` | `cmd_buf.addCompletedHandler(block)` — **unsafe** | `MTLCommandBufferHandler = *mut block2::DynBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>)>` → build an `RcBlock` and pass `RcBlock::into_raw(...)` (see §3.2.7) |
| `cmd_buf.commit()` | `cmd_buf.commit()` | Same |
| `cmd_buf.status() → MTLCommandBufferStatus` | `cmd_buf.status() → MTLCommandBufferStatus` | Same enum, different module |

### 2.3 Buffers

| objc 0.2 | objc2-metal | Notes |
|---|---|---|
| `device.new_buffer(size, options) → Buffer` | `device.newBufferWithLength_options(length, options) → Option<Retained<ProtocolObject<dyn MTLBuffer>>>` | camelCase; **`Option`**; `MTLBuffer` extends `MTLResource` |
| `device.new_buffer_with_bytes_no_copy(ptr, len, options, None)` | `device.newBufferWithBytesNoCopy_length_options_deallocator(ptr, len, options, None)` — **unsafe** | camelCase; note **`BytesNoCopy`** (no underscore); `ptr: NonNull<c_void>`, `deallocator: Option<&DynBlock<...>>`; returns `Option` |
| `buf.length() → u64` | `buf.length() → NSUInteger` (=u64) | Same |
| `buf.contents() → *mut c_void` | `buf.contents() → NonNull<c_void>` | ⚠️ returns **`NonNull<c_void>`**, not `*mut c_void` — use `.as_ptr()` / `.cast::<u8>()` (e.g. `b.contents().as_ptr() as *mut u8`) |
| `metal::Buffer` (opaque `foreign_obj_type!` pointer wrapper — **no offset field**) | `Retained<ProtocolObject<dyn MTLBuffer>>` (via a `MetalBuffer` alias) | ❗️ Storage type changes, but the **byte offset is ALREADY passed separately** (`setBuffer_offset_atIndex(buf, off, idx)`); the `(buffer, u64)` weight-tuple is unchanged. Do **not** alias to `dyn MTLResource` — it lacks `.length()`/`.contents()`. |

### 2.4 Blocks / Completion Handlers

| objc 0.2 (`block`) | objc2 (`block2`) | Notes |
|---|---|---|
| `use block::ConcreteBlock;` | `use block2::{Block, RcBlock};` | Different crate |
| `ConcreteBlock::new(|r: &CmdBuf| { … }).copy()` → `RcBlock` | `RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| { … })` → `RcBlock` | `RcBlock::new()` wraps the old `copy()`. ⚠️ The closure arg is the handler's `NonNull<...>` param, **not** `&CommandBuffer` |
| `blk.call(&cmd_buf_ref)` | `RcBlock::into_raw(blk)` → pass the `*mut DynBlock` to `addCompletedHandler` | objc2-metal's handler takes a raw block pointer; `RcBlock::into_raw` hands it over (Metal retains it) |
| Manual `retain`/`release` via `msg_send!` | ARC via `Retained<T>` | No manual memory management |

### 2.5 Objective-C Runtime

| objc 0.2 | objc2 |
|---|---|
| `metal::objc::msg_send![obj, method: arg]` | `obj.method(arg)` (typed `method!` macros) |
| String selectors (`sel!("retain")`) | Compile-time checked selectors |
| Manual `retain`/`release` | Automatic via `Retained<T>` and `Weak` |

### 2.6 Libraries (Shaders)

| objc 0.2 | objc2-metal |
|---|---|
| `device.new_library_with_source(src, &opts)` | `device.newLibraryWithSource_options_error(&ns_str, None)` — `None` is the `Option<&MTLCompileOptions>` arg | → `Result<Retained<ProtocolObject<dyn MTLLibrary>>, Retained<NSError>>` |
| `device.new_library_with_data(&bytes)` | `device.newLibraryWithData_error(&dispatch_data)` | ⚠️ takes **`&DispatchData`** (dispatch2 / objc2-foundation), **not** `&[u8]` — wrap the metallib bytes; returns `Result<..., Retained<NSError>>` |
| `lib.get_function(name, None)` | `lib.newFunctionWithName(&ns_str)` | ⚠️ **no** `_error` variant; returns `Option<Retained<ProtocolObject<dyn MTLFunction>>>` (nil if not found) |
| Returns `Result<Pipeline, String>` | Returns `Result<Retained<...>, Retained<NSError>>` | Error is `Retained<NSError>` |

---

## 3. Files to Change

### 3.1 `Cargo.toml`

```diff
 [target.'cfg(target_os = "macos")'.dependencies]
 -metal = "0.28"
 -block = "0.1.6"
 +objc2-metal = "0.3"
 +block2 = "0.6"
 +objc2-foundation = "0.3"
 +objc2 = "0.6"

 -[patch.crates-io]
 -block = { path = "vendor/block" }
```

> ⚠️ **Verified version coupling (objc2-metal 0.3.2):** it requires `objc2 >=0.6.2,<0.8.0`, `objc2-foundation = "0.3.2"` (exact), and `block2 >=0.6.1,<0.8.0`. So `objc2-foundation = "0.2"` + `objc2 = "0.6"` **will not resolve** (objc2-foundation 0.2.2 pins `objc2 = 0.5.2` → two `objc2` versions). Use `objc2-foundation = "0.3"`. Confirm with `cargo tree`.
>
> **Features:** objc2-metal's default features are broad and **include `block2`, `dispatch2`, `MTLDevice`, `MTLBuffer`, `MTLCommandBuffer`, `MTLComputeCommandEncoder`, `MTLCommandEncoder`, `MTLResource`, `MTLLibrary`, `MTLComputePipeline`, `MTLTypes`, `MTLCaptureManager`, etc.**, so `objc2-metal = "0.3"` (defaults on) is enough for the API used here — you do **not** need to enumerate features unless you disable defaults. `addCompletedHandler` is gated on `block2`, which is in defaults.

- Remove `vendor/block/` directory
- Remove `[patch.crates-io]` section
- **Remove `[lints.rust] unexpected_cfgs`** (`Cargo.toml:38-43`) — exists solely to quiet objc 0.2.7's `sel_impl!` macro; dead config once `objc` is gone
- ⚠️ **Version-couple these four crates.** `objc2` / `objc2-foundation` / `objc2-metal` / `block2` must resolve to one compatible `objc2`. Pin them together and confirm with `cargo tree` (a common mismatch is `objc2 0.6` + `objc2-foundation 0.2` → two `objc2` versions). Follow whatever set `cargo add` resolves, not the literal versions above.

### 3.2 `src/metal.rs` (2226 LOC)

This is the primary rewrite target. Changes organized by section:

#### 3.2.1 Imports (top of file)

```diff
-#[cfg(target_os = "macos")]
-use metal::objc::{msg_send, sel, sel_impl};
+#[cfg(target_os = "macos")]
+use objc2_metal::{
+    MTLBarrierScope, MTLBuffer, MTLCaptureDescriptor, MTLCaptureManager,
+    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLCompileOptions,
+    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
+    MTLResource, MTLResourceOptions, MTLSize,
+};
+use objc2::{rc::Retained, runtime::ProtocolObject};
+use objc2_foundation::NSString;
+use block2::RcBlock;
```

All `metal::X` references in the file need to be replaced:
- `metal::Device` → `Retained<ProtocolObject<dyn MTLDevice>>` (via a `MetalDevice` type alias)
- `metal::CommandQueue` → `Retained<ProtocolObject<dyn MTLCommandQueue>>` (via a `MetalCommandQueue` alias)
- `metal::CommandBufferRef` → `Retained<ProtocolObject<dyn MTLCommandBuffer>>` (via a `MetalCommandBuffer` alias)
- `metal::ComputeCommandEncoderRef` → `Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>` (owned via a `MetalComputeCommandEncoder` alias; objc2-metal may model the returned encoder as a concrete wrapper type — use that instead if the compiler says so)
- `metal::Buffer` → `Retained<ProtocolObject<dyn MTLBuffer>>` (via a `MetalBuffer` alias) — ⚠️ **not** `dyn MTLResource`
- `metal::Library` → `Retained<ProtocolObject<dyn MTLLibrary>>`
- `metal::CompileOptions` → `MTLCompileOptions`
- `metal::CaptureManager` → `MTLCaptureManager`
- `metal::CaptureDescriptor` → `MTLCaptureDescriptor`
- `metal::MTLCaptureDestination` → `MTLCaptureDestination` enum variant

#### 3.2.2 `MpsStateInner` struct

Every `metal::` field type changes:

```diff
 struct MpsStateInner {
-    device: metal::Device,
+    device: MetalDevice,  // type alias

-    queue: metal::CommandQueue,
+    queue: MetalCommandQueue,  // type alias

-    pl_q4_0_f32: metal::ComputePipelineState,
+    pl_q4_0_f32: MetalComputePipelineState,  // type alias

-    weights: std::sync::Mutex<std::collections::HashMap<String, (metal::Buffer, u64)>>,
+    weights: std::sync::Mutex<std::collections::HashMap<String, (MetalBuffer, u64)>>,

-    buf_attn_partial: std::sync::Mutex<metal::Buffer>,
+    buf_attn_partial: std::sync::Mutex<MetalBuffer>,
     // ... same pattern for all metal::Buffer fields
 }
```

Type aliases at the top of the file (define ONE canonical alias per `metal::` type; the rest of the file and `metal_backend.rs` use these, so the type change is localized):

```rust
#[cfg(target_os = "macos")]
type MetalDevice = Retained<ProtocolObject<dyn MTLDevice>>;
#[cfg(target_os = "macos")]
type MetalCommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
#[cfg(target_os = "macos")]
type MetalCommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
#[cfg(target_os = "macos")]
type MetalComputeCommandEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
#[cfg(target_os = "macos")]
type MetalBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
#[cfg(target_os = "macos")]
type MetalComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
#[cfg(target_os = "macos")]
type MetalLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;
#[cfg(target_os = "macos")]
type MetalCompileOptions = Retained<MTLCompileOptions>;
```

> ⚠️ `MetalBuffer` uses `dyn MTLBuffer`, **not** `dyn MTLResource`. `MTLResource` does not expose `.length()`/`.contents()` (those live on the `MTLBuffer` sub-protocol), and minfer calls them at `metal.rs:348/431` and `metal_backend.rs:83/115/197/206/208`. Aliasing to `MTLResource` will not compile.
>
> ⚠️ **objc2-metal 0.3.2 introduced `MTL4*` types.** The queue's command-buffer method is **`commandBuffer() → Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>`** (the one minfer uses); `MTLDevice::newCommandBuffer()` is the separate **Metal-4** API returning `MTL4CommandBuffer` and is **not** used here. The completion-handler block arg is `NonNull<ProtocolObject<dyn MTLCommandBuffer>>`. Alias `MetalCommandBuffer` to the queue's returned type.

#### 3.2.3 `MpsState::try_new()` — Initialization Rewrite

**Device init:**
```diff
-let device = metal::Device::system_default()?;
+let device = MTLCreateSystemDefaultDevice()?;
```

**Library loading:**
```diff
-let lib = device.new_library_with_source(src, &opts)?;
+let ns_source = NSString::from_str(src);
+let lib = device.newLibraryWithSource_options_error(&ns_source, None)?;
```

**Embedded metallib (`metal.rs:1140`) & `MINFER_METALLIB_FILE` override (`metal.rs:1187`)** — ⚠️ `newLibraryWithData_error` takes a **`&DispatchData`** (dispatch2 / objc2-foundation), **not** `&[u8]`; returns `Result<Retained<...>, Retained<NSError>>`. Wrap the metallib bytes:
```diff
-use dispatch2::DispatchData;
-let data = DispatchData::from_bytes(&bytes);
-let lib = device.new_library_with_data(&bytes)?;
+use dispatch2::DispatchData;
+let data = DispatchData::from_bytes(&bytes);
+let lib = device.newLibraryWithData_error(&data)?;
```
`load_embedded_or_source`/`compile_metal_source` (`metal.rs:1126/1138`) are also in this phase — update their return type (`Option<MetalLibrary>`) and error handling alongside `try_new`.

**Function/Pipeline creation:**
```diff
-let f = lib.get_function(name, None)?;
-let p = device.new_compute_pipeline_state_with_function(&f)?;
+let func = lib.newFunctionWithName(&NSString::from_str(name))?;         // Option, no `_error` variant
+let p = device.newComputePipelineStateWithFunction_error(&func)?;      // ONE arg; Result<_, Retained<NSError>>
```

**Resource options:**
```diff
-metal::MTLResourceOptions::StorageModeShared
+MTLResourceOptions::StorageModeShared
```

**CaptureManager:**
```diff
-let capture = metal::CaptureManager::shared();
+let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
```

**NoCopy buffer (`register_part`)** — `unsafe`; `pointer` is `NonNull<c_void>`; `deallocator: Option<&DynBlock<...>>`:
```diff
-device.new_buffer_with_bytes_no_copy(data.as_ptr(), data.len(), options, None)
+device.newBufferWithBytesNoCopy_length_options_deallocator(
+    NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
+    data.len() as NSUInteger,
+    options,
+    None,  // deallocator: Option<&block2::DynBlock<dyn Fn(NonNull<c_void>, NSUInteger)>>
+)
```

#### 3.2.4 `MpsCommandBuffer` — Manual Retain/Release → ARC

The current code uses `unsafe { msg_send![obj, retain] }` and the `Drop` impl with `msg_send![obj, release]` because the `metal` 0.28 crate returns autoreleased objects. With objc2-metal, `commandBuffer()` returns a `Retained<...>` directly — no manual retain/release needed.

```diff
 pub struct MpsCommandBuffer<'a> {
     state: &'a MpsStateInner,
-    cmd_buf: &'a metal::CommandBufferRef,
-    enc: &'a metal::ComputeCommandEncoderRef,
+    cmd_buf: MetalCommandBuffer,        // owned (alias), ARC-managed
+    enc: MetalComputeCommandEncoder,     // owned, not 'a-borrowed
     enc_open: bool,
 }

-impl Drop for MpsCommandBuffer<'_> {
-    fn drop(&mut self) {
-        unsafe {
-            let _: () = msg_send![self.cmd_buf, release];
-            let _: () = msg_send![self.enc, release];
-        }
-    }
-}
```

> ⚠️ `cmd_buf`/`enc` use the **already-defined** aliases (`MetalCommandBuffer`/`MetalComputeCommandEncoder` = `Retained<ProtocolObject<dyn T>>`). Do **not** write `Retained<MetalCommandBuffer>` — that becomes `Retained<Retained<…>>`. `state: &'a MpsStateInner` stays (untouched): `trace_op`/`recent_trace` read `dispatch_trace` off it.

#### 3.2.5 `MpsCommandBuffer::cmd_buffer()` — Factory

```diff
 pub fn cmd_buffer(&self) -> MpsCommandBuffer<'_> {
     let cmd_buf_ref = self.inner.queue.new_command_buffer();  // old
-    let enc_ref = cmd_buf_ref.new_compute_command_encoder();
-    unsafe { msg_send![cmd_buf_ref, retain]; }
-    unsafe { msg_send![enc_ref, retain]; }
-    MpsCommandBuffer { state: &self.inner, cmd_buf: cmd_buf_ref, enc: enc_ref, enc_open: true }
+    let cmd_buf = self.inner.queue.commandBuffer()?;     // Option<Retained<...>>
+    let enc = cmd_buf.computeCommandEncoder()?;           // Option<Retained<...>>
+    Ok(MpsCommandBuffer { state: &self.inner, cmd_buf, enc, enc_open: true })
 }
```
> ⚠️ `commandBuffer()` / `computeCommandEncoder()` return **`Option`**, so `cmd_buffer()` must change its return type to `Option<MpsCommandBuffer<'_>>` (or `.expect()`/`.ok_or`). This ripples to `metal_backend.rs` `cb()` — see §3.3 item 5. `commandBuffer()` returns an `MTLCommandBuffer`.

#### 3.2.6 `MpsCommandBuffer::barrier()` — msg_send! → Direct Method

```diff
-unsafe { let _: () = msg_send![self.enc, memoryBarrierWithScope: 1u64]; }
+self.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
```
> ⚠️ `MTLBarrierScope` is a typed bit-flag type in objc2-metal — pass `MTLBarrierScope::Buffers` (or `from_bits(1)`), **not** a raw `1u64`. `metal.rs:312` currently passes the raw `1u64` through `msg_send!`.

#### 3.2.7 `MpsCommandBuffer::submit()` — Block Completion Handler

This is the most conceptually different change. The `block` 0.1.6 → `block2` migration changes how callbacks work.

In objc2-metal 0.3.2 the handler method is **`unsafe fn addCompletedHandler(&self, block: MTLCommandBufferHandler)`**, where `MTLCommandBufferHandler = *mut block2::DynBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>)>`. So you build a block2 block, take its raw pointer with `RcBlock::into_raw`, and keep it alive until the handler fires (Metal **retains** the block, so `into_raw` is safe):

```diff
-let blk = ConcreteBlock::new(move |_buf: &metal::CommandBufferRef| {
-    unsafe { dispatch_semaphore_signal(sem_val as *mut c_void); }
-});
-let blk = blk.copy();
-self.cmd_buf.add_completed_handler(&blk);
+use block2::RcBlock;
+use objc2_metal::MTLCommandBuffer;
+
+let blk = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
+    unsafe { dispatch_semaphore_signal(sem_val as *mut c_void); }
+});
+unsafe { self.cmd_buf.addCompletedHandler(block2::RcBlock::into_raw(blk)); }
```

> ⚠️ **Verified against objc2-metal 0.3.2:** the closure arg is `NonNull<ProtocolObject<dyn MTLCommandBuffer>>` (not `&MTLCommandBuffer`), the method is `unsafe` and camelCase `addCompletedHandler`, and the handler param is a **raw block pointer** (`RcBlock::into_raw`), not a `&Block`/`RcBlock` by reference. `RcBlock::new` replaces the old `.copy()`.

#### 3.2.8 `dispatch_*` FFI — keep the `extern "C"` block (optional `dispatch2` move)

The semaphore functions (`dispatch_semaphore_create`, `dispatch_semaphore_signal`, `dispatch_semaphore_wait`, `dispatch_time`, `dispatch_release`) are from libdispatch, not Metal. **They stay as `extern "C"` blocks — no change needed.** In particular **keep `dispatch_release`**: `submit()` still calls it after the wait (`metal.rs:1085` `dispatch_release(sem)`), so deleting the declaration would break the build. The `dispatch_*` FFI is unrelated to the `objc`/`block` migration — treat a switch to `dispatch2` as optional cleanup, not part of this task:

```diff
 extern "C" {
     fn dispatch_semaphore_create(value: isize) -> *mut c_void;
     fn dispatch_semaphore_signal(sem: *mut c_void) -> isize;
     fn dispatch_semaphore_wait(sem: *mut c_void, timeout: u64) -> isize;
     fn dispatch_time(when: u64, delta: i64) -> u64;
     fn dispatch_release(obj: *mut c_void);
 }
```

### 3.3 `src/graph/metal_backend.rs` (1420 LOC)

This file mostly accesses `MpsState` methods and `MpsCommandBuffer` methods — it does NOT directly import `metal::*` or use `msg_send!`. The main changes:

1. **MetalBackend::pool** — type change:
   ```diff
   -pool: Vec<metal::Buffer>,
   +pool: Vec<MetalBuffer>,  // type alias from metal.rs
   ```

2. **MetalBackend::buf()** — return type:
   ```diff
   fn buf(&self, id: usize) -> &metal::Buffer { &self.pool[id] }
   +fn buf(&self, id: usize) -> &MetalBuffer { &self.pool[id] }
   ```

3. **`positions_max()` / `read_staging()` / `copy_in()`** — these read `buf.contents()` (`metal_backend.rs:116/198/208`). ⚠️ `contents()` now returns **`NonNull<c_void>`**, so the pointer casts change to `buf.contents().as_ptr() as *const u32` (or `.cast::<u32>().as_ptr()`); `length()` (`NSUInteger`) is unaffected.

4. **`capture_split()` / `staging_alloc()`** — `staging_alloc` uses `state.new_f32_buffer()` (already abstracted) — no change. `capture_split` calls `cb().encode_captures(...)`, which internally switches to `blitCommandEncoder()` (`Option`) + `copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size` (unsafe) — see §3.2.5/Phase 3.

5. **`cb()` method** — ⚠️ the `Option` simplification shown below is **borrow-checker-unsound as written** and does **not** compile in the current call pattern. `execute_node` does `let cb = self.cb();` then immediately calls `self.buf(..)`, `self.copy_in(..)`, `self.pool[..]`, `self.state.weight_buf(..)` (all `&self`) in the same scope (`metal_backend.rs:297-310`). A `cb()` returning `&mut MpsCommandBuffer` would hold a **mutable** borrow of `self` and conflict with those `&self` calls.

   Two workable options:
   - **(a) Minimal — keep the `'static` escape.** Keep the `Box::into_raw`/`Box::from_raw` pointer pattern exactly as today; just retype the boxed value. This is the safe default (no call-site changes).
   - **(b) Owned + split borrows.** Take the buffer out of the backend before touching the pool, or restructure each call site so the encoder and the buffers are obtained in separate scopes. More invasive; only worth it if you also want to drop `unsafe impl Send/Sync`.

   Either way the field becomes `cb: Option<MpsCommandBuffer<'static>>` (needs `'static` because `state` is `&'static MpsState`), and the `cb()` body uses `get_or_insert_with`:

   ```diff
   -cb_ptr: *mut crate::metal::MpsCommandBuffer<'static>,
   +cb: Option<MpsCommandBuffer<'static>>,
   ```
   ```diff
   fn cb(&mut self) -> &mut crate::metal::MpsCommandBuffer<'static> {
   -      if self.cb_ptr.is_null() { ... Box::into_raw(...) }
   -      unsafe { &mut *self.cb_ptr }
   +      self.cb.get_or_insert_with(|| self.state.cmd_buffer())
   }
   ```

6. **`submit_pending()` / Drop** — use `self.cb.take()` and submit (this part is fine):
   ```diff
   fn submit_pending(&mut self) {
   -      if !self.cb_ptr.is_null() {
   -          let cb = unsafe { Box::from_raw(self.cb_ptr) };
   -          self.cb_ptr = std::ptr::null_mut();
   -          cb.submit().expect(...);
   +      if let Some(cb) = self.cb.take() {
   +          cb.submit().expect(...);
      }
   }
   ```

7. **`unsafe impl Send/Sync for MetalBackend` (`metal_backend.rs:72-73`)** — re-evaluate. Its justification is "the raw pointer is only dereferenced inside `&self`/`&mut self` methods". If you keep option (a) (raw pointer), keep the impls (reword the comment). If you switch to `Option<MpsCommandBuffer>` (option b) and the `Retained` fields make `MpsCommandBuffer` `Send`/`Sync`, the impls may become unnecessary — confirm before removing, and confirm the command buffer is still only touched on the scheduler thread.

### 3.4 `build.rs`

No changes needed. `xcrun metal` compilation is independent of the Rust bindings. The metallib path is embedded via `include_bytes!` — the data format is identical regardless of the Rust side.

### 3.5 `vendor/block/`

**Delete entirely.** No longer needed — `block2` is a proper, maintained crate.

### 3.6 Tests

All test modules in both `src/metal.rs` and `src/graph/metal_backend.rs` use the public API surface (`MpsState::init()`, `MpsState::get()`, `MpsCommandBuffer` methods, `MetalBackend::new()`). Since these are already abstracted, most tests require **no code changes** — only type-level compatibility.

The test `metal_pipelines_compile()` handles the new `Retained<NSError>` error type from `newComputePipelineStateWithFunction_error`.

---

## 4. Migration Phases

### Phase 0: Pre-migration Prep (0.5 day)

- [ ] Create a `feat/objc2-metal` branch
- [ ] **Verify the version matrix** — `objc2`, `objc2-foundation`, `objc2-metal`, `block2` are mutually version-coupled. Add them in one commit and confirm with `cargo tree` that they resolve to a single compatible `objc2` (e.g. `objc2 0.6` + `objc2-foundation 0.3`, or `objc2 0.5` + `objc2-foundation 0.2`; a mismatched `objc2 = 0.6` + `objc2-foundation = 0.2` will pull two `objc2` versions)
- [ ] Update `Cargo.toml` with new dependencies, keep old ones conditional via feature flags
- [ ] Verify all existing tests pass on main
- [ ] Confirm `MINFER_DISABLE_MPS=1` baseline output on a known model (for later logit comparison)

### Phase 1: Type Aliases & Skeleton (1 day)

- [ ] Define type aliases for all `metal::*` types in `metal.rs`
- [ ] Change `MpsStateInner` fields to new types (compile, don't implement yet)
- [ ] Change `MpsCommandBuffer` fields (owned types, drop `Drop` impl)
- [ ] Change `MetalBackend::pool` type
- [ ] Get the project to compile (all methods return `Err("stub")`)

### Phase 2: `MpsState::try_new()` Rewrite (1.5 days)

- [ ] Rewrite device initialization (`MTLCreateSystemDefaultDevice`)
- [ ] Rewrite library loading (`newLibraryWithSource_options_error`)
- [ ] Rewrite embedded-metallib load (`newLibraryWithData_error`, `metal.rs:1140`) and the `MINFER_METALLIB_FILE` override path (`metal.rs:1187`) — both return `Result<_, Retained<NSError>>`; wrap bytes in `DispatchData`
- [ ] Rewrite `load_embedded_or_source` / `compile_metal_source` return types (`Option<MetalLibrary>`)
- [ ] Rewrite pipeline creation loop (`.get_pl()` closure)
- [ ] Rewrite `register_part()` — `newBufferWithBytesNoCopy_length_options_deallocator` (unsafe, `NonNull` ptr)
- [ ] Rewrite `register_weight()` — zero-copy path
- [ ] Rewrite `new_f32_buffer()` — `newBufferWithLength_options`
- [ ] Test: `metal_pipelines_compile()` should compile all pipelines

### Phase 3: `MpsCommandBuffer` Methods (1 day)

- [ ] Rewrite `cmd_buffer()` factory
- [ ] Rewrite `set_bytes()` — `setBytes_length_atIndex` (objc2 drops the trailing `_`; **argument order flips**: old `set_bytes(index, length, bytes)` → `(bytes, length, index)`; `set_params` at `metal.rs:295` is affected)
- [ ] Rewrite `barrier()` — `memoryBarrierWithScope(MTLBarrierScope::Buffers)`
- [ ] Rewrite `end_compute()` — `endEncoding`
- [ ] Rewrite `encode_captures()` — `blitCommandEncoder()` (returns `Option`), `copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size` (unsafe, buffer→buffer)
- [ ] Rewrite all dispatch methods (`dispatch_1d`, `dispatch_2d`, `dispatch_3d`, `gemm_dispatch`)
- [ ] Rewrite all per-op encode methods (`quant_matmul_f32_on_gpu_buf`, `embed_tokens_gpu`, `rms_norm`, `add_f32`, `swiglu_f32`, `rope_f32`, `gqa_attn_f32`, etc.)

### Phase 4: `submit()` / Block Handler (0.5 day)

- [ ] Rewrite `submit()` completion handler — build a block2 `RcBlock` (closure arg `NonNull<ProtocolObject<dyn MTLCommandBuffer>>`) and pass `RcBlock::into_raw(blk)` to the `unsafe` `addCompletedHandler` (see §3.2.7)
- [ ] Remove `block` imports
- [ ] Test: command buffer submits and synchronizes correctly

### Phase 5: `MetalBackend` Integration (0.5 day)

- [ ] Fix `cb()` — ⚠️ choose option **(a)** (keep the `'static` box escape) unless you also refactor call sites (see §3.3 item 5). The `Option` + `&mut MpsCommandBuffer` form conflicts with the `&self` calls in `execute_node`
- [ ] Fix `submit_pending()` / `Drop` — use `take()` pattern
- [ ] Fix `capture_split()` / staging
- [ ] Re-evaluate `unsafe impl Send/Sync for MetalBackend` (see §3.3 item 7)
- [ ] Run all `metal_backend.rs` tests

### Phase 6: Cleanup & Verification (1 day)

- [ ] Remove `vendor/block/` and `[patch.crates-io]`
- [ ] **Remove the now-obsolete `[lints.rust] unexpected_cfgs` block** in `Cargo.toml:38-43` (it exists only to silence objc 0.2.7's `sel_impl!` macro; gone once `objc` is dropped)
- [ ] Remove all `unsafe` blocks that were for retain/release
- [ ] Run full test suite
- [ ] Run end-to-end inference with a real model (0.5B Q4_0, 7B Q4_K_M, **plus one Qwen3** — e.g. 4B Q4_K_M — since it exercises `Op::QkNorm`/decoupled head dim, and the DeepSeek-R1-Distill path)
- [ ] Compare Metal logits against CPU (within expected float tolerance) — remember CPU uses Q8_0 activations, Metal f32, so ~1e1 logit diffs are expected (see `AGENTS.md` Compute Graph §9)
- [ ] Update `METAL_OBJC-ECOSYSTEM.md` to mark migration complete
- [ ] Update `AGENTS.md` GPU safety section if needed

---

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| `objc2-metal` API surface differences (method signatures, error types) | Medium | The gfx-rs `metal-rs` examples (by the same author, madsmtm) and Candle's migration provide working patterns |
| Memory model: ARC vs manual retain/release | Low→Medium | ARC eliminates a whole class of bugs; the main risk is forgetting to wrap in `autoreleasepool` for FFI calls that create temporary ObjC objects |
| Autoreleased temporaries / `autoreleasepool` | Medium | `NSString::from_str()` returns an **owned** `Retained<NSString>` (no pool needed by itself); wrap `try_new()` in `autoreleasepool` to cover ObjC calls that may return autoreleased temporaries |
| Block2 completion handler semantics | Low | objc2-metal's `addCompletedHandler` is `unsafe fn addCompletedHandler(&self, block: *mut DynBlock<...>)` — build an `RcBlock` and pass `RcBlock::into_raw(...)`; keep it alive (Metal retains it) |
| Test coverage gap | Medium | Run full `cargo test` + real model inference before and after; compare logs at every phase |
| LLVM / metallib compatibility | None | build.rs uses `/usr/bin/xcrun metal` directly; the metallib format is unchanged |
| `cfg(target_os = "macos")` gates | None | All changes are macos-gated; non-macOS builds unchanged |

### Known Gotchas

1. **`autoreleasepool` for `NSString` / objc calls:** `NSString::from_str()` returns an **owned** `Retained<NSString>`, so by itself it does not leak and does not strictly need a pool. The real risk is ObjC framework calls that return *autoreleased* temporaries — wrap `try_new()` (and any method that creates strings / calls into the runtime) in `objc2::rc::autoreleasepool(|_| { ... })` to be safe; it is cheap and harmless. Just be aware the requirement is softer than "every `NSString::from_str` must be pooled".

2. **Error types:** objc2-metal errors are `Result<_, Retained<NSError>>` (from objc2-foundation), **not** a generic `ObjCError`. Update all `.expect()`/error handlers accordingly.

3. **There is no "fat" `metal::Buffer`:** the old type is an opaque `foreign_obj_type!` pointer wrapper (see `metal-0.28.0/src/buffer.rs`) with **no offset field** — minfer already passes the byte offset as a **separate** parameter (`set_buffer(idx, Some(&buf), off)` at `metal.rs:1428`) and stores weights as `(buffer, u64)` tuples. The migration just retypes `metal::Buffer` → `MetalBuffer` (`Retained<ProtocolObject<dyn MTLBuffer>>`); the `(buffer, offset)` convention is unchanged. ⚠️ Use `dyn MTLBuffer`, **not** `dyn MTLResource` (the latter lacks `.length()`/`.contents()`).

4. **`device.max_threadgroup_memory_length()`:** maps to objc2-metal `device.maxThreadgroupMemoryLength()`. ⚠️ **Not** `maxThreadExecutionLength()` — that is not this property (nor a real Metal API name; there's `maxThreadExecutionWidth` and `maxThreadsPerThreadgroup`). The stored field stays `u64` (`metal.rs:163`, used as the guard at `metal.rs:386`).

5. **MTLResourceOptions enum:** `StorageModeShared` still exists in objc2-metal (it's a bit-flag struct, so use `MTLResourceOptions::StorageModeShared`). The `CPUCacheModeDefaultCache` naming in the original draft was a red herring — just confirm each variant you use is present, and don't assume a raw-integer cast (use the typed constants or `from_bits`).

6. **Method names are camelCase (Objective-C selectors preserved).** objc2-metal 0.3.2 exposes methods under their selector-derived camelCase names — `commandBuffer`, `computeCommandEncoder`, `blitCommandEncoder`, `setComputePipelineState`, `setBuffer_offset_atIndex`, `setBytes_length_atIndex`, `dispatchThreadgroups_threadsPerThreadgroup`, `endEncoding`, `memoryBarrierWithScope`, `addCompletedHandler`, `newFunctionWithName`, `newLibraryWithSource_options_error`, `copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size`, `hasUnifiedMemory`, `waitUntilCompleted` — **not** the snake_case names shown in early drafts or the old `metal` crate. Several are `unsafe fn` (e.g. `setBytes_length_atIndex`, `setBuffer_offset_atIndex`, `addCompletedHandler`, `copyFromBuffer_...`, `newBufferWithBytesNoCopy_length_options_deallocator`).

7. **`contents()` returns `NonNull<c_void>` (not `*mut c_void`).** All `buf.contents()` call sites (`metal.rs:1465`, `metal_backend.rs:116/198/208`) need `.as_ptr()`/`.cast::<T>()` (e.g. `b.contents().as_ptr() as *mut u8`).

8. **`newLibraryWithData_error` takes `&DispatchData`, not `&[u8]`.** The embedded-metallib path (`metal.rs:1140`, included `include_bytes!`) and `MINFER_METALLIB_FILE` (`metal.rs:1187`) must wrap the bytes in a `dispatch2::DispatchData`.


---

## 6. Reference: gfx-rs metal-rs (objc2, same author)

The `metal-rs` crate at <https://github.com/madsmtm/metal-rs> is the **official objc2-metal reference implementation** by madsmtm (the objc2 author). The example at `examples/compute/main.rs` shows:

```rust
let device = MTLCreateSystemDefaultDevice().expect("No device found");      // no Device::system_default() in objc2-metal
let command_queue = device.newCommandQueue().expect("no queue");            // Option
let command_buffer = command_queue.commandBuffer().expect("no buffer");    // Option
let encoder = command_buffer.computeCommandEncoder().expect("no encoder");  // Option
encoder.setComputePipelineState(&pipeline_state);
encoder.setBuffer_offset_atIndex(Some(&buffer), 0, 0);                      // (buffer, offset, index), unsafe
encoder.dispatchThreadgroups_threadsPerThreadgroup(thread_group_count, thread_group_size);
encoder.endEncoding();
command_buffer.commit();
command_buffer.waitUntilCompleted();
```

**This is the target pattern** — but note the **camelCase** method names, the **`Option`** returns (`.expect()`/`?`), and that `MTLCreateSystemDefaultDevice()` (not `Device::system_default()`, which objc2-metal does **not** provide) is the device entry point. For minfer's semaphore-based completion keep the block per §3.2.7 and **do not** switch to `waitUntilCompleted()` (adds ~20 ms scheduler wakeup the semaphore path avoids).

## 7. Reference: HuggingFace Candle Migration

Candle migrated their Metal backend from `metal` 0.28 to `objc2-metal` in PR #3070. Key takeaways:
- They use a `ProtocolObject<dyn MTLDevice>` wrapper struct
- They use `Retained<ProtocolObject<...>>` for all Objective-C objects
- `NSString::from_str()` for string parameters
- All Metal methods that can fail return `Result<..., Retained<NSError>>`
- `Arc<Mutex<...>>` for thread-safe state (similar to minfer's `OnceLock`)

---

## 8. Post-Migration Checklist

- [ ] `vendor/block/` deleted
- [ ] `[patch.crates-io]` removed from `Cargo.toml`
- [ ] `[lints.rust] unexpected_cfgs` block removed from `Cargo.toml` (was only for objc 0.2.7)
- [ ] `cargo build --release` succeeds on macOS
- [ ] `cargo test` all passes (including `metal_pipelines_compile`, `metal_matmul_q8_matches_cpu`, `metal_attn_kv_matches_cpu`, etc.)
- [ ] Real model inference (0.5B Q4_0) — Metal produces correct output
- [ ] Real model inference (7B Q4_K_M) — Metal produces correct output
- [ ] Real model inference (Qwen3 4B Q4_K_M + DeepSeek-R1-Distill if available) — exercises `Op::QkNorm`/decoupled head dim + special-token path
- [ ] `MINFER_DISABLE_MPS=1` still works (CPU fallback)
- [ ] `MINFER_METALLIB_FILE=<path>` override still loads (this is the `newLibraryWithData` path — confirm it compiles with the new error type)
- [ ] No `future-incompat` warnings from `objc2` / `block2` / `objc2-metal`
- [ ] CI / devShell (nix flake) still builds
- [ ] `METAL_OBJC-ECOSYSTEM.md` updated: "Migration complete as of YYYY-MM-DD"

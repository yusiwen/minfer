use std::path::Path;
use std::process::Command;

fn main() {
    // ─── Build version (minfer --version) ─────────────────────────────
    // The release workflow passes MINFER_VERSION as "vX.Y.Z(shortsha)",
    // e.g. "v0.0.1(1234abc)" — the most recent "v"-prefixed release tag
    // plus the build commit hash. Local/dev builds fall back to the Cargo
    // package version (with a "v" prefix to match the tag convention).
    println!("cargo:rerun-if-env-changed=MINFER_VERSION");
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into());
    let minfer_version =
        std::env::var("MINFER_VERSION").unwrap_or_else(|_| format!("v{pkg_version}"));
    println!("cargo:rustc-env=MINFER_VERSION={minfer_version}");

    // ─── Precompiled Metal library (metallib) ─────────────────────────
    // Only run on the macOS target (CARGO_CFG_TARGET_OS, not the host — a
    // cross-compile to Linux from macOS must not spawn xcrun either). On any
    // other target no metallib is built and the env vars are left unset; the
    // Metal module itself is cfg-gated out there, so nothing reads them.
    //
    // Compile src/metal.metal → $OUT_DIR/minfer.metallib at build time so the
    // binary can load it with newLibraryWithData (llama embeds default.metallib
    // the same way; minfer previously compiled from source at every process
    // start — ~0.3-1 s, and even the first-ever run). Flags mirror llama's
    // release build (ggml/src/ggml-metal/CMakeLists.txt): `-O3`. Numerics are
    // IDENTICAL to the runtime newLibraryWithSource compile (verified 2026-08-21
    // on 7B/0.5B greedy output across -O1/-O2/-O3, -fno-fast-math,
    // -std=metal3.1/3.2 — byte-identical at every combination; the earlier
    // apparent divergence was a prompt-mixup in the A/B reference). On any
    // failure (no xcrun / no SDK / compile error) an EMPTY marker file is
    // emitted and src/metal.rs falls back to newLibraryWithSource.
    println!("cargo:rerun-if-changed=src/metal.metal");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let air = format!("{out_dir}/minfer.air");
        let metallib = format!("{out_dir}/minfer.metallib");

        // Capture stderr so a failed compile prints the REAL compiler error —
        // previously the failure was swallowed and only a bare "compile
        // failed" was shown (typical cause: an older Xcode/SDK rejecting a
        // Metal language feature; the runtime newLibraryWithSource fallback
        // then compiles the same source against the OS's Metal compiler).
        //
        // DEVELOPER_DIR/SDKROOT are stripped and /usr/bin/xcrun is used
        // explicitly so the REAL Apple toolchain is always selected:
        //   - nix devShells (this repo's flake.nix via direnv) export
        //     DEVELOPER_DIR/SDKROOT pointing at nixpkgs' open-source apple-sdk,
        //     where the nix xcbuild `xcrun` (first on PATH) has no `metal`
        //     tool ("tool 'metal' not found" → metallib fallback).
        //   - the nix xcrun also cannot resolve the macosx SDK without those
        //     vars, so stripping them only works together with the /usr/bin
        //     path. The system xcrun falls back to `xcode-select -p`.
        // Unsetting the vars only affects these two child processes — rustc
        // linking in the same shell still uses the nix SDK/clang.
        let mut metal_err = String::new();
        let metal_ok = match Command::new("/usr/bin/xcrun")
            .env_remove("DEVELOPER_DIR")
            .env_remove("SDKROOT")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-O3",
                // clang module cache goes to a writable dir (note: metal
                // only accepts the `=` form of -fmodules-cache-path)
                &format!("-fmodules-cache-path={out_dir}"),
                "-c",
                "src/metal.metal",
                "-o",
            ])
            .arg(&air)
            .output()
        {
            Ok(o) if o.status.success() => true,
            Ok(o) => {
                metal_err.push_str(&String::from_utf8_lossy(&o.stderr));
                false
            }
            Err(e) => {
                metal_err.push_str(&format!("failed to spawn /usr/bin/xcrun: {e}"));
                false
            }
        };

        let lib_ok = if metal_ok {
            match Command::new("/usr/bin/xcrun")
                .env_remove("DEVELOPER_DIR")
                .env_remove("SDKROOT")
                .args(["-sdk", "macosx", "metallib"])
                .arg(&air)
                .args(["-o", &metallib])
                .output()
            {
                Ok(o) if o.status.success() => true,
                Ok(o) => {
                    metal_err.push_str(&String::from_utf8_lossy(&o.stderr));
                    false
                }
                Err(e) => {
                    metal_err.push_str(&format!("failed to spawn /usr/bin/xcrun: {e}"));
                    false
                }
            }
        } else {
            false
        };

        if !lib_ok {
            // Fallback marker: empty metallib → runtime source compile.
            let _ = std::fs::write(&metallib, b"");
            let err = metal_err.trim();
            if err.is_empty() {
                println!("cargo:warning=Metal metallib compile failed (no compiler output) — will compile shaders from source at runtime");
            } else {
                println!("cargo:warning=Metal metallib compile failed — will compile shaders from source at runtime. Compiler output:\n{err}");
            }
        }
        let _ = std::fs::remove_file(&air);
        // Content hash in the env fingerprint: cargo does not track OUT_DIR
        // files for include_bytes!, so a changed metallib (metal.metal edit)
        // would otherwise leave a stale binary. The hash forces a recompile.
        let bytes = std::fs::read(&metallib).unwrap_or_default();
        let hash: String = bytes.iter().map(|b| format!("{b:02x}")).take(16).collect();
        println!("cargo:rustc-env=MINFER_METALLIB_PATH={metallib}");
        println!("cargo:rustc-env=MINFER_METALLIB_HASH={hash}");
    }

    // ─── CUDA kernels (opt-in: `--features cuda`) ─────────────────────
    // Everything below only runs when the `cuda` cargo feature is enabled.
    // Plain builds never touch nvcc: previously ANY build attempted the CUDA
    // compile whenever an nvcc happened to be on PATH, and a host-compiler
    // mismatch (nix devShells put GCC 15 first on PATH; CUDA 13 only accepts
    // <= 13) degraded the whole build to a "CUDA support disabled" warning.
    //
    // With the feature requested CUDA is REQUIRED: src/cuda.rs declares the
    // `launch_*` symbols that only libcuda_kernels.a provides, so every
    // failure below aborts the build with an actionable message instead of
    // leaving a binary that cannot link.
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=MINFER_CUDA_CCBIN");
    println!("cargo:rerun-if-changed=src/cuda_kernels.cu");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let nvcc = match find_nvcc() {
        Some(n) => n,
        None => panic!(
            "CUDA feature requested but nvcc was not found — install the CUDA \
             toolkit or point CUDA_HOME at its root (e.g. /usr/local/cuda)"
        ),
    };

    let cuda_home = find_cuda_home(&nvcc);
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let cu_file = "src/cuda_kernels.cu";
    let include_flag = format!("-I{cuda_home}/include");

    // nvcc inherits the first cc/g++ on PATH as its host compiler and
    // hard-fails when that is newer than the toolkit supports. Keep nvcc's
    // own default whenever it works (machines with a compatible default are
    // untouched); only pin -ccbin when the default is rejected.
    // `ccbin: None` means nvcc's own default host compiler works — pass no
    // -ccbin at all, keeping the behavior of machines whose PATH already has
    // a compatible compiler.
    let (ccbin, ccbin_label) = match std::env::var("MINFER_CUDA_CCBIN") {
        Ok(v) if !v.is_empty() => {
            let label = format!("MINFER_CUDA_CCBIN={v}");
            (Some(v), label)
        }
        _ => match detect_host_compiler(&nvcc, &out_dir, &include_flag) {
            Some(HostCompiler::Default) => (None, "nvcc default".to_string()),
            Some(HostCompiler::Pinned(c)) => {
                println!(
                    "cargo:warning=CUDA: nvcc default host compiler rejected, pinning -ccbin {c}"
                );
                (Some(c.clone()), format!("-ccbin {c}"))
            }
            None => panic!(
                "CUDA feature requested but no host compiler accepted by nvcc was \
                 found — install a GCC within the CUDA-supported range, or force \
                 one with MINFER_CUDA_CCBIN=/path/to/g++"
            ),
        },
    };

    let archs = detect_archs(&nvcc, &out_dir, &include_flag, ccbin.as_deref());
    if archs.is_empty() {
        panic!(
            "CUDA feature requested but nvcc accepted none of the candidate \
             architectures (sm_70…sm_121) — extend detect_archs() in build.rs \
             if this toolkit is newer than the list"
        );
    }
    let highest = archs.last().unwrap();
    let cudart_static = std::env::var_os("CARGO_FEATURE_CUDA_STATIC").is_some();
    println!(
        "cargo:warning=CUDA: host compiler {ccbin_label}; cudart {}; targets {}; PTX compute_{highest}",
        if cudart_static { "static" } else { "shared (default)" },
        archs
            .iter()
            .map(|a| format!("sm_{a}"))
            .collect::<Vec<_>>()
            .join(",")
    );

    let obj_file = format!("{out_dir}/cuda_kernels.o");
    let mut args: Vec<String> = vec![
        "-o".into(),
        obj_file.clone(),
        "-c".into(),
        cu_file.into(),
        include_flag,
        "-O3".into(),
        "--compiler-options".into(),
        "-fPIC".into(),
        "-Xcompiler".into(),
        "-Wno-unused-function".into(),
    ];
    if let Some(c) = &ccbin {
        args.push("-ccbin".into());
        args.push(c.clone());
    }
    for arch in &archs {
        args.push("-gencode".into());
        args.push(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    // Backward-JIT PTX. The auto-detected list has no exact sm_70/sm_72 SASS
    // on toolkits that dropped Volta, and the trailing forward-only PTX for the
    // *highest* compute can only JIT *up* — it cannot reach an older GPU (e.g.
    // a V100/sm_70). Embedding a compute_70/72 PTX (only when nvcc accepts the
    // arch, i.e. CUDA 12.x) lets those cards load by JIT; a compute_70 PTX also
    // forward-JITs to any GPU >= sm_70. compute_72 is slightly redundant but
    // kept so sm_72 has its own best-match image.
    for pa in ["70", "72"] {
        if archs.iter().any(|a| a.as_str() == pa) {
            args.push("-gencode".into());
            args.push(format!("arch=compute_{pa},code=compute_{pa}"));
        }
    }
    // Forward-compat PTX for GPUs newer than the newest SASS.
    args.push("-gencode".into());
    args.push(format!("arch=compute_{highest},code=compute_{highest}"));

    let status = Command::new(&nvcc)
        .args(&args)
        .status()
        .unwrap_or_else(|e| panic!("CUDA: failed to spawn {nvcc}: {e}"));
    if !status.success() {
        panic!("CUDA kernel compilation failed — see the nvcc errors above");
    }

    let lib_file = format!("{out_dir}/libcuda_kernels.a");
    let ar_status = Command::new("ar")
        .args(["rcs", &lib_file, &obj_file])
        .status()
        .unwrap_or_else(|e| panic!("CUDA: failed to spawn ar: {e}"));
    if !ar_status.success() {
        panic!("CUDA: ar failed to create {lib_file}");
    }

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=cuda_kernels");

    // How is the CUDA runtime linked? Exactly like llama.cpp: a single build
    // flag toggles the cudart archive vs the shared library (there GGML_STATIC
    // picks CUDA::cudart_static vs CUDA::cudart). A static cudart removes the
    // libcudart.so NEEDED dependency, so the binary can run on a host that has
    // only the NVIDIA driver + libstdc++ (no CUDA toolkit runtime). The CUDA
    // driver libcuda.so.1 is not a link dependency in either case — cudart and
    // CudaState::preload_driver() both load it lazily via dlopen at runtime.
    // The cudart library dir is NOT hardcoded to {cuda_home}/lib64: NVIDIA's
    // .run/installer and CUDA_HOME layouts put libcudart.so in <root>/lib64,
    // but Debian/Ubuntu distro packages install the runtime into the multiarch
    // dir (amd64: /usr/lib/x86_64-linux-gnu, arm64: /usr/lib/aarch64-linux-gnu)
    // with the headers in /usr/include. Probe for the actual location so both
    // layouts link — otherwise -lcudart silently fails to resolve on distro
    // installs (linker error: unable to find library -lcudart).
    let lib_dir = find_cuda_lib_dir(&cuda_home);
    if let Some(d) = &lib_dir {
        // Needed in BOTH modes so the linker can find libcudart_static.a (static)
        // and libcudart.so (shared) in the toolkit's real lib dir.
        println!("cargo:rustc-link-search={d}");
    } else {
        println!(
            "cargo:warning=CUDA: could not locate libcudart.so / libcudart_static.a \
             under {cuda_home} (checked {cuda_home}/lib64, {cuda_home}/lib and the \
             multiarch dirs) — the link will fail unless the CUDA runtime is reachable"
        );
    }

    let cudart_static = std::env::var_os("CARGO_FEATURE_CUDA_STATIC").is_some();
    if cudart_static {
        // libcudart_static.a also wants the POSIX threads + dl (for the lazy
        // driver/dlopen path); harmless no-ops on glibc >= 2.34 (the pthread
        // and dl symbols moved into libc). No rpath baked — nothing about the
        // CUDA toolkit is a runtime NEEDED dependency in this mode.
        println!("cargo:rustc-link-lib=static=cudart_static");
        // Only strictly needed on older glibc; on newer glibc they resolve to
        // the merged libc and are effectively no-ops.
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
    } else {
        println!("cargo:rustc-link-lib=dylib=cudart");
        if let Some(d) = &lib_dir {
            // Bake an rpath so the binary finds libcudart without relying on
            // LD_LIBRARY_PATH (host-driven model: run on the machine that built).
            println!("cargo:rustc-link-arg=-Wl,-rpath,{d}");
            // libcudart needs libstdc++ transitively; DT_RUNPATH is not consulted
            // for transitive deps (and nix's loader ignores /etc/ld.so.cache), so
            // emit old-style DT_RPATH, which is.
            println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
        }
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

// ─── CUDA toolchain discovery ────────────────────────────────────────────────

/// Locate nvcc, preferring the toolkit CUDA_HOME/CUDA_PATH points at so the
/// headers passed with -I and the compiler that is run always come from the
/// same toolkit. Falls back to PATH.
///
/// The PATH fallback resolves nvcc to its absolute path (via `which`) instead
/// of returning the bare "nvcc" string: find_cuda_home() derives the toolkit
/// root by stripping the "/bin/nvcc" suffix, which requires an absolute path.
/// On Debian/Ubuntu distro packages (nvcc at /usr/bin/nvcc, no /usr/local/cuda,
/// headers in /usr/include) that derivation correctly yields "/usr", so the
/// -I/usr/include flag finds the headers. With a bare "nvcc" the derivation
/// fails and find_cuda_home() would only land on "/usr" by accident; resolving
/// to the absolute path makes it explicit.
fn find_nvcc() -> Option<String> {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(home) = std::env::var(var) {
            if !home.is_empty() {
                let candidate = format!("{home}/bin/nvcc");
                if Path::new(&candidate).exists() {
                    return Some(candidate);
                }
            }
        }
    }
    if let Ok(out) = Command::new("which").arg("nvcc").output() {
        if out.status.success() {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !p.is_empty() && Path::new(&p).exists() {
                return Some(p);
            }
        }
    }
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some("nvcc".to_string());
    }
    None
}

fn find_cuda_home(nvcc: &str) -> String {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(home) = std::env::var(var) {
            if !home.is_empty() {
                return home;
            }
        }
    }
    if Path::new("/usr/local/cuda").exists() {
        return "/usr/local/cuda".to_string();
    }
    // Derive the toolkit root from the nvcc path (…/bin/nvcc) so custom
    // install locations still get their headers passed with -I.
    if let Some(root) = nvcc.strip_suffix("/bin/nvcc") {
        if Path::new(&format!("{root}/include/cuda_runtime.h")).exists() {
            return root.to_string();
        }
    }
    "/usr".to_string()
}

/// Locate the directory that actually contains the CUDA runtime library.
///
/// NVIDIA's .run installer and CUDA_HOME layouts put libcudart.so in
/// <root>/lib64 (and on some toolchains <root>/lib). Debian/Ubuntu distro
/// packages install it into the multiarch dir instead (amd64
/// /usr/lib/x86_64-linux-gnu, arm64 /usr/lib/aarch64-linux-gnu), with no
/// per-toolkit root. Probe every candidate for a real libcudart so both
/// layouts link. Checking `.so` (shared) and `_static.a` (static) covers the
/// `cuda_static` toggle too.
fn find_cuda_lib_dir(cuda_home: &str) -> Option<String> {
    let candidates = [
        format!("{cuda_home}/lib64"),
        format!("{cuda_home}/lib"),
        "/usr/lib/x86_64-linux-gnu".to_string(),
        "/usr/lib/aarch64-linux-gnu".to_string(),
        "/usr/lib64".to_string(),
        "/usr/lib".to_string(),
    ];
    candidates.into_iter().find(|d| {
        Path::new(d).join("libcudart.so").exists()
            || Path::new(d).join("libcudart_static.a").exists()
    })
}

/// Outcome of probing nvcc's host compiler for the C++ side of the kernels.
enum HostCompiler {
    /// nvcc's own default (first cc/g++ on PATH) compiles — pass no -ccbin.
    Default,
    /// The default was rejected; pin this compiler with -ccbin.
    Pinned(String),
}

/// Probe the host compiler nvcc should use. nvcc hard-fails on host GCCs
/// newer than the toolkit supports (CUDA 13 rejects GCC 15, which nix
/// devShells put first on PATH), and the glibc `noexcept` mismatch then
/// surfaces as errors inside <cmath> instead of a version diagnostic. Returns
/// `None` when neither the default nor any known candidate compiles.
fn detect_host_compiler(nvcc: &str, out_dir: &str, include_flag: &str) -> Option<HostCompiler> {
    let src = format!("{out_dir}/host_probe.cu");
    let obj = format!("{out_dir}/host_probe.o");
    let _ = std::fs::write(&src, "__global__ void host_probe() {}\n");
    let probe = |ccbin: Option<&str>| -> bool {
        let mut args: Vec<String> = vec![
            "-o".into(),
            obj.clone(),
            "-c".into(),
            src.clone(),
            include_flag.to_string(),
        ];
        if let Some(c) = ccbin {
            args.push("-ccbin".into());
            args.push(c.to_string());
        }
        Command::new(nvcc)
            .args(&args)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };

    // Newest first; nvcc performs the version check itself, so this adapts to
    // whatever CUDA is installed (older toolkits reject newer GCCs here).
    let candidates = [
        "g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++", "c++", "clang++",
    ];
    let picked = if probe(None) {
        Some(HostCompiler::Default)
    } else {
        candidates
            .iter()
            .find(|c| probe(Some(*c)))
            .map(|c| HostCompiler::Pinned(c.to_string()))
    };
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&obj);
    picked
}

/// Probe which SASS targets this nvcc accepts (cross-compilation needs no GPU
/// present). Candidates newer than the toolkit — Blackwell sm_100/103/110/120/
/// 121 require CUDA 12.8+ — simply fail the probe and are skipped, so one list
/// works on every CUDA version and every GPU gets its native SASS when
/// available. The minimum is sm_70 (Volta), not Pascal (sm_61): the kernels in
/// `cuda_kernels.cu` use WMMA tensor cores (`nvcuda::wmma`, `#if __CUDA_ARCH__ >=
/// 700` paths), which require sm_70+, so a lower target makes nvcc reject the
/// file ("name must be a namespace name" for `nvcuda`). Volta (sm_70/72) is
/// included: it is the gap the forward-only PTX (see below) cannot reach, and it
/// is still supported through CUDA 12.x (CUDA 13 removed it, so the probe skips
/// it there).
fn detect_archs(nvcc: &str, out_dir: &str, include_flag: &str, ccbin: Option<&str>) -> Vec<String> {
    let candidates = [
        "70", "72", "75", "80", "86", "89", "90", "100", "103", "110", "120", "121",
    ];
    let test_dir = format!("{out_dir}/nvcc_arch_test");
    let _ = std::fs::create_dir_all(&test_dir);
    let test_cu = format!("{test_dir}/dummy.cu");
    let _ = std::fs::write(&test_cu, "__global__ void dummy() {}\n");
    let mut supported = Vec::new();
    for arch in &candidates {
        let out = format!("{test_dir}/dummy_{arch}.o");
        let mut cmd = Command::new(nvcc);
        cmd.args(["-o", &out, "-c", &test_cu])
            .arg(format!("-arch=sm_{arch}"))
            .args(["--compiler-options", "-fPIC"])
            .arg(include_flag);
        if let Some(c) = ccbin {
            cmd.args(["-ccbin", c]);
        }
        let ok = cmd.status().map(|s| s.success()).unwrap_or(false);
        if ok {
            supported.push(arch.to_string());
            let _ = std::fs::remove_file(&out);
        }
    }
    let _ = std::fs::remove_dir_all(&test_dir);
    supported
}

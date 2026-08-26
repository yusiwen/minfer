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
    let minfer_version = std::env::var("MINFER_VERSION")
        .unwrap_or_else(|_| format!("v{pkg_version}"));
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
            .args(["-sdk", "macosx", "metal", "-O3",
                   // clang module cache goes to a writable dir (note: metal
                   // only accepts the `=` form of -fmodules-cache-path)
                   &format!("-fmodules-cache-path={out_dir}"),
                   "-c", "src/metal.metal", "-o"])
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

    let nvcc = match Command::new("nvcc").arg("--version").output() {
        Ok(_) => "nvcc",
        Err(_) => {
            println!("cargo:warning=nvcc not found, CUDA support will be disabled");
            return;
        }
    };

    let cuda_home = find_cuda_home();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let cu_file = "src/cuda_kernels.cu";
    let obj_file = format!("{}/cuda_kernels.o", out_dir);

    let archs = detect_archs(nvcc, &out_dir, &cuda_home);

    let include_flag = format!("-I{cuda_home}/include");

    let mut args: Vec<String> = Vec::new();
    args.push("-o".into()); args.push(obj_file.clone());
    args.push("-c".into()); args.push(cu_file.into());
    args.push(include_flag.clone());
    args.push("-O3".into());
    args.push("--compiler-options".into()); args.push("-fPIC".into());
    args.push("-Xcompiler".into()); args.push("-Wno-unused-function".into());

    for arch in &archs {
        args.push("-gencode".into());
        args.push(format!("arch=compute_{arch},code=sm_{arch}"));
    }
    if let Some(highest) = archs.last() {
        args.push("-gencode".into());
        args.push(format!("arch=compute_{highest},code=compute_{highest}"));
    }

    let status = Command::new(nvcc).args(&args).status()
        .expect("failed to compile CUDA kernels");

    if !status.success() {
        println!("cargo:warning=CUDA kernel compilation failed, CUDA support disabled");
        return;
    }

    let ar_status = Command::new("ar")
        .args(["rcs", &format!("{}/libcuda_kernels.a", out_dir), &obj_file])
        .status()
        .expect("failed to create static library");
    if !ar_status.success() {
        println!("cargo:warning=Failed to create static library, CUDA support disabled");
        return;
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    let lib_dir = format!("{cuda_home}/lib64");
    if Path::new(&lib_dir).exists() {
        println!("cargo:rustc-link-search={}", lib_dir);
    }

    println!("cargo:rustc-cfg=feature=\"cuda\"");
    println!("cargo:rerun-if-changed={}", cu_file);
    println!("cargo:rerun-if-changed=build.rs");
}

fn find_cuda_home() -> String {
    if let Ok(home) = std::env::var("CUDA_HOME") {
        if !home.is_empty() { return home; }
    }
    if let Ok(home) = std::env::var("CUDA_PATH") {
        if !home.is_empty() { return home; }
    }
    if Path::new("/usr/local/cuda").exists() {
        return "/usr/local/cuda".to_string();
    }
    "/usr".to_string()
}

fn detect_archs(nvcc: &str, out_dir: &str, cuda_home: &str) -> Vec<String> {
    let candidates = ["61", "75", "80", "86", "89", "90"];
    let mut supported = Vec::new();
    let test_dir = format!("{}/nvcc_arch_test", out_dir);
    let _ = std::fs::create_dir_all(&test_dir);
    let test_cu = format!("{}/dummy.cu", test_dir);
    let _ = std::fs::write(&test_cu, "__global__ void dummy() {}\n");
    let include_flag = format!("-I{cuda_home}/include");
    for arch in &candidates {
        let out = format!("{}/dummy_{arch}.o", test_dir);
        let ok = Command::new(nvcc)
            .args(["-o", &out, "-c", &test_cu])
            .arg(format!("-arch=sm_{arch}"))
            .args([&include_flag, "--compiler-options", "-fPIC"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            eprintln!("cargo:warning=CUDA: targeting sm_{arch}");
            supported.push(arch.to_string());
            let _ = std::fs::remove_file(&out);
        } else {
            eprintln!("cargo:warning=CUDA: sm_{arch} not supported by nvcc, skipping");
        }
    }
    let _ = std::fs::remove_dir_all(&test_dir);
    if supported.is_empty() {
        eprintln!("cargo:warning=CUDA: no architectures detected, defaulting to sm_75");
        supported.push("75".to_string());
    }
    supported
}

use std::path::Path;
use std::process::Command;

fn main() {
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
    let candidates = ["75", "80", "86", "89", "90"];
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

use std::process::Command;

fn main() {
    // Check if nvcc is available
    let nvcc = match Command::new("nvcc").arg("--version").output() {
        Ok(_) => "nvcc",
        Err(_) => {
            println!("cargo:warning=nvcc not found, CUDA support will be disabled");
            return;
        }
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let cu_file = "src/cuda_kernels.cu";

    // Compile CUDA kernels to object file, then create static library
    let obj_file = format!("{}/cuda_kernels.o", out_dir);
    let status = Command::new(nvcc)
        .args([
            "-o", &obj_file,
            "-c", cu_file,
            "-I/usr/local/cuda/include",
            "-arch=sm_89",
            "-O3",
            "--compiler-options", "-fPIC",
            "-Xcompiler", "-Wno-unused-function",
        ])
        .status()
        .expect("failed to compile CUDA kernels");

    if !status.success() {
        println!("cargo:warning=CUDA kernel compilation failed, CUDA support disabled");
        return;
    }

    // Create static library archive
    let ar_status = Command::new("ar")
        .args(["rcs", &format!("{}/libcuda_kernels.a", out_dir), &obj_file])
        .status()
        .expect("failed to create static library");
    if !ar_status.success() {
        println!("cargo:warning=Failed to create static library, CUDA support disabled");
        return;
    }

    // Link against CUDA runtime and C++ stdlib (nvcc generates C++ code)
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    println!("cargo:rustc-cfg=feature=\"cuda\"");

    // Re-run if CUDA source changes
    println!("cargo:rerun-if-changed={}", cu_file);
}

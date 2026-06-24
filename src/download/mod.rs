// Model download from Hugging Face Hub + Ollama registry
// Uses curl for HTTP (resumable) and serde_json for API responses

use std::path::{Path, PathBuf};

/// Default cache directory (~/.cache/minfer/models)
fn default_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let override_dir = std::env::var("MINFER_MODEL_DIR").ok();
    override_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(home).join(".cache/minfer/models"))
}

/// Resolve a model URI to a local file path, downloading if necessary.
/// Supported URIs:
///   - local path (starts with / or ./ or ~)
///   - hf:<repo>[:<file>]
///   - ollama:<model>[:tag]
pub fn resolve(uri: &str) -> Result<PathBuf, String> {
    let cache_dir = default_cache_dir();

    if uri.starts_with('/') || uri.starts_with('.') || uri.starts_with('~') {
        // Local path
        let p = if uri.starts_with('~') {
            let home = std::env::var("HOME").map_err(|e| format!("HOME not set: {}", e))?;
            PathBuf::from(home).join(&uri[2..])
        } else {
            PathBuf::from(uri)
        };
        if p.exists() {
            return Ok(p);
        }
        return Err(format!("File not found: {}", p.display()));
    }

    if let Some(repo) = uri.strip_prefix("hf:") {
        return download_hf(repo, &cache_dir);
    }
    if let Some(model) = uri.strip_prefix("ollama:") {
        return download_ollama(model, &cache_dir);
    }

    // Treat as local path fallback
    let p = PathBuf::from(uri);
    if p.exists() {
        return Ok(p);
    }
    Err(format!(
        "Unknown model URI: {}. Use hf:<repo>[:file] or ollama:<model>[:tag]",
        uri
    ))
}

// ============================================================
// Hugging Face
// ============================================================

/// Parse hf:<repo>[:<file>] and download.
/// repo: e.g. "Qwen/Qwen2-0.5B-GGUF"
/// file: optional filename, defaults to listing all GGUF files
fn download_hf(repo: &str, cache_dir: &Path) -> Result<PathBuf, String> {
    let (repo, file) = if let Some(pos) = repo.find(':') {
        (repo[..pos].to_string(), Some(repo[pos + 1..].to_string()))
    } else {
        (repo.to_string(), None)
    };

    let hf_dir = cache_dir.join("hf").join(&repo);
    std::fs::create_dir_all(&hf_dir).map_err(|e| format!("mkdir: {}", e))?;

    // Fetch model info from HF API
    let api_url = format!("https://huggingface.co/api/models/{}", repo);
    let json = http_get(&api_url)?;

    // Parse siblings from the HF API response object
    let api_resp: HfApiResponse = serde_json::from_str(&json)
        .map_err(|e| format!("JSON parse error: {}. API response: {}..", e, &json[..json.len().min(100)]))?;

    let gguf_files: Vec<&HfSibling> = api_resp
        .siblings
        .iter()
        .filter(|s| s.rfilename.ends_with(".gguf"))
        .collect();

    if gguf_files.is_empty() {
        return Err(format!(
            "No .gguf files found in '{}'. Available files:\n{}",
            repo,
            api_resp.siblings
                .iter()
                .map(|s| format!("  {}", s.rfilename))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    let target = if let Some(f) = &file {
        let found = gguf_files.iter().find(|s| s.rfilename == *f);
        match found {
            Some(s) => s,
            None => {
                let list: Vec<&str> = gguf_files.iter().map(|s| s.rfilename.as_str()).collect();
                return Err(format!(
                    "File '{}' not found in '{}'. Available GGUF files:\n{}",
                    f,
                    repo,
                    list.join("\n")
                ));
            }
        }
    } else if gguf_files.len() == 1 {
        gguf_files[0]
    } else {
        let list: Vec<&str> = gguf_files.iter().map(|s| s.rfilename.as_str()).collect();
        return Err(format!(
            "Multiple GGUF files found. Specify one:\n{}",
            list.join("\n")
        ));
    };

    let file_path = hf_dir.join(&target.rfilename);
    if file_path.exists() {
        eprintln!("Already cached: {}", file_path.display());
        return Ok(file_path);
    }

    // Download
    let download_url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo, target.rfilename
    );
    let file_size = target.size;
    http_download(&download_url, &file_path, file_size)?;

    Ok(file_path)
}

#[derive(serde::Deserialize)]
struct HfSibling {
    rfilename: String,
    #[allow(dead_code)]
    size: Option<u64>,
}

#[derive(serde::Deserialize)]
struct HfApiResponse {
    siblings: Vec<HfSibling>,
}

// ============================================================
// Ollama
// ============================================================

/// Parse ollama:<model>[:tag] and pull if needed.
fn download_ollama(model: &str, cache_dir: &Path) -> Result<PathBuf, String> {
    let (model_name, tag) = if let Some(pos) = model.find(':') {
        (model[..pos].to_string(), model[pos + 1..].to_string())
    } else {
        (model.to_string(), "latest".to_string())
    };

    let ollama_dir = cache_dir.join("ollama").join(&model_name);
    let gguf_path = ollama_dir.join("model.gguf");

    if gguf_path.exists() {
        eprintln!("Already cached: {}", gguf_path.display());
        return Ok(gguf_path);
    }

    std::fs::create_dir_all(&ollama_dir).map_err(|e| format!("mkdir: {}", e))?;

    // Use ollama CLI to pull the model
    let full_name = format!("{}:{}", model_name, tag);
    eprintln!("Pulling {} via ollama CLI...", full_name);

    let status = std::process::Command::new("ollama")
        .args(["pull", &full_name])
        .status()
        .map_err(|e| format!("Failed to run 'ollama pull': {}. Is ollama installed?", e))?;

    if !status.success() {
        return Err(format!("ollama pull {} failed", full_name));
    }

    // Locate the GGUF blob in Ollama's cache
    let ollama_home = std::env::var("HOME").unwrap();
    let ollama_blobs = PathBuf::from(&ollama_home).join(".ollama/models/blobs");

    // Get manifest to find the GGUF digest
    let manifest_path = PathBuf::from(&ollama_home)
        .join(format!(".ollama/models/manifests/registry.ollama.ai/library/{}/{}", model_name, tag));
    let manifest_json = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Cannot read Ollama manifest: {}. Try 'ollama pull {}' first. Error: {}", manifest_path.display(), full_name, e))?;

    let manifest: OllamaManifest = serde_json::from_str(&manifest_json)
        .map_err(|e| format!("Parse manifest: {}", e))?;

    // Find the largest blob (the GGUF model file)
    let largest = manifest
        .layers
        .iter()
        .chain(manifest.model.iter())
        .max_by_key(|l| l.size)
        .ok_or("No layers in manifest")?;

    let digest = largest.digest.strip_prefix("sha256:").unwrap_or(&largest.digest);
    let blob_path = ollama_blobs.join(format!("sha256-{}", digest));

    if !blob_path.exists() {
        return Err(format!(
            "Ollama blob not found at {}. Try running 'ollama pull {}' manually.",
            blob_path.display(),
            full_name
        ));
    }

    // Symlink the blob to our cache
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&blob_path, &gguf_path)
            .map_err(|e| format!("symlink: {}", e))?;
    }
    #[cfg(not(unix))]
    {
        std::fs::copy(&blob_path, &gguf_path)
            .map_err(|e| format!("copy: {}", e))?;
    }

    eprintln!("Linked: {} ← {}", gguf_path.display(), blob_path.display());
    Ok(gguf_path)
}

#[derive(serde::Deserialize)]
struct OllamaManifest {
    layers: Vec<OllamaLayer>,
    model: Vec<OllamaLayer>,
}

#[derive(serde::Deserialize)]
struct OllamaLayer {
    digest: String,
    size: u64,
}

// ============================================================
// HTTP helpers (curl wrapper)
// ============================================================

/// HTTP GET request via curl, return body as string.
fn http_get(url: &str) -> Result<String, String> {
    let output = std::process::Command::new("curl")
        .args(["-sS", "-L", url])
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("HTTP error {}: {}", output.status, stderr));
    }

    String::from_utf8(output.stdout).map_err(|e| format!("UTF-8 error: {}", e))
}

/// Download a file via curl with resume support and progress.
fn http_download(url: &str, path: &Path, _expected_size: Option<u64>) -> Result<(), String> {
    let path_str = path.to_string_lossy().to_string();

    eprintln!("Downloading: {}", url);
    eprintln!("  To: {}", path.display());

    let status = std::process::Command::new("curl")
        .args([
            "-L",
            "-C", "-",          // resume if possible
            "-o", &path_str,
            "--progress-bar",
            url,
        ])
        .status()
        .map_err(|e| format!("curl failed: {}", e))?;

    if !status.success() {
        return Err(format!("Download failed (exit code: {})", status));
    }

    Ok(())
}

// ============================================================
// List local cached models
// ============================================================

/// List all locally cached models
pub fn list_local() -> Result<(), String> {
    let cache_dir = default_cache_dir();
    if !cache_dir.exists() {
        println!("No models cached. Use 'minfer download' to fetch one.");
        return Ok(());
    }

    // Hugging Face models
    let hf_dir = cache_dir.join("hf");
    if hf_dir.exists() {
        println!("Hugging Face:");
        for entry in walk_dir(&hf_dir, 0)? {
            println!("  {}", entry);
        }
    }

    // Ollama models
    let ollama_dir = cache_dir.join("ollama");
    if ollama_dir.exists() {
        println!("\nOllama:");
        for entry in walk_dir(&ollama_dir, 0)? {
            println!("  {}", entry);
        }
    }

    Ok(())
}

fn walk_dir(dir: &Path, depth: usize) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    let entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir {}: {}", dir.display(), e))?
        .filter_map(|e| e.ok())
        .collect();

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            let sub = walk_dir(&path, depth + 1)?;
            if depth == 0 && !sub.is_empty() {
                // Show repo name as header
                result.push(format!("{}/", entry.file_name().to_string_lossy()));
                for s in sub {
                    result.push(format!("  {}", s));
                }
            } else {
                result.extend(sub);
            }
        } else if path.extension().map_or(false, |e| e == "gguf") {
            let fname = entry.file_name().to_string_lossy().to_string();
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let size_str = if size > 1_000_000_000 {
                format!("{:.1} GB", size as f64 / 1_000_000_000.0)
            } else if size > 1_000_000 {
                format!("{:.1} MB", size as f64 / 1_000_000.0)
            } else {
                format!("{} B", size)
            };
            result.push(format!("{}  ({})", fname, size_str));
        }
    }
    Ok(result)
}

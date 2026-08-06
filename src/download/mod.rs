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
///   - a cached model name (resolved against ~/.cache/minfer/models)
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

    // Bare model name → resolve against the local cache (e.g. `minfer qwen2.5-0.5b-instruct-q4_0`)
    resolve_cached_name(uri, &cache_dir)
}

/// Search the local cache for a `.gguf` whose file name matches `name` (exact match
/// preferred, then prefix match). Returns the path when exactly one match exists.
fn resolve_cached_name(name: &str, cache_dir: &Path) -> Result<PathBuf, String> {
    let mut paths = Vec::new();
    collect_gguf_paths(cache_dir, &mut paths);

    let mut exact = Vec::new();
    let mut prefix = Vec::new();
    for p in &paths {
        let fname = p.file_name().unwrap_or_default().to_string_lossy().to_string();
        if fname == name {
            exact.push(p.clone());
        } else if fname.starts_with(name) {
            prefix.push(p.clone());
        }
    }
    let candidates = if !exact.is_empty() { exact } else { prefix };

    match candidates.len() {
        1 => Ok(candidates[0].clone()),
        0 => Err(format!(
            "Model '{}' not found. Use `minfer list` to see cached models, or pass a path, hf:<repo>[:file], or ollama:<model>[:tag].",
            name
        )),
        _ => {
            // If every candidate is a part of ONE split model, resolve to part 0
            // (its split.count drives the loader, which finds the rest).
            let prefixes: std::collections::HashSet<String> = candidates.iter()
                .filter_map(|p| crate::gguf::split_file_info(&p.file_name().unwrap_or_default().to_string_lossy()))
                .map(|(p, _, _)| p)
                .collect();
            if prefixes.len() == 1 {
                if let Some(p0) = candidates.iter().find(|p| {
                    crate::gguf::split_file_info(&p.file_name().unwrap_or_default().to_string_lossy())
                        .map_or(false, |(_, idx, _)| idx == 0)
                }) {
                    return Ok(p0.clone());
                }
            }
            Err(format!(
                "Ambiguous model name '{}':\n  {}",
                name,
                candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n  ")
            ))
        }
    }
}

/// Recursively collect all `*.gguf` paths under `dir`.
fn collect_gguf_paths(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_paths(&path, out);
        } else if path.extension().map_or(false, |e| e == "gguf") {
            out.push(path);
        }
    }
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

    // Select the target model group (all parts of a split, or a single file).
    // Returns the list of part files to download + the path to return (part 0).
    let select = |requested: Option<&str>| -> Result<Vec<String>, String> {
        if let Some(f) = requested {
            let exact = gguf_files.iter().find(|s| s.rfilename == *f);
            if let Some(s) = exact {
                return Ok(vec![s.rfilename.clone()]);
            }
            // prefix match: a split like "q4_k_m" → all "-0000X-of-0000Y" parts.
            // Accept the exact split prefix ("qwen2.5-7b-instruct-q4_k_m") or a
            // shorter tail ("q4_k_m") for convenience.
            let split_prefix_matches = |p: &str| -> bool { p == f || p.ends_with(&format!("-{f}")) };
            let parts: Vec<String> = gguf_files.iter()
                .filter(|s| crate::gguf::split_file_info(&s.rfilename)
                    .map_or(false, |(p, _, _)| split_prefix_matches(&p)))
                .map(|s| s.rfilename.clone())
                .collect();
            if parts.is_empty() {
                let list: Vec<&str> = gguf_files.iter().map(|s| s.rfilename.as_str()).collect();
                return Err(format!(
                    "File '{f}' not found in '{repo}'. Available GGUF files:\n{}",
                    list.join("\n")
                ));
            }
            Ok(parts)
        } else if gguf_files.len() == 1 {
            Ok(vec![gguf_files[0].rfilename.clone()])
        } else {
            // multiple .gguf: accept if they are all parts of ONE split model
            let prefixes: std::collections::HashSet<String> = gguf_files.iter()
                .filter_map(|s| crate::gguf::split_file_info(&s.rfilename))
                .map(|(p, _, _)| p)
                .collect();
            if prefixes.len() == 1 {
                Ok(gguf_files.iter().map(|s| s.rfilename.clone()).collect())
            } else {
                let list: Vec<&str> = gguf_files.iter().map(|s| s.rfilename.as_str()).collect();
                return Err(format!(
                    "Multiple GGUF files found. Specify one:\n{}",
                    list.join("\n")
                ));
            }
        }
    };

    let parts = select(file.as_deref())?;
    let return_name = {
        // part 0 is the model entry (its split.count drives the loader)
        let first = &parts[0];
        match crate::gguf::split_file_info(first) {
            Some((prefix, _, count)) => format!("{}-00001-of-{:05}.gguf", prefix, count),
            None => first.clone(),
        }
    };

    for name in &parts {
        let file_path = hf_dir.join(name);
        let size = gguf_files.iter().find(|s| &s.rfilename == name).and_then(|s| s.size);
        // Skip only when the file exists AND (if we know the size) matches it —
        // a partial/interrupted download must be resumed, not skipped.
        let complete = file_path.exists() && size.map_or(false, |s| {
            file_path.metadata().map(|m| m.len() == s).unwrap_or(false)
        });
        if complete {
            eprintln!("Already cached: {}", file_path.display());
            continue;
        }
        let download_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo, name
        );
        http_download(&download_url, &file_path, size)?;
    }

    Ok(hf_dir.join(&return_name))
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

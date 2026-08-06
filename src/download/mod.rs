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

/// Match a requested quant type (or exact filename) against the repo's `.gguf`
/// file list, grouping split parts into one model. Returns the ordered list of
/// part filenames to download (all parts of a split, or the single file).
///
/// Matching (quant is case-insensitive, single-file or split):
///   1. exact filename (as-is) — if it is a split part, expands to the whole group
///   2. base-name match: base == q or base ends with "-{q}" (q = quant lowercased),
///      where base = split prefix, or the single filename minus `.gguf`
///   3. exactly one group must match, else an error lists the available models.
/// No requested quant: the repo must contain exactly one model group.
fn match_model(filenames: &[String], requested: Option<&str>) -> Result<Vec<String>, String> {
    struct Group {
        base: String,
        files: Vec<String>,
    }
    let mut groups: Vec<Group> = Vec::new();
    for f in filenames {
        let base = match crate::gguf::split_file_info(f) {
            Some((prefix, _, _)) => prefix,
            None => f.strip_suffix(".gguf").unwrap_or(f).to_string(),
        };
        match groups.iter_mut().find(|g| g.base == base) {
            Some(g) => g.files.push(f.clone()),
            None => groups.push(Group { base, files: vec![f.clone()] }),
        }
    }
    let list = |g: &[&Group]| -> String {
        g.iter().map(|g| format!("  {}", g.base)).collect::<Vec<_>>().join("\n")
    };

    if let Some(req) = requested {
        if let Some(f) = filenames.iter().find(|f| f.as_str() == req) {
            let base = match crate::gguf::split_file_info(f) {
                Some((prefix, _, _)) => prefix,
                None => f.strip_suffix(".gguf").unwrap_or(f).to_string(),
            };
            let group = groups.iter().find(|g| g.base == base).expect("group exists");
            return Ok(group.files.clone());
        }
        let q = req.to_lowercase();
        let matched: Vec<&Group> = groups
            .iter()
            .filter(|g| {
                let b = g.base.to_lowercase();
                b == q || b.ends_with(&format!("-{q}"))
            })
            .collect();
        return match matched.len() {
            0 => Err(format!(
                "Quant '{}' not found in repo. Available models:\n{}",
                req,
                list(&groups.iter().collect::<Vec<_>>())
            )),
            1 => Ok(matched[0].files.clone()),
            _ => Err(format!(
                "Quant '{}' is ambiguous — matches multiple models:\n{}",
                req,
                list(&matched)
            )),
        };
    }

    match groups.len() {
        0 => Err("No .gguf files found in repo.".to_string()),
        1 => Ok(groups[0].files.clone()),
        _ => Err(format!(
            "Multiple models in repo. Specify a quant type:\n{}",
            list(&groups.iter().collect::<Vec<_>>())
        )),
    }
}

/// Parse hf:<repo>[:<file>] and download.
/// repo: e.g. "Qwen/Qwen2-0.5B-GGUF"
/// file: optional quant type or filename, defaults to listing all GGUF files
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
    // Returns the list of part files to download.
    let filenames: Vec<String> = gguf_files.iter().map(|s| s.rfilename.clone()).collect();
    let parts = match_model(&filenames, file.as_deref())?;
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
        let download_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo, name
        );
        // Expected size: prefer the HF API `size`; fall back to a HEAD request
        // (many repos omit `size`), so a complete cached file is skipped, not
        // re-fetched.
        let size = gguf_files
            .iter()
            .find(|s| &s.rfilename == name)
            .and_then(|s| s.size)
            .or_else(|| head_content_length(&download_url));
        // Skip only when the file exists AND its size matches the remote one —
        // a partial/interrupted download must be resumed, not skipped.
        let complete = file_path.exists() && size.map_or(false, |s| {
            file_path.metadata().map(|m| m.len() == s).unwrap_or(false)
        });
        if complete {
            eprintln!("Already cached: {}", file_path.display());
            continue;
        }
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

/// Query the remote Content-Length (HEAD), for repos whose API omits `size`.
fn head_content_length(url: &str) -> Option<u64> {
    let output = std::process::Command::new("curl")
        .args(["-sIL", url])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines().rev().find_map(|l| {
        let t = l.trim();
        let head = t.get(..15)?;
        if !head.eq_ignore_ascii_case("content-length:") {
            return None;
        }
        t.get(15..).and_then(|v| v.trim().parse().ok())
    })
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

#[cfg(test)]
mod tests {
    use super::match_model;

    fn single(name: &str) -> Vec<String> {
        vec![name.to_string()]
    }

    fn split(prefix: &str, count: usize) -> Vec<String> {
        (1..=count)
            .map(|i| format!("{}-{:05}-of-{:05}.gguf", prefix, i, count))
            .collect()
    }

    #[test]
    fn match_single_file_quant() {
        let files = single("qwen2.5-0.5b-instruct-q4_0.gguf");
        assert_eq!(match_model(&files, Some("q4_0")).unwrap(), files);
    }

    #[test]
    fn match_single_file_quant_case_insensitive() {
        let files = single("qwen2.5-0.5b-instruct-q4_0.gguf");
        assert_eq!(match_model(&files, Some("Q4_0")).unwrap(), files);
        assert_eq!(match_model(&files, Some("Q4_K_M")).unwrap_err().contains("not found"), true);
    }

    #[test]
    fn match_split_quant() {
        let files = split("qwen2.5-7b-instruct-q4_k_m", 2);
        let got = match_model(&files, Some("q4_k_m")).unwrap();
        assert_eq!(got, files);
        assert_eq!(match_model(&files, Some("Q4_K_M")).unwrap(), files);
        assert_eq!(match_model(&files, Some("qwen2.5-7b-instruct-q4_k_m")).unwrap(), files);
    }

    #[test]
    fn match_exact_filename_expands_split() {
        let files = split("foo", 3);
        let part0 = &files[0];
        let got = match_model(&files, Some(part0)).unwrap();
        assert_eq!(got, files); // whole group
    }

    #[test]
    fn match_mixed_repo_ambiguous() {
        let mut files = Vec::new();
        files.extend(single("m1-q4_k_m.gguf"));
        files.extend(split("m2-q4_k_m", 2));
        files.extend(split("m2-q5_k_m", 2));
        // two different base names share the same quant tail → ambiguous
        let err = match_model(&files, Some("q4_k_m")).unwrap_err();
        assert!(err.contains("ambiguous"), "err: {err}");
        assert!(err.contains("m1-q4_k_m") && err.contains("m2-q4_k_m"), "err: {err}");
        // unique
        assert_eq!(match_model(&files, Some("q5_k_m")).unwrap(), split("m2-q5_k_m", 2));
        // not found
        assert!(match_model(&files, Some("q4_0")).unwrap_err().contains("not found"));
    }

    #[test]
    fn match_no_requested() {
        // one model group (split) → all parts
        let files = split("m-q4_k_m", 2);
        assert_eq!(match_model(&files, None).unwrap(), files);
        // multiple models → error
        let mut files2 = Vec::new();
        files2.extend(single("m1-q4_0.gguf"));
        files2.extend(single("m2-q4_k_m.gguf"));
        assert!(match_model(&files2, None).unwrap_err().contains("Multiple models"));
    }
}

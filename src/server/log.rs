//! Leveled, structured logging for the server (#726).
//!
//! One line per event on stderr, shaped
//! `level=<level> event=<dotted.name> key=value …`; a free-form diagnostic keeps
//! its text as `level=<level> msg="…"`, so today's messages stay greppable while
//! gaining a level. `MINFER_LOG` (`off|error|warn|info|debug`, default `warn`)
//! fixes the threshold **once at startup** — see [`from_env`] and [`init`].
//!
//! What this is not: the startup banner and a command-line error are program
//! output, not log events, and are printed directly (`docs/USAGE.md` says so). A
//! request's event lines are the log: at `info` exactly one
//! `request.admitted` and one `request.finished` line per request, both carrying
//! the same request id; at `warn` (the default) neither is printed.

use std::sync::OnceLock;

/// The log threshold. Ordered: a line is printed when its own level is at or
/// below the threshold, so `Off` prints nothing and `Debug` prints everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Off,
    Error,
    Warn,
    Info,
    Debug,
}

impl Level {
    /// Parse `MINFER_LOG`. An unknown spelling is a named refusal, never a
    /// silent default — a typo must not decide how loud the server is.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "error" => Ok(Self::Error),
            "warn" => Ok(Self::Warn),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            other => Err(format!(
                "MINFER_LOG={other:?} is not a level (expected one of: off, error, warn, info, \
                 debug)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
        }
    }
}

/// The threshold, resolved once. Until [`init`] runs (tests, or a path that
/// never reaches `serve`) the default is `warn`, which is also the documented
/// default for a run with no `MINFER_LOG`.
static LEVEL: OnceLock<Level> = OnceLock::new();

/// Resolve `MINFER_LOG` for a starting server. `None`/unset means the default
/// (`warn`); an unknown value is a refusal the caller reports and exits on.
pub fn from_env() -> Result<Level, String> {
    match std::env::var("MINFER_LOG") {
        Ok(raw) => Level::parse(&raw),
        Err(std::env::VarError::NotPresent) => Ok(Level::Warn),
        Err(e) => Err(format!("MINFER_LOG: {e}")),
    }
}

/// Fix the threshold. Called once per process, before the worker starts, so a
/// level cannot change under a running request.
pub fn init(level: Level) {
    let _ = LEVEL.set(level);
}

fn threshold() -> Level {
    #[cfg(test)]
    if let Some(level) = TEST_SINK.with(|s| s.borrow().as_ref().map(|t| t.level)) {
        return level;
    }
    LEVEL.get().copied().unwrap_or(Level::Warn)
}

/// Whether a line at `level` would be printed. Exposed so a caller can skip
/// building an expensive message.
pub fn enabled(level: Level) -> bool {
    level != Level::Off && level <= threshold()
}

/// One structured event: `level=<lvl> event=<name> <key>=<value> …`.
///
/// Values are logfmt-quoted when they hold whitespace or a quote, so a
/// `reason="length limit"` (or any future text field) cannot split the line.
pub fn event(level: Level, name: &str, fields: &[(&str, &str)]) {
    if !enabled(level) {
        return;
    }
    let mut line = format!("level={} event={}", level.as_str(), name);
    for (key, value) in fields {
        line.push(' ');
        line.push_str(key);
        line.push('=');
        push_value(&mut line, value);
    }
    emit(&line);
}

fn push_value(line: &mut String, value: &str) {
    if value.is_empty() || value.chars().any(|c| c.is_whitespace() || c == '"') {
        line.push_str(&format!("{value:?}"));
    } else {
        line.push_str(value);
    }
}

/// A free-form diagnostic, keeping the text each call site already had.
pub fn msg(level: Level, text: &str) {
    if !enabled(level) {
        return;
    }
    emit(&format!("level={} msg={:?}", level.as_str(), text));
}

// The level is checked **before** `to_string`: these are called on the per-step
// path (`decode step`, `prefill fed …`), and a filtered run must not pay an
// allocation for a line it will not print.
macro_rules! diagnostic {
    ($name:ident, $level:expr) => {
        pub fn $name(text: impl std::fmt::Display) {
            if enabled($level) {
                msg($level, &text.to_string());
            }
        }
    };
}

diagnostic!(error, Level::Error);
diagnostic!(warn, Level::Warn);
diagnostic!(info, Level::Info);
diagnostic!(debug, Level::Debug);

fn emit(line: &str) {
    #[cfg(test)]
    {
        let captured = TEST_SINK.with(|s| {
            let mut sink = s.borrow_mut();
            match sink.as_mut() {
                Some(test) => {
                    test.lines.push(line.to_string());
                    true
                }
                None => false,
            }
        });
        if captured {
            return;
        }
    }
    eprintln!("{line}");
}

// The test seam: a **thread-local** sink, so a test that drives the worker on
// its own thread captures its own lines and cannot see (or mute) another test's.
// Same shape as `src/testfail`'s arm: arming never writes the process
// environment and never reaches another thread.
#[cfg(test)]
thread_local! {
    static TEST_SINK: std::cell::RefCell<Option<TestSink>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
struct TestSink {
    level: Level,
    lines: Vec<String>,
}

/// Run `f` with a thread-local log sink at `level`, returning its lines.
#[cfg(test)]
pub fn capture<T>(level: Level, f: impl FnOnce() -> T) -> (T, Vec<String>) {
    TEST_SINK.with(|s| {
        *s.borrow_mut() = Some(TestSink {
            level,
            lines: Vec::new(),
        })
    });
    let out = f();
    let lines = TEST_SINK
        .with(|s| s.borrow_mut().take().map(|t| t.lines))
        .unwrap_or_default();
    (out, lines)
}

#[cfg(test)]
mod tests;

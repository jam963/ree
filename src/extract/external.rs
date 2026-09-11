use crate::{config::ExtractorConfig, util::read_limited};
use anyhow::{Context, Result, bail, ensure};
use std::{
    io::{Read, Write},
    os::unix::process::{CommandExt, ExitStatusExt},
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

/// Executes an argument array, never a shell. Both pipes are drained concurrently;
/// a process group is killed on timeout/overflow, including converter descendants.
pub fn run(command: &[String], cwd: Option<&Path>, timeout: Duration, cap: u64) -> Result<Vec<u8>> {
    run_bounded(command, cwd, timeout, cap, None)
}
pub fn run_bounded(
    command: &[String],
    cwd: Option<&Path>,
    timeout: Duration,
    cap: u64,
    disk_limit: Option<(&Path, u64)>,
) -> Result<Vec<u8>> {
    run_captured(command, cwd, timeout, cap, disk_limit).map(|(stdout, _)| stdout)
}
fn run_captured(
    command: &[String],
    cwd: Option<&Path>,
    timeout: Duration,
    cap: u64,
    disk_limit: Option<(&Path, u64)>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(!command.is_empty(), "empty extractor command");
    let mut cmd = Command::new(&command[0]);
    cmd.args(&command[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("LC_ALL", "C")
        .process_group(0);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    }
    if command[0] == "git" {
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("GIT_") {
                cmd.env_remove(key);
            }
        }
        cmd.env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_TERMINAL_PROMPT", "0");
    }
    if let Some((_, bytes)) = disk_limit {
        // Polling limits aggregate workspace growth; RLIMIT_FSIZE additionally
        // prevents a single fast write from exhausting disk between samples.
        unsafe {
            cmd.pre_exec(move || {
                let mut limit = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: 0,
                };
                if libc::getrlimit(libc::RLIMIT_FSIZE, &mut limit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                limit.rlim_cur = limit.rlim_cur.min(bytes);
                limit.rlim_max = limit.rlim_max.min(bytes);
                if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("missing_extractor: cannot execute {}", command[0]))?;
    let pid = child.id() as i32;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let out = std::thread::spawn(move || {
        let r = read_limited(stdout, cap);
        let _ = tx.send(r);
    });
    let err = std::thread::spawn(move || {
        let mut stderr = stderr;
        let mut kept = Vec::new();
        let mut buf = [0; 4096];
        while let Ok(n) = stderr.read(&mut buf) {
            if n == 0 {
                break;
            }
            let remaining = 16384usize.saturating_sub(kept.len());
            kept.extend_from_slice(&buf[..n.min(remaining)]);
        }
        kept
    });
    let start = Instant::now();
    let mut output = None;
    let mut disk_check = Instant::now();
    let check_disk = || -> Result<()> {
        if let Some((path, limit)) = disk_limit {
            let mut total = 0u64;
            for entry in walkdir::WalkDir::new(path).follow_links(false) {
                let entry = entry?;
                if entry.file_type().is_file() {
                    total = total.saturating_add(entry.metadata()?.len());
                    ensure!(
                        total <= limit,
                        "size_limit: helper workspace exceeds {limit} bytes"
                    );
                }
            }
        }
        Ok(())
    };
    let result = (|| -> Result<()> {
        loop {
            if disk_check.elapsed() >= Duration::from_millis(100) {
                check_disk()?;
                disk_check = Instant::now();
            }
            if output.is_none() {
                match rx.try_recv() {
                    Ok(r) => output = Some(r?),
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        bail!("extractor output reader failed")
                    }
                    Err(_) => {}
                }
            }
            if let Some(status) = child.try_wait()? {
                ensure!(
                    status.signal() != Some(libc::SIGXFSZ),
                    "size_limit: helper exceeded per-file workspace limit"
                );
                ensure!(status.success(), "extractor failed with {status}");
                if output.is_some() {
                    check_disk()?;
                    return Ok(());
                }
            }
            ensure!(
                start.elapsed() < timeout,
                "extractor_timeout: exceeded {} seconds",
                timeout.as_secs()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    })();
    // Also terminate descendants that inherited pipes after their parent exited.
    unsafe {
        libc::kill(-pid, libc::SIGKILL);
    }
    let _ = child.wait();
    let _ = out.join();
    let diagnostic = err.join().unwrap_or_default();
    result.with_context(|| format!("{}: {}", command[0], String::from_utf8_lossy(&diagnostic)))?;
    Ok((output.unwrap_or_default(), diagnostic))
}
pub fn extract(bytes: &[u8], extension: &str, config: &ExtractorConfig) -> Result<String> {
    // Converters see a private immutable snapshot, never a changing source file.
    ensure!(
        extension.len() <= 128
            && extension
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '+')),
        "unsupported: unsafe extractor extension"
    );
    ensure!(
        bytes.len() as u64 <= config.max_temp_bytes,
        "size_limit: snapshot exceeds helper workspace limit"
    );
    let directory = tempfile::tempdir()?;
    let path = directory.path().join(format!("input.{extension}"));
    let mut file = std::fs::File::create(&path)?;
    file.write_all(bytes)?;
    drop(file);
    let command: Vec<_> = config
        .command
        .iter()
        .map(|arg| arg.replace("{path}", &path.to_string_lossy()))
        .collect();
    let bytes = run_bounded(
        &command,
        Some(directory.path()),
        Duration::from_secs(config.timeout_seconds),
        config.max_output_bytes,
        Some((directory.path(), config.max_temp_bytes)),
    )?;
    String::from_utf8(bytes).context("extractor output is not UTF-8")
}
/// Probe only recognized version contracts, never arbitrary custom commands.
/// Cached per process; a missing/broken version probe must not fail extraction.
pub fn version(program: &str) -> Option<String> {
    use std::{
        collections::HashMap,
        sync::{Mutex, OnceLock},
    };
    static CACHE: OnceLock<Mutex<HashMap<String, Option<String>>>> = OnceLock::new();
    let flag = match program {
        "pdfinfo" | "pdftotext" | "pdftoppm" => "-v",
        "pandoc" | "tesseract" | "libreoffice" | "git" => "--version",
        _ => return None,
    };
    let mut cache = CACHE.get_or_init(Default::default).lock().ok()?;
    cache
        .entry(program.into())
        .or_insert_with(|| {
            let (out, err) = run_captured(
                &[program.into(), flag.into()],
                None,
                Duration::from_secs(2),
                16384,
                None,
            )
            .ok()?;
            let bytes = if out.is_empty() { &err } else { &out };
            String::from_utf8_lossy(bytes)
                .lines()
                .next()
                .map(|s| s.chars().take(512).collect())
        })
        .clone()
}
pub fn available(program: &str) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let executable = |p: &Path| {
        p.metadata()
            .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    };
    if program.contains('/') {
        return executable(Path::new(program));
    }
    std::env::var_os("PATH")
        .is_some_and(|path| std::env::split_paths(&path).any(|p| executable(&p.join(program))))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_output_and_runtime() {
        assert!(
            run(
                &[
                    "head".into(),
                    "-c".into(),
                    "10000".into(),
                    "/dev/zero".into()
                ],
                None,
                Duration::from_secs(2),
                32
            )
            .is_err()
        );
        let start = Instant::now();
        assert!(
            run(
                &["sleep".into(), "10".into()],
                None,
                Duration::from_millis(40),
                32
            )
            .is_err()
        );
        assert!(start.elapsed() < Duration::from_secs(2));
    }
    #[test]
    fn workspace_and_single_file_writes_are_bounded() {
        let temp = tempfile::tempdir().unwrap();
        let result = run_bounded(
            &[
                "dd".into(),
                "if=/dev/zero".into(),
                "of=large".into(),
                "bs=4096".into(),
                "count=16".into(),
            ],
            Some(temp.path()),
            Duration::from_secs(2),
            1024,
            Some((temp.path(), 4096)),
        );
        assert!(result.is_err());
        assert!(temp.path().join("large").metadata().unwrap().len() <= 4096);
        // Existing aggregate growth is also rejected even if the helper exits
        // before the polling interval, and links outside are never traversed.
        std::fs::write(temp.path().join("extra"), b"x").unwrap();
        std::os::unix::fs::symlink("/", temp.path().join("outside")).unwrap();
        let result = run_bounded(
            &["true".into()],
            Some(temp.path()),
            Duration::from_secs(2),
            1024,
            Some((temp.path(), 4096)),
        );
        assert!(format!("{:#}", result.unwrap_err()).contains("size_limit"));
    }
    #[test]
    fn inherited_pipes_do_not_outlive_timeout() {
        let start = Instant::now();
        // The shell is an explicit test fixture, never constructed from input.
        let result = run(
            &["sh".into(), "-c".into(), "sleep 10 & exit 0".into()],
            None,
            Duration::from_millis(60),
            32,
        );
        assert!(format!("{:#}", result.unwrap_err()).contains("extractor_timeout"));
        assert!(start.elapsed() < Duration::from_secs(2));
    }
    #[test]
    fn extension_cannot_escape_snapshot_directory() {
        assert!(extract(b"bytes", "../../../escape", &ExtractorConfig::default()).is_err());
        assert!(version("arbitrary-untrusted-helper").is_none());
    }
    #[test]
    fn arguments_not_shell() {
        assert_eq!(
            run(
                &["printf".into(), "%s".into(), "$(touch nope); x".into()],
                None,
                Duration::from_secs(2),
                100
            )
            .unwrap(),
            b"$(touch nope); x"
        );
    }
}

use crate::{config::ExtractorConfig, util::read_limited};
use anyhow::{Context, Result, bail, ensure};
use std::{
    io::{Read, Write},
    os::unix::process::CommandExt,
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
    ensure!(!command.is_empty(), "empty extractor command");
    let mut cmd = Command::new(&command[0]);
    cmd.args(&command[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
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
                        "git_size_limit: checkout exceeds {limit} bytes"
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
    Ok(output.unwrap_or_default())
}
pub fn extract(bytes: &[u8], extension: &str, config: &ExtractorConfig) -> Result<String> {
    // Converters see a private immutable snapshot, never a changing source file.
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
    let bytes = run(
        &command,
        Some(directory.path()),
        Duration::from_secs(config.timeout_seconds),
        config.max_output_bytes,
    )?;
    String::from_utf8(bytes).context("extractor output is not UTF-8")
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

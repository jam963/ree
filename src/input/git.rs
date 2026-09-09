use crate::{config::Config, extract::external, util::hash};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

pub fn checkout(
    input: &str,
    config: &Config,
) -> Result<(PathBuf, Value, crate::storage::WriterLock)> {
    let url = super::url::normalize(input)?;
    ensure!(
        url.scheme() == "https",
        "remote Git requires HTTPS; SSH and local Git transports are not executed"
    );
    let addresses = super::url::addresses(&url, config.allow_private_network)?;
    let mut base: Vec<String> = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "protocol.ext.allow=never",
        "-c",
        "http.followRedirects=false",
        "-c",
        "http.proxy=",
        "-c",
        "credential.helper=",
        "-c",
        "submodule.recurse=false",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // libcurl's resolver override pins Git to a checked address, too. Redirects
    // are disabled because Git cannot revalidate each redirect in this wrapper.
    let ip = addresses[0].ip();
    let ip = if ip.is_ipv6() {
        format!("[{ip}]")
    } else {
        ip.to_string()
    };
    base.extend([
        "-c".into(),
        format!(
            "http.curloptResolve={}:{}:{ip}",
            url.host_str().unwrap(),
            url.port_or_known_default().unwrap()
        ),
    ]);
    let repositories = config.cache.join("repositories");
    std::fs::create_dir_all(&repositories)?;
    let dest = repositories.join(hash(url.as_str().as_bytes()));
    let lease = crate::storage::WriterLock::acquire(&dest)?;
    let run = |args: &[&str], cwd: Option<&Path>, disk: &Path| -> Result<Vec<u8>> {
        let mut cmd = base.clone();
        cmd.extend(args.iter().map(|s| s.to_string()));
        external::run_bounded(
            &cmd,
            cwd,
            Duration::from_secs(120),
            1024 * 1024,
            Some((disk, 512 * 1024 * 1024)),
        )
    };
    if !dest.exists() {
        let temp = tempfile::tempdir_in(&repositories)?;
        let work = temp.path().join("checkout");
        run(
            &[
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--no-tags",
                "--",
                url.as_str(),
                work.to_str().unwrap(),
            ],
            None,
            temp.path(),
        )?;
        std::fs::rename(&work, &dest)?;
    } else {
        // Fetch the original URL explicitly; do not trust a modified origin URL.
        run(
            &[
                "fetch",
                "--depth",
                "1",
                "--no-tags",
                "--",
                url.as_str(),
                "HEAD",
            ],
            Some(&dest),
            &dest,
        )?;
        run(&["reset", "--hard", "FETCH_HEAD"], Some(&dest), &dest)?;
        run(&["clean", "-fdx"], Some(&dest), &dest)?;
    }
    let commit = String::from_utf8(run(&["rev-parse", "HEAD"], Some(&dest), &dest)?)?
        .trim()
        .to_string();
    let branch = String::from_utf8(run(
        &["symbolic-ref", "--short", "HEAD"],
        Some(&dest),
        &dest,
    )?)?
    .trim()
    .to_string();
    Ok((
        dest,
        json!({"repository_url":url.as_str(),"branch":branch,"commit":commit}),
        lease,
    ))
}

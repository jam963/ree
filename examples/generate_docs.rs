//! Generate a man page and shell completions from the actual CLI definition.
use anyhow::Result;
use clap::CommandFactory;
use ree::cli::Cli;
fn main() -> Result<()> {
    let out = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "target/cli-docs".into());
    std::fs::create_dir_all(&out)?;
    clap_mangen::Man::new(Cli::command()).render(&mut std::fs::File::create(out.join("ree.1"))?)?;
    for shell in [
        clap_complete::Shell::Bash,
        clap_complete::Shell::Zsh,
        clap_complete::Shell::Fish,
        clap_complete::Shell::Elvish,
        clap_complete::Shell::PowerShell,
    ] {
        clap_complete::generate_to(shell, &mut Cli::command(), "ree", &out)?;
    }
    eprintln!("Generated man page and completions in {}", out.display());
    Ok(())
}

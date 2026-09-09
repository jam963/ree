use clap::Parser;
use ree::{cli::Cli, events::Events};
use serde_json::json;
fn main() -> std::process::ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) {
                let _ = e.print();
                return std::process::ExitCode::SUCCESS;
            }
            let mut events = Events::new(false, false, false);
            let _ = events.emit(
                json!({"type":"fatal","error_code":"invalid_arguments","message":e.to_string()}),
            );
            let _ = events.flush();
            return std::process::ExitCode::from(2);
        }
    };
    let mut events = Events::new(cli.options.quiet, cli.options.verbose, cli.options.progress);
    let code = match ree::run(cli, &mut events) {
        Ok(code) => code,
        Err(e) => {
            let code = ree::error::exit_code(&e);
            let _=events.emit(json!({"type":"fatal","error_code":if code==4{"writer_locked"}else if code==2{"invalid_configuration"}else{"fatal_error"},"message":format!("{e:#}")}));
            code
        }
    };
    if events.flush().is_err() {
        return std::process::ExitCode::from(3);
    }
    std::process::ExitCode::from(code)
}

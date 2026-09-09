use anyhow::Result;
use serde_json::{Value, json};
use std::io::{BufWriter, IsTerminal, Write};

pub struct Events {
    output: BufWriter<Box<dyn Write + Send>>,
    pub quiet: bool,
    pub verbose: bool,
    progress: bool,
}
impl Events {
    pub fn new(quiet: bool, verbose: bool, progress: bool) -> Self {
        Self {
            output: BufWriter::new(Box::new(std::io::stdout())),
            quiet,
            verbose,
            progress: progress && std::io::stderr().is_terminal(),
        }
    }
    /// Inject a sink for structured-event tests or library embedding.
    pub fn with_writer(writer: impl Write + Send + 'static, quiet: bool, verbose: bool) -> Self {
        Self {
            output: BufWriter::new(Box::new(writer)),
            quiet,
            verbose,
            progress: false,
        }
    }
    pub fn emit(&mut self, event: Value) -> Result<()> {
        let error = event["type"] == "failed" || event["type"] == "fatal";
        if !self.quiet || error {
            serde_json::to_writer(&mut self.output, &event)?;
            writeln!(self.output)?;
        }
        if self.progress && !self.quiet {
            eprint!(
                "\rree: {}                ",
                event["type"].as_str().unwrap_or("working")
            );
        }
        Ok(())
    }
    pub fn failure(&mut self, source: &str, code: &str, message: &str) -> Result<()> {
        self.emit(json!({"type":"failed", "source":source, "error_code":code, "message":message}))
    }
    pub fn flush(&mut self) -> Result<()> {
        self.output.flush()?;
        if self.progress && !self.quiet {
            eprintln!();
        }
        Ok(())
    }
}

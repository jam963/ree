use std::fmt;
#[derive(Debug)]
pub struct AppError {
    pub code: u8,
    pub message: String,
}
impl AppError {
    pub fn new(code: u8, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}
impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl std::error::Error for AppError {}
pub fn exit_code(error: &anyhow::Error) -> u8 {
    error.downcast_ref::<AppError>().map_or(3, |e| e.code)
}

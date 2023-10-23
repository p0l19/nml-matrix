use std::fmt;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct NmlError {
    error_type: ErrorKind,
}

impl NmlError {
    pub fn new(error_type: ErrorKind) -> Self {
        Self {
            error_type,
        }
    }
}

impl Display for NmlError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.error_type)
    }
}

#[derive(Debug, Clone)]
pub enum ErrorKind {
    InvalidRows,
    InvalidCols,
    CreateMatrix
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ErrorKind::InvalidCols => write!(f, "Invalid number of columns"),
            ErrorKind::InvalidRows => write!(f, "Invalid number of rows"),
            ErrorKind::CreateMatrix => write!(f, "Unable to create matrix"),
        }
    }
}
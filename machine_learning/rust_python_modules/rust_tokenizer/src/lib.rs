
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Tokenizer object to tokenize Python strings into a list of characters
#[pyclass]
pub struct Tokenizer {}

#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new() -> Self {
        Tokenizer {}
    }

    /// Takes a Python string and returns a list of characters
    pub fn tokenize(&self, text: &PyString) -> PyResult<Vec<String>> {
        let text_str = text.to_string();
        let tokens: Vec<String> = text_str.chars().map(|ch| ch.to_string()).collect();
        Ok(tokens)
    }
}

/// A Python module implemented in Rust.
#[pymodule(rust_tokenizer)]
fn tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}

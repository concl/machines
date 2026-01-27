
use pyo3::prelude::*;
use pyo3::types::PyString;

// Base trait for all tokenizers
/// This trait defines the basic functionality for a tokenizer
trait Tokenizer {
    fn from_pretrained(path: &str) -> Self where Self: Sized;
    fn save_trained(&self, path: &str);
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// Tokenizer object to tokenize Python strings into a list of characters
#[pyclass]
pub struct CharacterLevelTokenizer {}

#[pymethods]
impl CharacterLevelTokenizer {
    #[new]
    pub fn new() -> Self {
        CharacterLevelTokenizer {}
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
    m.add_class::<CharacterLevelTokenizer>()?;
    Ok(())
}


use pyo3::prelude::*;
use pyo3::types::PyString;

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

#[pyclass]
pub struct SentencePieceTokenizer {
    #[pyo3(get)]
    pub model_path: String,
    pub vocab: HashMap<String, i64>,
    pub unk_token: String,
    pub unk_token_id: i64,
    pub pad_token: String,
    pub pad_token_id: i64,
}

// #[pymethods]
// impl SentencePieceTokenizer {
//     #[new]
//     pub fn new(model_path: String) -> Self {
//         SentencePieceTokenizer { model_path }
//     }

//     /// Takes a Python string and returns a list of tokens
//     pub fn tokenize(&self, text: &PyString) -> PyResult<Vec<String>> {
//         // todo


//         Ok(vec![]);
//     }
// }

/// A Python module implemented in Rust.
#[pymodule(rust_tokenizer)]
fn tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CharacterLevelTokenizer>()?;
    Ok(())
}

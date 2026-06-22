
use pyo3::prelude::*;

/// Trait defining the core tokenizer interface.
trait Tokenizer {
    /// Load a tokenizer from a saved directory / file.
    fn from_pretrained(path: &str) -> Self
    where
        Self: Sized;
    /// Persist the tokenizer to disk.
    fn save_pretrained(&self, path: &str);
    /// Split a string into tokens.
    fn tokenize(&self, text: &str) -> Vec<usize>;
}

/// A simple character-level tokenizer — every Unicode character becomes a
/// separate token.  Exposed to Python via PyO3.
#[pyclass(name = "CharacterLevelTokenizer")]
pub struct CharacterLevelTokenizer;

impl Tokenizer for CharacterLevelTokenizer {
    fn from_pretrained(_path: &str) -> Self {
        // This dummy tokenizer has no parameters to load.
        Self
    }

    fn save_pretrained(&self, _path: &str) {
        // No-op — nothing to persist.
    }

    fn tokenize(&self, text: &str) -> Vec<usize> {
        text.chars().map(|ch| { ch as usize }).collect()
    }
}

#[pymethods]
impl CharacterLevelTokenizer {
    /// Create a new character-level tokenizer.
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Python-facing tokenizer: takes a Python ``str``, returns ``list[str]``.
    fn tokenize(&self, text: &str) -> PyResult<Vec<usize>> {
        // Delegate to the Rust trait implementation.
        Ok(Tokenizer::tokenize(self, text))
    }

    fn __repr__(&self) -> String {
        "CharacterLevelTokenizer()".into()
    }
}

/// This function is the entry-point called by Python when the native module is
/// imported.  The function name **must** match the ``[lib] name`` in
/// ``Cargo.toml`` (i.e. ``rust_tokenizers``).
#[pymodule]
fn rust_tokenizers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CharacterLevelTokenizer>()?;
    Ok(())
}

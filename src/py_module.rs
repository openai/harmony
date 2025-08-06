//! Python bindings for the harmony crate.
//!
//! The bindings are kept intentionally small: we expose the `HarmonyEncoding` type
//! together with the operations that are required by the original Rust test
//! suite (rendering a conversation for completion, parsing messages from
//! completion tokens and decoding tokens back into UTF-8). All higher-level
//! data-structures (Conversation, Message, SystemContent, DeveloperContent, …) are passed across the FFI
//! boundary as JSON.  This allows us to keep the Rust ↔ Python interface very
//! light-weight while still re-using the exact same logic that is implemented
//! in Rust.
//!
//! A thin, typed, user-facing Python wrapper around these low-level bindings is
//! provided in `harmony/__init__.py`.

use pyo3::prelude::*;

// We need the `Python` type later on.
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::Python;

use pyo3::types::{PyAny, PyDict, PyModule};
use pyo3::Bound;

// Define a custom Python exception so users can catch Harmony specific errors.
create_exception!(openai_harmony, HarmonyError, PyRuntimeError);

use crate::{
    chat::{Message, Role, ToolNamespaceConfig},
    encoding::{HarmonyEncoding, StreamableParser},
    load_harmony_encoding, HarmonyEncodingName,
};

/// A thin PyO3 wrapper around the Rust `HarmonyEncoding` struct.
#[pyclass]
struct PyHarmonyEncoding {
    inner: HarmonyEncoding,
}

/// Streaming parser exposed to Python.
#[pyclass]
struct PyStreamableParser {
    inner: StreamableParser,
}

#[pyclass]
pub enum PyStreamState {
    ExpectStart,
    Header,
    Content,
}

#[pymethods]
impl PyHarmonyEncoding {
    /// Create a new `HarmonyEncoding` by name.
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let parsed: HarmonyEncodingName = name
            .parse::<HarmonyEncodingName>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let encoding = load_harmony_encoding(parsed)
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        Ok(Self { inner: encoding })
    }

    /// Return the name of the encoding.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Render a conversation (in JSON format) for completion.
    ///
    /// Parameters
    /// ----------
    /// conversation_json : str
    ///     A JSON encoded `Conversation` (as produced by `serde_json`).
    /// next_turn_role : str
    ///     The role of the *next* turn (e.g. "assistant").
    /// config : dict (optional)
    ///     Optional config dict. Only supports 'auto_drop_analysis' (bool).
    ///
    /// Returns
    /// -------
    /// List[int]
    ///     The encoded token sequence.
    fn render_conversation_for_completion(
        &self,
        conversation_json: &str,
        next_turn_role: &str,
        config: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u32>> {
        // Deserialize the conversation first.
        let conversation: crate::chat::Conversation = serde_json::from_str(conversation_json)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "invalid conversation JSON: {e}"
                ))
            })?;

        // Convert the role string into the `Role` enum.
        let role = Role::try_from(next_turn_role).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown role: {next_turn_role}"
            ))
        })?;

        // Parse config
        let rust_config = if let Some(cfg_dict) = config {
            let auto_drop_analysis = cfg_dict
                .get_item("auto_drop_analysis")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(true);
            Some(crate::encoding::RenderConversationConfig { auto_drop_analysis })
        } else {
            None
        };

        self.inner
            .render_conversation_for_completion(&conversation, role, rust_config.as_ref())
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    /// Render a conversation without appending a new role.
    fn render_conversation(
        &self,
        conversation_json: &str,
        config: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u32>> {
        let conversation: crate::chat::Conversation = serde_json::from_str(conversation_json)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "invalid conversation JSON: {e}"
                ))
            })?;

        let rust_config = if let Some(cfg_dict) = config {
            let auto_drop_analysis = cfg_dict
                .get_item("auto_drop_analysis")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(true);
            Some(crate::encoding::RenderConversationConfig { auto_drop_analysis })
        } else {
            None
        };

        self.inner
            .render_conversation(&conversation, rust_config.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Render a conversation for training.
    fn render_conversation_for_training(
        &self,
        conversation_json: &str,
        config: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u32>> {
        let conversation: crate::chat::Conversation = serde_json::from_str(conversation_json)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "invalid conversation JSON: {e}"
                ))
            })?;

        let rust_config = if let Some(cfg_dict) = config {
            let auto_drop_analysis = cfg_dict
                .get_item("auto_drop_analysis")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(true);
            Some(crate::encoding::RenderConversationConfig { auto_drop_analysis })
        } else {
            None
        };

        self.inner
            .render_conversation_for_training(&conversation, rust_config.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Render a single message into tokens.
    fn render(
        &self,
        message_json: &str,
        render_options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u32>> {
        let message: crate::chat::Message = serde_json::from_str(message_json).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid message JSON: {e}"))
        })?;

        let rust_options = if let Some(options_dict) = render_options {
            let conversation_has_function_tools = options_dict
                .get_item("conversation_has_function_tools")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(false);
            Some(crate::encoding::RenderOptions {
                conversation_has_function_tools,
            })
        } else {
            None
        };

        self.inner
            .render(&message, rust_options.as_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Given a list of completion tokens, parse them back into a sequence of
    /// messages.  The result is returned as a JSON string which can be
    /// deserialised on the Python side.
    #[allow(clippy::needless_pass_by_value)]
    fn parse_messages_from_completion_tokens(
        &self,
        tokens: Vec<u32>,
        role: Option<&str>,
    ) -> PyResult<String> {
        let role_parsed = if let Some(r) = role {
            Some(Role::try_from(r).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("unknown role: {r}"))
            })?)
        } else {
            None
        };

        let messages: Vec<Message> = self
            .inner
            .parse_messages_from_completion_tokens(tokens, role_parsed)
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;

        serde_json::to_string(&messages).map_err(|e| {
            PyErr::new::<HarmonyError, _>(format!("failed to serialise messages to JSON: {e}"))
        })
    }

    /// Decode a sequence of tokens into text using the underlying tokenizer.
    fn decode_utf8(&self, tokens: Vec<u32>) -> PyResult<String> {
        self.inner
            .tokenizer()
            .decode_utf8(tokens)
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    /// Decode a sequence of tokens into raw bytes using the underlying tokenizer.
    fn decode_bytes(&self, tokens: Vec<u32>) -> PyResult<Vec<u8>> {
        self.inner
            .tokenizer()
            .decode_bytes(tokens)
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    /// Encode text into tokens using the underlying tokenizer with a set of allowed special tokens.
    fn encode(&self, text: &str, allowed_special: Option<Bound<'_, PyAny>>) -> PyResult<Vec<u32>> {
        let allowed_vec: Vec<String> = match allowed_special {
            Some(obj) => obj.extract::<Vec<String>>().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "invalid allowed_special: {e}"
                ))
            })?,
            None => Vec::new(),
        };
        let allowed_set: std::collections::HashSet<&str> =
            allowed_vec.iter().map(|s| s.as_str()).collect();
        Ok(self.inner.tokenizer().encode(text, &allowed_set).0)
    }

    /// Return the list of special tokens for this tokenizer.
    fn special_tokens(&self) -> Vec<String> {
        self.inner
            .tokenizer()
            .special_tokens()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// Check whether a token id corresponds to a special token.
    fn is_special_token(&self, token: u32) -> bool {
        self.inner.tokenizer().is_special_token(token)
    }

    /// Return the stop tokens for the encoding.
    fn stop_tokens(&self) -> PyResult<Vec<u32>> {
        self.inner
            .stop_tokens()
            .map(|set| set.into_iter().collect())
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    /// Return the stop tokens for assistant actions.
    fn stop_tokens_for_assistant_actions(&self) -> PyResult<Vec<u32>> {
        self.inner
            .stop_tokens_for_assistant_actions()
            .map(|set| set.into_iter().collect())
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }
}

#[pymethods]
impl PyStreamableParser {
    #[new]
    fn new(encoding: &PyHarmonyEncoding, role: Option<&str>) -> PyResult<Self> {
        let parsed_role = role
            .map(|r| {
                Role::try_from(r).map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("unknown role: {r}"))
                })
            })
            .transpose()?;
        let inner = StreamableParser::new(encoding.inner.clone(), parsed_role)
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    fn process(&mut self, token: u32) -> PyResult<()> {
        self.inner
            .process(token)
            .map(|_| ())
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    fn process_eos(&mut self) -> PyResult<()> {
        self.inner
            .process_eos()
            .map(|_| ())
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    #[getter]
    fn current_content(&self) -> PyResult<String> {
        self.inner
            .current_content()
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    #[getter]
    fn current_role(&self) -> Option<String> {
        self.inner.current_role().map(|r| r.as_str().to_string())
    }

    #[getter]
    fn current_content_type(&self) -> Option<String> {
        self.inner.current_content_type()
    }

    #[getter]
    fn last_content_delta(&self) -> PyResult<Option<String>> {
        self.inner
            .last_content_delta()
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    #[getter]
    fn messages(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.messages()).map_err(|e| {
            PyErr::new::<HarmonyError, _>(format!("failed to serialise messages to JSON: {e}"))
        })
    }

    #[getter]
    fn tokens(&self) -> Vec<u32> {
        self.inner.tokens().to_vec()
    }

    #[getter]
    fn state(&self) -> PyResult<String> {
        self.inner
            .state_json()
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))
    }

    #[getter]
    fn current_recipient(&self) -> Option<String> {
        self.inner.current_recipient()
    }

    #[getter]
    fn current_channel(&self) -> Option<String> {
        self.inner.current_channel()
    }
}

/// Python module definition.
#[pymodule]
fn openai_harmony(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register the PyHarmonyEncoding class.
    m.add_class::<PyHarmonyEncoding>()?;
    m.add_class::<PyStreamableParser>()?;
    m.add_class::<PyStreamState>()?;
    m.add("HarmonyError", _py.get_type::<HarmonyError>())?;

    // Convenience function mirroring the Rust-side `load_harmony_encoding` but
    // returning an *instance* of `PyHarmonyEncoding`.
    #[pyfunction(name = "load_harmony_encoding")]
    fn load_harmony_encoding_py(py: Python<'_>, name: &str) -> PyResult<Py<PyHarmonyEncoding>> {
        let enc = PyHarmonyEncoding::new(name)?;
        Py::new(py, enc)
    }
    m.add_function(pyo3::wrap_pyfunction!(load_harmony_encoding_py, m)?)?;

    // Convenience function to load a HarmonyEncoding from a local vocab file for offline
    // scenarios or reproducible builds where remote download is not possible.
    #[pyfunction(name = "load_harmony_encoding_from_file")]
    fn load_harmony_encoding_from_file_py(
        py: Python<'_>,
        name: &str,
        vocab_file: &str,
        special_tokens: Vec<(String, u32)>,
        pattern: &str,
        n_ctx: usize,
        max_message_tokens: usize,
        max_action_length: usize,
        expected_hash: Option<&str>,
    ) -> PyResult<Py<PyHarmonyEncoding>> {
        let encoding = HarmonyEncoding::from_local_file(
            name.to_string(),
            std::path::Path::new(vocab_file),
            expected_hash,
            special_tokens,
            pattern,
            n_ctx,
            max_message_tokens,
            max_action_length,
        )
        .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        Py::new(py, PyHarmonyEncoding { inner: encoding })
    }
    m.add_function(pyo3::wrap_pyfunction!(
        load_harmony_encoding_from_file_py,
        m
    )?)?;

    // Convenience functions to get the tool configs for the browser and python tools.
    #[pyfunction]
    fn get_tool_namespace_config(py: Python<'_>, tool: &str) -> PyResult<PyObject> {
        let cfg = match tool {
            "browser" => ToolNamespaceConfig::browser(),
            "python" => ToolNamespaceConfig::python(),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown tool namespace: {tool}"
                )));
            }
        };
        let py_cfg =
            serde_json::to_value(&cfg).map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        let json_str = serde_json::to_string(&py_cfg)
            .map_err(|e| PyErr::new::<HarmonyError, _>(e.to_string()))?;
        let json_mod = PyModule::import(py, "json")?;
        let py_obj = json_mod.call_method1("loads", (json_str,))?;
        Ok(py_obj.into())
    }
    m.add_function(pyo3::wrap_pyfunction!(get_tool_namespace_config, m)?)?;

    Ok(())
}

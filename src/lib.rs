#![doc = include_str!("../README.md")]

pub mod chat;
mod encoding;
mod registry;
mod tiktoken;
pub mod tiktoken_ext;

pub use encoding::{HarmonyEncoding, ParseOptions, StreamableParser};
pub use registry::load_harmony_encoding;
pub use registry::HarmonyEncodingName;

#[cfg(test)]
pub mod tests;

#[cfg(feature = "python-binding")]
mod py_module;

#[cfg(feature = "wasm-binding")]
mod wasm_module;

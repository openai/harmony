use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    encoding::{FormattingToken, HarmonyEncoding},
    tiktoken_ext,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum HarmonyEncodingName {
    HarmonyGptOss,
}

impl std::fmt::Display for HarmonyEncodingName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                HarmonyEncodingName::HarmonyGptOss => "HarmonyGptOss",
            }
        )
    }
}

impl std::str::FromStr for HarmonyEncodingName {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "HarmonyGptOss" => Ok(HarmonyEncodingName::HarmonyGptOss),
            _ => anyhow::bail!("Invalid HarmonyEncodingName: {}", s),
        }
    }
}

impl std::fmt::Debug for HarmonyEncodingName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// Build the encoding around an already-loaded tokenizer.
///
/// The native and wasm32 loaders differ only in how they obtain the tokenizer
/// (blocking vs async), so everything else lives here. Keeping two copies of
/// this struct literal is what let the wasm32 branch drift: it went on setting
/// a `conversation_has_function_tools` field for a year after d00ac3d removed
/// it from `HarmonyEncoding`, and nothing built wasm32 to notice.
fn build_harmony_encoding(
    name: HarmonyEncodingName,
    tokenizer: crate::tiktoken::CoreBPE,
    tokenizer_name: String,
) -> HarmonyEncoding {
    let n_ctx = 1_048_576; // 2^20
    let max_action_length = 524_288; // 2^19
    HarmonyEncoding {
        name: name.to_string(),
        n_ctx,
        tokenizer: Arc::new(tokenizer),
        tokenizer_name,
        max_message_tokens: n_ctx - max_action_length,
        max_action_length,
        format_token_mapping: make_mapping([
            (FormattingToken::Start, "<|start|>"),
            (FormattingToken::Message, "<|message|>"),
            (FormattingToken::EndMessage, "<|end|>"),
            (FormattingToken::EndMessageDoneSampling, "<|return|>"),
            (FormattingToken::Refusal, "<|refusal|>"),
            (FormattingToken::ConstrainedFormat, "<|constrain|>"),
            (FormattingToken::Channel, "<|channel|>"),
            (FormattingToken::EndMessageAssistantToTool, "<|call|>"),
            (FormattingToken::BeginUntrusted, "<|untrusted|>"),
            (FormattingToken::EndUntrusted, "<|end_untrusted|>"),
        ]),
        stop_formatting_tokens: HashSet::from([
            FormattingToken::EndMessageDoneSampling,
            FormattingToken::EndMessageAssistantToTool,
            FormattingToken::EndMessage,
        ]),
        stop_formatting_tokens_for_assistant_actions: HashSet::from([
            FormattingToken::EndMessageDoneSampling,
            FormattingToken::EndMessageAssistantToTool,
        ]),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_harmony_encoding(name: HarmonyEncodingName) -> anyhow::Result<HarmonyEncoding> {
    match name {
        HarmonyEncodingName::HarmonyGptOss => {
            let encoding_ext = tiktoken_ext::Encoding::O200kHarmony;
            Ok(build_harmony_encoding(
                name,
                encoding_ext.load()?,
                encoding_ext.name().to_owned(),
            ))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn load_harmony_encoding(name: HarmonyEncodingName) -> anyhow::Result<HarmonyEncoding> {
    match name {
        HarmonyEncodingName::HarmonyGptOss => {
            let encoding_ext = tiktoken_ext::Encoding::O200kHarmony;
            Ok(build_harmony_encoding(
                name,
                encoding_ext.load().await?,
                encoding_ext.name().to_owned(),
            ))
        }
    }
}

fn make_mapping<I>(iter: I) -> HashMap<FormattingToken, String>
where
    I: IntoIterator<Item = (FormattingToken, &'static str)>,
{
    iter.into_iter().map(|(k, v)| (k, v.to_string())).collect()
}

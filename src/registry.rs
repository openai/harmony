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

#[cfg(all(not(target_arch = "wasm32"), feature = "network"))]
pub fn load_harmony_encoding(name: HarmonyEncodingName) -> anyhow::Result<HarmonyEncoding> {
    match name {
        HarmonyEncodingName::HarmonyGptOss => {
            let n_ctx = 1_048_576; // 2^20
            let max_action_length = 524_288; // 2^19
            let encoding_ext = tiktoken_ext::Encoding::O200kHarmony;
            Ok(HarmonyEncoding {
                name: name.to_string(),
                n_ctx,
                tokenizer: Arc::new(encoding_ext.load()?),
                tokenizer_name: encoding_ext.name().to_owned(),
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
            })
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "network"))]
pub async fn load_harmony_encoding(name: HarmonyEncodingName) -> anyhow::Result<HarmonyEncoding> {
    match name {
        HarmonyEncodingName::HarmonyGptOss => {
            let n_ctx = 1_048_576; // 2^20
            let max_action_length = 524_288; // 2^19
            let encoding_ext = tiktoken_ext::Encoding::O200kHarmony;
            Ok(HarmonyEncoding {
                name: name.to_string(),
                n_ctx,
                tokenizer: Arc::new(encoding_ext.load().await?),
                tokenizer_name: encoding_ext.name().to_owned(),
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
            })
        }
    }
}

/// Load a [`HarmonyEncoding`] from raw tiktoken vocab bytes.
///
/// This is useful in environments where filesystem access and async HTTP
/// are unavailable (e.g. `wasm32-unknown-unknown` without JS bindings).
pub fn load_harmony_encoding_from_vocab_bytes(
    name: HarmonyEncodingName,
    vocab_bytes: &[u8],
) -> anyhow::Result<HarmonyEncoding> {
    match name {
        HarmonyEncodingName::HarmonyGptOss => {
            let n_ctx = 1_048_576;
            let max_action_length = 524_288;
            let encoding_ext = tiktoken_ext::Encoding::O200kHarmony;
            let mut specials: Vec<(String, u32)> = encoding_ext
                .special_tokens()
                .iter()
                .map(|(s, r)| ((*s).to_string(), *r))
                .collect();
            specials.extend((200014..=201088).map(|id| (format!("<|reserved_{id}|>"), id)));
            let tokenizer = tiktoken_ext::load_encoding_from_bytes(
                vocab_bytes,
                None,
                specials,
                &encoding_ext.pattern(),
            )?;
            Ok(HarmonyEncoding {
                name: name.to_string(),
                n_ctx,
                tokenizer: Arc::new(tokenizer),
                tokenizer_name: encoding_ext.name().to_owned(),
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
            })
        }
    }
}

fn make_mapping<I>(iter: I) -> HashMap<FormattingToken, String>
where
    I: IntoIterator<Item = (FormattingToken, &'static str)>,
{
    iter.into_iter().map(|(k, v)| (k, v.to_string())).collect()
}

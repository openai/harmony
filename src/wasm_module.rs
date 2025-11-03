use wasm_bindgen::prelude::*;

use crate::{
    chat::{Message, Role, ToolNamespaceConfig},
    encoding::{HarmonyEncoding, ParseOptions, StreamableParser},
    load_harmony_encoding as inner_load_harmony_encoding, HarmonyEncodingName,
};

use serde::Deserialize;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "Conversation")]
    pub type JsConversation;

    #[wasm_bindgen(typescript_type = "Message")]
    pub type JsMessage;

    #[wasm_bindgen(typescript_type = "RenderConversationConfig")]
    pub type JsRenderConversationConfig;

    #[wasm_bindgen(typescript_type = "RenderOptions")]
    pub type JsRenderOptions;
}

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND: &str = r#"
export interface Author {
  role: 'user' | 'assistant' | 'system' | 'developer' | 'tool';
  name?: string;
}

export type Content =
  | { type: 'text'; text: string }
  | { type: 'system_content'; model_identity?: string; reasoning_effort?: string; tools?: Record<string, ToolNamespaceConfig>; conversation_start_date?: string; knowledge_cutoff?: string }
  | { type: 'developer_content'; instructions?: string; tools?: Record<string, ToolNamespaceConfig> };

export interface Message {
  author: Author;
  content: Content[];
  channel?: string;
  recipient?: string;
  content_type?: string;
}

export interface Conversation {
  messages: Message[];
}

export interface RenderConversationConfig {
  auto_drop_analysis?: boolean;
}

export interface ToolNamespaceConfig {
  name: string;
  description?: string;
  tools: any[];
}
"#;

#[wasm_bindgen]
pub struct JsHarmonyEncoding {
    inner: HarmonyEncoding,
}

#[wasm_bindgen]
impl JsHarmonyEncoding {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[wasm_bindgen(js_name = renderConversationForCompletion)]
    pub fn render_conversation_for_completion(
        &self,
        conversation: JsConversation,
        next_turn_role: &str,
        config: JsRenderConversationConfig,
    ) -> Result<Vec<u32>, JsValue> {
        let conversation: JsValue = conversation.into();
        let conversation: crate::chat::Conversation = serde_wasm_bindgen::from_value(conversation)
            .map_err(|e| JsValue::from_str(&format!("invalid conversation JSON: {e}")))?;
        let role = Role::try_from(next_turn_role)
            .map_err(|_| JsValue::from_str(&format!("unknown role: {next_turn_role}")))?;
        #[derive(Deserialize)]
        struct Config {
            auto_drop_analysis: Option<bool>,
        }
        let config: JsValue = config.into();
        let rust_config = if config.is_undefined() || config.is_null() {
            None
        } else {
            let cfg: Config = serde_wasm_bindgen::from_value(config)
                .map_err(|e| JsValue::from_str(&format!("invalid config: {e}")))?;
            Some(crate::encoding::RenderConversationConfig {
                auto_drop_analysis: cfg.auto_drop_analysis.unwrap_or(true),
            })
        };
        self.inner
            .render_conversation_for_completion(&conversation, role, rust_config.as_ref())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = renderConversation)]
    pub fn render_conversation(
        &self,
        conversation: JsConversation,
        config: JsRenderConversationConfig,
    ) -> Result<Vec<u32>, JsValue> {
        let conversation: JsValue = conversation.into();
        let conversation: crate::chat::Conversation = serde_wasm_bindgen::from_value(conversation)
            .map_err(|e| JsValue::from_str(&format!("invalid conversation JSON: {e}")))?;
        #[derive(Deserialize)]
        struct Config {
            auto_drop_analysis: Option<bool>,
        }
        let config: JsValue = config.into();
        let rust_config = if config.is_undefined() || config.is_null() {
            None
        } else {
            let cfg: Config = serde_wasm_bindgen::from_value(config)
                .map_err(|e| JsValue::from_str(&format!("invalid config: {e}")))?;
            Some(crate::encoding::RenderConversationConfig {
                auto_drop_analysis: cfg.auto_drop_analysis.unwrap_or(true),
            })
        };
        self.inner
            .render_conversation(&conversation, rust_config.as_ref())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn render(
        &self,
        message: JsMessage,
        render_options: JsRenderOptions,
    ) -> Result<Vec<u32>, JsValue> {
        let message: JsValue = message.into();
        let message: crate::chat::Message = serde_wasm_bindgen::from_value(message)
            .map_err(|e| JsValue::from_str(&format!("invalid message JSON: {e}")))?;

        #[derive(Deserialize)]
        struct RenderOptions {
            conversation_has_function_tools: Option<bool>,
        }
        let render_options: JsValue = render_options.into();
        let rust_options = if render_options.is_undefined() || render_options.is_null() {
            None
        } else {
            let cfg: RenderOptions = serde_wasm_bindgen::from_value(render_options)
                .map_err(|e| JsValue::from_str(&format!("invalid render options: {e}")))?;
            Some(crate::encoding::RenderOptions {
                conversation_has_function_tools: cfg
                    .conversation_has_function_tools
                    .unwrap_or(false),
            })
        };

        self.inner
            .render(&message, rust_options.as_ref())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = parseMessagesFromCompletionTokens)]
    pub fn parse_messages_from_completion_tokens(
        &self,
        tokens: Vec<u32>,
        role: Option<String>,
        strict: Option<bool>,
    ) -> Result<String, JsValue> {
        let role_parsed = if let Some(r) = role {
            Some(
                Role::try_from(r.as_str())
                    .map_err(|_| JsValue::from_str(&format!("unknown role: {r}")))?,
            )
        } else {
            None
        };
        let options = ParseOptions {
            strict: strict.unwrap_or(true),
        };
        let messages: Vec<Message> = self
            .inner
            .parse_messages_from_completion_tokens_with_options(tokens, role_parsed, options)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        serde_json::to_string(&messages)
            .map_err(|e| JsValue::from_str(&format!("failed to serialise messages to JSON: {e}")))
    }

    #[wasm_bindgen(js_name = decodeUtf8)]
    pub fn decode_utf8(&self, tokens: Vec<u32>) -> Result<String, JsValue> {
        self.inner
            .tokenizer()
            .decode_utf8(tokens)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = decodeBytes)]
    pub fn decode_bytes(&self, tokens: Vec<u32>) -> Result<Vec<u8>, JsValue> {
        self.inner
            .tokenizer()
            .decode_bytes(tokens)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn encode(&self, text: &str, allowed_special: JsValue) -> Result<Vec<u32>, JsValue> {
        let allowed_vec: Vec<String> =
            if allowed_special.is_undefined() || allowed_special.is_null() {
                Vec::new()
            } else {
                serde_wasm_bindgen::from_value(allowed_special)
                    .map_err(|e| JsValue::from_str(&format!("invalid allowed_special: {e}")))?
            };
        let allowed_set: std::collections::HashSet<&str> =
            allowed_vec.iter().map(|s| s.as_str()).collect();
        Ok(self.inner.tokenizer().encode(text, &allowed_set).0)
    }

    #[wasm_bindgen(js_name = specialTokens)]
    pub fn special_tokens(&self) -> Vec<String> {
        self.inner
            .tokenizer()
            .special_tokens()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    #[wasm_bindgen(js_name = isSpecialToken)]
    pub fn is_special_token(&self, token: u32) -> bool {
        self.inner.tokenizer().is_special_token(token)
    }

    #[wasm_bindgen(js_name = stopTokens)]
    pub fn stop_tokens(&self) -> Result<Vec<u32>, JsValue> {
        self.inner
            .stop_tokens()
            .map(|set| set.into_iter().collect())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = stopTokensForAssistantActions)]
    pub fn stop_tokens_for_assistant_actions(&self) -> Result<Vec<u32>, JsValue> {
        self.inner
            .stop_tokens_for_assistant_actions()
            .map(|set| set.into_iter().collect())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[wasm_bindgen]
pub struct JsStreamableParser {
    inner: StreamableParser,
}

#[wasm_bindgen]
impl JsStreamableParser {
    #[wasm_bindgen(constructor)]
    pub fn new(
        encoding: &JsHarmonyEncoding,
        role: &str,
        strict: Option<bool>,
    ) -> Result<JsStreamableParser, JsValue> {
        let parsed_role = Role::try_from(role)
            .map_err(|_| JsValue::from_str(&format!("unknown role: {role}")))?;
        let options = ParseOptions {
            strict: strict.unwrap_or(true),
        };
        let inner =
            StreamableParser::new_with_options(encoding.inner.clone(), Some(parsed_role), options)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn process(&mut self, token: u32) -> Result<(), JsValue> {
        self.inner
            .process(token)
            .map(|_| ())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(getter, js_name = currentContent)]
    pub fn current_content(&self) -> Result<String, JsValue> {
        self.inner
            .current_content()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(getter, js_name = currentRole)]
    pub fn current_role(&self) -> String {
        self.inner
            .current_role()
            .map(|r| r.as_str().to_string())
            .unwrap_or_default()
    }

    #[wasm_bindgen(getter, js_name = currentContentType)]
    pub fn current_content_type(&self) -> String {
        self.inner.current_content_type().unwrap_or_default()
    }

    #[wasm_bindgen(getter, js_name = lastContentDelta)]
    pub fn last_content_delta(&self) -> Result<String, JsValue> {
        match self.inner.last_content_delta() {
            Ok(Some(s)) => Ok(s),
            Ok(None) => Ok(String::new()),
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn messages(&self) -> Result<String, JsValue> {
        serde_json::to_string(self.inner.messages())
            .map_err(|e| JsValue::from_str(&format!("failed to serialise messages to JSON: {e}")))
    }

    #[wasm_bindgen(getter)]
    pub fn tokens(&self) -> Vec<u32> {
        self.inner.tokens().to_vec()
    }

    #[wasm_bindgen(getter)]
    pub fn state(&self) -> Result<String, JsValue> {
        self.inner
            .state_json()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(getter, js_name = currentRecipient)]
    pub fn current_recipient(&self) -> String {
        self.inner.current_recipient().unwrap_or_default()
    }

    #[wasm_bindgen(getter, js_name = currentChannel)]
    pub fn current_channel(&self) -> String {
        self.inner.current_channel().unwrap_or_default()
    }
}

#[wasm_bindgen]
pub enum StreamState {
    ExpectStart,
    Header,
    Content,
}

#[wasm_bindgen]
pub async fn load_harmony_encoding(
    name: &str,
    base_url: Option<String>,
) -> Result<JsHarmonyEncoding, JsValue> {
    if let Some(base) = base_url {
        crate::tiktoken_ext::set_tiktoken_base_url(base);
    }
    let parsed: HarmonyEncodingName = name
        .parse::<HarmonyEncodingName>()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let encoding =
        inner_load_harmony_encoding(parsed).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(JsHarmonyEncoding { inner: encoding })
}

#[wasm_bindgen]
pub fn get_tool_namespace_config(tool: &str) -> Result<JsValue, JsValue> {
    let cfg = match tool {
        "browser" => ToolNamespaceConfig::browser(),
        "python" => ToolNamespaceConfig::python(),
        _ => {
            return Err(JsValue::from_str(&format!(
                "Unknown tool namespace: {tool}"
            )))
        }
    };
    serde_wasm_bindgen::to_value(&cfg).map_err(|e| JsValue::from_str(&e.to_string()))
}

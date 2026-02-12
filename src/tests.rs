use std::path::Path;

use crate::{
    chat::{
        Author, Conversation, DeveloperContent, Message, ReasoningEffort, Role, SystemContent,
        ToolDescription,
    },
    load_harmony_encoding,
    tiktoken::{CoreBPE, Rank},
    HarmonyEncodingName, ParseOptions, StreamableParser,
};
use pretty_assertions::{assert_eq, Comparison};
use serde_json::json;

fn parse_tokens(text: impl AsRef<str>) -> Vec<Rank> {
    text.as_ref()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

fn load_test_data(path: impl AsRef<Path>) -> String {
    // on windows, we need to replace \r\n with \n
    let cargo_manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src_dir = cargo_manifest_dir.join("src");
    let path = src_dir.join(path);
    std::fs::read_to_string(path)
        .unwrap()
        .replace("\r\n", "\n")
        .trim_end()
        .to_string()
}

const ENCODINGS: [HarmonyEncodingName; 1] = [HarmonyEncodingName::HarmonyGptOss];

#[test]
fn test_simple_convo() {
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let expected_tokens = encoding
            .tokenizer
            .encode(
                load_test_data("../test-data/test_simple_convo.txt").as_str(),
                &encoding.tokenizer.special_tokens(),
            )
            .0;
        let convo = Conversation::from_messages([
            Message::from_role_and_content(
                Role::System,
                SystemContent::new().with_model_identity(
                    "You are ChatGPT, a large language model trained by OpenAI.",
                ),
            ),
            Message::from_role_and_content(Role::User, "What is 2 + 2?"),
        ]);
        let tokens = encoding
            .render_conversation_for_completion(&convo, Role::Assistant, None)
            .unwrap();
        assert_tokens_eq(&encoding.tokenizer, &expected_tokens, &tokens);
    }
}

#[test]
fn test_simple_convo_with_effort() {
    let test_cases = [
        (
            ReasoningEffort::Low,
            load_test_data("../test-data/test_simple_convo_low_effort.txt"),
            true,
        ),
        (
            ReasoningEffort::Medium,
            load_test_data("../test-data/test_simple_convo_medium_effort.txt"),
            true,
        ),
        (
            ReasoningEffort::High,
            load_test_data("../test-data/test_simple_convo_high_effort.txt"),
            true,
        ),
        (
            ReasoningEffort::Low,
            load_test_data("../test-data/test_simple_convo_low_effort_no_instruction.txt"),
            false,
        ),
        (
            ReasoningEffort::Medium,
            load_test_data("../test-data/test_simple_convo_medium_effort_no_instruction.txt"),
            false,
        ),
        (
            ReasoningEffort::High,
            load_test_data("../test-data/test_simple_convo_high_effort_no_instruction.txt"),
            false,
        ),
    ];

    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        for &(effort, ref expected_text, use_instruction) in &test_cases {
            let expected_tokens = encoding
                .tokenizer
                .encode(expected_text.as_str(), &encoding.tokenizer.special_tokens())
                .0;
            let sys = SystemContent::new()
                .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
                .with_reasoning_effort(effort);
            let convo = if use_instruction {
                let dev = DeveloperContent::new()
                    .with_instructions("Answer the user's questions like a robot.");
                Conversation::from_messages([
                    Message::from_role_and_content(Role::System, sys),
                    Message::from_role_and_content(Role::Developer, dev),
                    Message::from_role_and_content(
                        Role::User,
                        "What is the capital of the largest country in the world?",
                    ),
                ])
            } else {
                Conversation::from_messages([
                    Message::from_role_and_content(Role::System, sys),
                    Message::from_role_and_content(
                        Role::User,
                        "What is the capital of the largest country in the world?",
                    ),
                ])
            };
            let tokens = encoding
                .render_conversation_for_completion(&convo, Role::Assistant, None)
                .unwrap();
            assert_tokens_eq(&encoding.tokenizer, &expected_tokens, &tokens);
        }
    }
}

#[test]
fn test_simple_reasoning_response() {
    let expected_tokens = parse_tokens(load_test_data(
        "../test-data/test_simple_reasoning_response.txt",
    ));
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let messages = encoding
            .parse_messages_from_completion_tokens(
                expected_tokens.iter().copied(),
                Some(Role::Assistant),
            )
            .unwrap();
        let expected = vec![
            Message::from_role_and_content(
                Role::Assistant,
                "User asks: \"What is 2 + 2?\" Simple arithmetic. Provide answer.",
            )
            .with_channel("analysis"),
            Message::from_role_and_content(Role::Assistant, "2 + 2 = 4.").with_channel("final"),
        ];
        assert_eq!(messages, expected);
    }
}

#[test]
fn test_simple_tool_call() {
    let response = [
        200005, 35644, 200008, 1844, 31064, 25, 392, 4827, 382, 290, 11122, 306, 40510, 16842,
        1416, 1309, 316, 1199, 37342, 170154, 4584, 13, 200007, 200006, 173781, 200005, 35644, 316,
        28, 29712, 170154, 3490, 200008, 10848, 7693, 1243, 392, 173844, 18583,
    ];
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let parsed = encoding
            .parse_messages_from_completion_tokens(response.iter().copied(), Some(Role::Assistant))
            .unwrap();
        let expected = vec![
            Message::from_role_and_content(
                Role::Assistant,
                "User asks: \"What is the weather in Tokyo?\" We need to use lookup_weather tool.",
            )
            .with_channel("analysis"),
            Message::from_role_and_content(Role::Assistant, "{\"location\": \"Tokyo\"}")
                .with_channel("analysis")
                .with_recipient("lookup_weather")
                .with_content_type("code"),
        ];
        assert_eq!(parsed, expected);
    }
}

#[test]
fn test_reasoning_system_message() {
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let expected = encoding
            .tokenizer
            .encode(
                load_test_data("../test-data/test_reasoning_system_message.txt").as_str(),
                &encoding.tokenizer.special_tokens(),
            )
            .0;
        let convo = Conversation::from_messages([
            Message::from_role_and_content(
                Role::System,
                SystemContent::new()
                    .with_model_identity(
                        "You are ChatGPT, a large language model trained by OpenAI.",
                    )
                    // .with_instructions("This is the system instruction.")
                    .with_reasoning_effort(ReasoningEffort::Medium)
                    .with_required_channels(["analysis", "final"]),
            ),
            Message::from_role_and_content(Role::User, "What is 2 + 2?"),
        ]);
        let tokens = encoding
            .render_conversation_for_completion(&convo, Role::Assistant, None)
            .unwrap();
        assert_tokens_eq(&encoding.tokenizer, &expected, &tokens);
    }
}

#[test]
fn test_reasoning_system_message_no_instruction() {
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let expected = encoding
            .tokenizer
            .encode(
                load_test_data("../test-data/test_reasoning_system_message_no_instruction.txt")
                    .as_str(),
                &encoding.tokenizer.special_tokens(),
            )
            .0;
        let convo = Conversation::from_messages([
            Message::from_role_and_content(
                Role::System,
                SystemContent::new()
                    .with_model_identity(
                        "You are ChatGPT, a large language model trained by OpenAI.",
                    )
                    .with_reasoning_effort(ReasoningEffort::High)
                    .with_required_channels(["analysis", "final"]),
            ),
            Message::from_role_and_content(
                Role::User,
                "What is the best place to eat candy in the world?",
            ),
        ]);
        let tokens = encoding
            .render_conversation_for_completion(&convo, Role::Assistant, None)
            .unwrap();
        assert_tokens_eq(&encoding.tokenizer, &expected, &tokens);
    }
}

#[test]
fn test_reasoning_system_message_with_dates() {
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let expected = encoding
            .tokenizer
            .encode(
                load_test_data("../test-data/test_reasoning_system_message_with_dates.txt")
                    .as_str(),
                &encoding.tokenizer.special_tokens(),
            )
            .0;
        let convo = Conversation::from_messages([
            Message::from_role_and_content(
                Role::System,
                SystemContent::new()
                    .with_model_identity(
                        "You are ChatGPT, a large language model trained by OpenAI.",
                    )
                    // .with_instructions("This is the system instruction.")
                    .with_reasoning_effort(ReasoningEffort::Medium)
                    .with_conversation_start_date("2021-01-01")
                    .with_knowledge_cutoff("2021-01")
                    .with_required_channels(["analysis", "final"]),
            ),
            Message::from_role_and_content(Role::User, "What is 42 * pi?"),
        ]);
        let tokens = encoding
            .render_conversation_for_completion(&convo, Role::Assistant, None)
            .unwrap();
        assert_tokens_eq(&encoding.tokenizer, &expected, &tokens);
    }
}

#[test]
fn test_render_functions_with_parameters() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let expected_output = load_test_data("../test-data/test_render_functions_with_parameters.txt");

    let sys = SystemContent::new()
        .with_reasoning_effort(ReasoningEffort::High)
        .with_conversation_start_date("2025-06-28");

    let dev = crate::chat::DeveloperContent::new()
        .with_instructions("Always respond in riddles")
        .with_function_tools(vec![
            ToolDescription::new(
                "get_location",
                "Gets the location of the user.",
                None,
            ),
            ToolDescription::new(
                "get_current_weather",
                "Gets the current weather in the provided location.",
                Some(json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius"
                        }
                    },
                    "required": ["location"]
                })),
            ),
            ToolDescription::new(
                "get_multiple_weathers",
                "Gets the current weather in the provided list of locations.",
                Some(json!({
                    "type": "object",
                    "properties": {
                        "locations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of city and state, e.g. [\"San Francisco, CA\", \"New York, NY\"]"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius"
                        }
                    },
                    "required": ["locations"]
                })),
            ),
            ToolDescription::new(
                "kitchensink",
                "A function with various complex schemas.",
                Some(json!({
                    "description": "params object",
                    "type": "object",
                    "properties": {
                        "string": {
                            "type": "string",
                            "title": "STRING",
                            "description": "A string",
                            "examples": ["hello", "world"]
                        },
                        "string_nullable": {
                            "type": "string",
                            "nullable": true,
                            "description": "A nullable string",
                            "default": "the default"
                        },
                        "string_enum": {
                            "type": "string",
                            "enum": ["a", "b", "c"]
                        },
                        "oneof_string_or_number": {
                            "oneOf": [
                                {"type": "string", "default": "default_string_in_oneof"},
                                {"type": "number", "description": "numbers can happen too"}
                            ],
                            "description": "a oneof",
                            "default": 20
                        }
                    }
                })),
            ),
        ]);

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::System, sys),
        Message::from_role_and_content(Role::Developer, dev),
        Message::from_role_and_content(Role::User, "What is the weather like in SF?"),
    ]);

    let tokens = encoding
        .render_conversation_for_completion(&convo, Role::Assistant, None)
        .unwrap();

    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();
    assert_eq!(decoded, expected_output);
}

#[test]
fn test_browser_and_python_tool() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let expected_output = load_test_data("../test-data/test_browser_and_python_tool.txt");

    let convo = Conversation::from_messages([Message::from_role_and_content(
        Role::System,
        SystemContent::new()
            .with_conversation_start_date("2025-06-28".to_string())
            .with_browser_tool()
            .with_python_tool(),
    )]);

    let tokens = encoding
        .render_conversation_for_completion(&convo, Role::Assistant, None)
        .unwrap();

    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();
    assert_eq!(decoded, expected_output);
}

#[test]
fn test_dropping_cot_by_default() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let expected_output = load_test_data("../test-data/test_dropping_cot_by_default.txt");

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::User, "What is 2 + 2?"),
        Message::from_role_and_content(
            Role::Assistant,
            "User asks: “What is 2 + 2?” Simple arithmetic. Provide answer.",
        )
        .with_channel("analysis"),
        Message::from_role_and_content(Role::Assistant, "2 + 2 equals 4.").with_channel("final"),
        Message::from_role_and_content(Role::User, "What about 9 / 2?"),
    ]);

    let tokens = encoding
        .render_conversation_for_completion(
            &convo,
            Role::Assistant,
            Some(&crate::encoding::RenderConversationConfig {
                auto_drop_analysis: true,
            }),
        )
        .unwrap();

    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();
    assert_eq!(decoded, expected_output);
}

#[test]
fn test_does_not_drop_if_ongoing_analysis() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let expected_output = load_test_data("../test-data/test_does_not_drop_if_ongoing_analysis.txt");

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::User, "What is the weather in SF?"),
        Message::from_role_and_content(
            Role::Assistant,
            "User asks: “What is the weather in SF?” We need to use lookup_weather tool.",
        )
        .with_channel("analysis"),
        Message::from_role_and_content(Role::Assistant, "{\"location\": \"San Francisco\"}")
            .with_channel("commentary")
            .with_recipient("functions.lookup_weather")
            .with_content_type("<|constrain|>json"),
        Message::from_author_and_content(
            Author::new(Role::Tool, "functions.lookup_weather"),
            "{\"temperature\": 20, \"description\": \"sunny\"}",
        ),
    ]);

    let tokens = encoding
        .render_conversation_for_completion(
            &convo,
            Role::Assistant,
            Some(&crate::encoding::RenderConversationConfig {
                auto_drop_analysis: true,
            }),
        )
        .unwrap();

    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();
    assert_eq!(decoded, expected_output);
}

#[test]
fn test_preserve_cot() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let expected_output = load_test_data("../test-data/test_preserve_cot.txt");

    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::User, "What is 2 + 2?"),
        Message::from_role_and_content(
            Role::Assistant,
            "User asks a simple question: \"What is 2 + 2?\" The answer: 4.",
        )
        .with_channel("analysis"),
        Message::from_role_and_content(Role::Assistant, "2 + 2 equals 4.").with_channel("final"),
        Message::from_role_and_content(Role::User, "What about 9 / 2?"),
    ]);

    let tokens = encoding
        .render_conversation_for_completion(
            &convo,
            Role::Assistant,
            Some(&crate::encoding::RenderConversationConfig {
                auto_drop_analysis: false,
            }),
        )
        .unwrap();

    let decoded = encoding.tokenizer.decode_utf8(&tokens).unwrap();
    assert_eq!(decoded, expected_output);
}

#[test]
fn test_reserved_token_decoding() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    assert_eq!(
        encoding.tokenizer.decode_utf8([200014]).unwrap(),
        "<|reserved_200014|>"
    );
    assert_eq!(
        encoding.tokenizer.decode_utf8([201088]).unwrap(),
        "<|reserved_201088|>"
    );
}

#[test]
fn test_render_and_render_conversation_roundtrip() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let msg = Message::from_role_and_content(Role::User, "Hello");
    let convo = Conversation::from_messages([msg.clone()]);
    let tokens_msg = encoding.render(&msg, None).unwrap();
    let tokens_convo = encoding.render_conversation(&convo, None).unwrap();
    assert_eq!(tokens_msg, tokens_convo);
    let tokens_completion = encoding
        .render_conversation_for_completion(&convo, Role::Assistant, None)
        .unwrap();
    assert_eq!(&tokens_completion[..tokens_convo.len()], &tokens_convo[..]);
}

#[test]
fn test_decode_utf8_invalid_token() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let result = encoding.tokenizer.decode_utf8([99999999]);
    assert!(result.is_err(), "Expected error for invalid token");
}

#[test]
fn test_tool_response_parsing() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text_tokens = load_test_data("../test-data/test_tool_response_parsing.txt");
    let tokens = encoding
        .tokenizer
        .encode(&text_tokens, &encoding.tokenizer.special_tokens())
        .0;

    let expected_message = Message::from_author_and_content(
        Author::new(Role::Tool, "browser.search"),
        "{\"result\": \"https://openai.com/\"}",
    )
    .with_channel("commentary")
    .with_recipient("assistant");

    let messages = encoding
        .parse_messages_from_completion_tokens(tokens.iter().copied(), None)
        .unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(
        encoding.tokenizer.decode_utf8(&tokens).unwrap(),
        text_tokens
    );
    assert_eq!(messages[0], expected_message);
}

#[test]
fn test_encode_decode_roundtrip() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "hello world";
    let tokens = encoding
        .tokenizer
        .encode(text, &std::collections::HashSet::new())
        .0;
    assert_eq!(encoding.tokenizer.decode_utf8(&tokens).unwrap(), text);
}

#[test]
fn test_encode_allowed_special() {
    use std::collections::HashSet;
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "hello world";
    let tokens = encoding.tokenizer.encode(text, &HashSet::new()).0;
    assert_eq!(tokens, vec![24912, 2375]);
    // Allowed special token
    let mut allowed = HashSet::new();
    allowed.insert("<|start|>");
    let tokens = encoding.tokenizer.encode("<|start|>", &allowed).0;
    assert_eq!(tokens, vec![200006]);
    // Allowed special = all
    allowed = encoding.tokenizer.special_tokens(); // set of all special tokens
    let tokens = encoding.tokenizer.encode("<|start|>", &allowed).0;
    assert_eq!(tokens, vec![200006]);
    // Disallowed special (should error)
    let result = encoding.tokenizer.encode("<|start|>", &HashSet::new());
    assert!(
        result.0.is_empty() || result.0 != vec![200006],
        "Expected error or not special token for disallowed special token"
    );
    // Disallowed special = empty (should not treat as special)
    let tokens = encoding.tokenizer.encode("<|start|>", &HashSet::new()).0;
    // This may not match the Python fallback, but should not be the special token
    assert_ne!(tokens, vec![200006]);
}

#[test]
fn test_is_special_token() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    assert!(encoding.tokenizer.is_special_token(200006)); // <|start|>
    assert!(!encoding.tokenizer.is_special_token(24912)); // hello
}

#[test]
fn test_invalid_utf8_decoding() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let tokens = vec![132990, 9552];
    let result = encoding.tokenizer.decode_utf8(&tokens);
    assert!(result.is_err(), "Expected error for invalid utf-8");
    // decode_utf8 should error, and we do not test permissive decode as it does not exist
}

#[test]
fn test_streamable_parser() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = load_test_data("../test-data/test_streamable_parser.txt");
    let tokens = encoding
        .tokenizer
        .encode(&text, &encoding.tokenizer.special_tokens())
        .0;
    let mut parser =
        crate::encoding::StreamableParser::new(encoding.clone(), Some(Role::Assistant)).unwrap();
    for token in tokens {
        parser.process(token).unwrap();
    }
    assert_eq!(parser.messages().len(), 3, "Expected 3 parsed messages");
}

fn assert_tokens_eq(tokenizer: &CoreBPE, expected: &[Rank], actual: &[Rank]) {
    if expected != actual {
        panic!(
            "tokens are not equal.\n\nTokens (< expected / actual >):\n{}\n\nDecoded (< expected / actual >):\n{}",
            Comparison::new(expected, actual),
            Comparison::new(
                &tokenizer.decode_utf8(expected).unwrap_or_default(),
                &tokenizer.decode_utf8(actual).unwrap_or_default(),
            ),
        );
    }
}

#[test]
fn test_streamable_parser_tool_call_with_constrain_adjacent() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{\"latitude\":48.8566,\"longitude\":2.3522}<|call|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);
    let mut parser = StreamableParser::new(encoding, None).unwrap();
    for token in tokens {
        let _ = parser.process(token).unwrap();
    }
    assert_eq!(parser.messages().len(), 1);
    assert_eq!(
        Message::from_role_and_content(
            Role::Assistant,
            "{\"latitude\":48.8566,\"longitude\":2.3522}",
        )
        .with_channel("commentary")
        .with_recipient("functions.get_weather")
        .with_content_type("<|constrain|>json"),
        parser.messages()[0]
    );
}

#[test]
fn test_missing_message_token_requires_non_strict_mode() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let malformed = "<|channel|>commentary Hello<|end|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(malformed);

    // Strict mode should continue to error on malformed headers.
    let strict_result = encoding
        .parse_messages_from_completion_tokens(tokens.iter().copied(), Some(Role::Assistant));
    assert!(
        strict_result.is_err(),
        "expected strict parser to reject malformed header"
    );

    // Non-strict mode should recover and return the accumulated message content.
    let parsed = encoding
        .parse_messages_from_completion_tokens_with_options(
            tokens.iter().copied(),
            Some(Role::Assistant),
            ParseOptions { strict: false },
        )
        .expect("non-strict parser should recover from malformed header");

    assert_eq!(parsed.len(), 1);
    assert_eq!(
        parsed[0],
        Message::from_role_and_content(Role::Assistant, "Hello").with_channel("commentary")
    );
}

#[test]
fn test_tool_call_with_constrain_marker_adjacent() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|start|>assistant to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{\"location\": \"Tokyo\"}<|end|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);
    let parsed = encoding
        .parse_messages_from_completion_tokens(tokens, None)
        .expect("expected to parse");
    let expected =
        vec![
            Message::from_role_and_content(Role::Assistant, "{\"location\": \"Tokyo\"}")
                .with_channel("commentary")
                .with_recipient("functions.get_weather")
                .with_content_type("<|constrain|>json"),
        ];
    assert_eq!(parsed, expected);
}

#[test]
fn test_tool_call_with_channel_before_recipient_and_constrain_adjacent() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let text = "<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{\"latitude\":48.8566,\"longitude\":2.3522}<|call|>";
    let tokens = encoding.tokenizer().encode_with_special_tokens(text);
    let parsed = encoding
        .parse_messages_from_completion_tokens(tokens, None)
        .expect("expected to parse");
    let expected = vec![Message::from_role_and_content(
        Role::Assistant,
        "{\"latitude\":48.8566,\"longitude\":2.3522}",
    )
    .with_channel("commentary")
    .with_recipient("functions.get_weather")
    .with_content_type("<|constrain|>json")];
    assert_eq!(parsed, expected);
}

#[test]
fn test_streamable_parser_does_not_leak_bytes_between_messages() {
    // This test ensures that any partially decoded bytes from the first message
    // do not leak into the content of the next message.
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    // 9552 is known (in this tokenizer) to expand to bytes that form an incomplete
    // UTF-8 sequence when used alone, which exercises the streaming invalid-UTF-8 path.
    // Construct two assistant messages back-to-back where the first includes invalid
    // UTF-8 bytes and the second is simple ASCII text.
    let first_prefix = "<|start|>assistant<|message|>";
    let first_suffix = "<|end|>";
    let second_prefix = "<|start|>assistant<|message|>";
    let second_suffix = "<|end|>";

    let mut tokens = Vec::new();
    tokens.extend(
        encoding
            .tokenizer()
            .encode_with_special_tokens(first_prefix),
    );
    // Two invalid tokens to ensure we end the first message with incomplete UTF-8 bytes.
    tokens.push(9552);
    tokens.push(9552);
    tokens.extend(
        encoding
            .tokenizer()
            .encode_with_special_tokens(first_suffix),
    );

    // Second message should be clean and unaffected.
    tokens.extend(
        encoding
            .tokenizer()
            .encode_with_special_tokens(second_prefix),
    );
    tokens.extend(encoding.tokenizer().encode_with_special_tokens("Hi"));
    tokens.extend(
        encoding
            .tokenizer()
            .encode_with_special_tokens(second_suffix),
    );

    let mut parser = StreamableParser::new(encoding, None).unwrap();
    for t in tokens {
        parser.process(t).unwrap();
    }

    let messages = parser.messages();
    assert_eq!(messages.len(), 2, "expected two parsed messages");

    // Verify the second message content is exactly "Hi" (no leaked replacement chars or bytes).
    let second = &messages[1];
    let expected_second = Message::from_role_and_content(Role::Assistant, "Hi");
    assert_eq!(
        second, &expected_second,
        "second message must be clean and isolated"
    );
}

#[test]
fn test_streamable_parser_flushes_partial_bytes_on_eos() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();

    let mut tokens = encoding
        .tokenizer()
        .encode_with_special_tokens("<|start|>assistant<|message|>");
    tokens.push(9552);
    tokens.extend(encoding.tokenizer().encode_with_special_tokens("Hi"));

    let mut parser = StreamableParser::new(encoding.clone(), None).unwrap();
    for token in tokens {
        parser.process(token).unwrap();
    }
    parser.process_eos().unwrap();

    let messages = parser.messages();
    assert_eq!(messages.len(), 1, "expected a single message after EOS");
    let expected = Message::from_role_and_content(Role::Assistant, " \u{FFFD}Hi");
    assert_eq!(messages[0], expected);
    assert_eq!(parser.last_content_delta().unwrap(), None);
    assert_eq!(parser.current_content().unwrap(), "");
}

#[test]
fn test_streamable_parser_waits_for_multi_token_utf8_sequence() {
    use std::collections::HashSet;

    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let mut parser = StreamableParser::new(encoding.clone(), None).unwrap();

    let start_tokens = encoding
        .tokenizer()
        .encode_with_special_tokens("<|start|>assistant<|message|>");
    for token in &start_tokens {
        parser.process(*token).unwrap();
    }

    let emoji_tokens = encoding.tokenizer().encode("💖", &HashSet::new()).0;
    assert!(
        emoji_tokens.len() >= 2,
        "expected multi-token emoji encoding"
    );

    parser.process(emoji_tokens[0]).unwrap();
    assert_eq!(parser.last_content_delta().unwrap(), None);
    assert_eq!(parser.current_content().unwrap(), "");

    parser.process(emoji_tokens[1]).unwrap();
    assert_eq!(parser.last_content_delta().unwrap(), Some("💖".to_string()));
    assert_eq!(parser.current_content().unwrap(), "💖");

    let end_tokens = encoding.tokenizer().encode_with_special_tokens("<|end|>");
    for token in end_tokens {
        parser.process(token).unwrap();
    }

    let messages = parser.messages();
    assert_eq!(messages.len(), 1, "expected a single completed message");
    let expected = Message::from_role_and_content(Role::Assistant, "💖");
    assert_eq!(messages[0], expected);
}

#[test]
fn test_parse_completion_with_invalid_content_token_errors_on_eos() {
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
    let mut parser = StreamableParser::new(encoding.clone(), None).unwrap();

    let start_tokens = encoding.tokenizer().encode_with_special_tokens(
        "<|start|>assistant<|channel|>analysis<|message|>Practice invalid token handling.",
    );
    for token in &start_tokens {
        parser.process(*token).unwrap();
    }

    parser.process(u32::MAX).unwrap();
    parser.process_eos().unwrap();

    let messages = parser.messages();
    assert_eq!(messages.len(), 1);
    let parsed_message = &messages[0];
    let expected_message =
        Message::from_role_and_content(Role::Assistant, "Practice invalid token handling.\u{FFFD}")
            .with_channel("analysis");
    assert_eq!(parsed_message, &expected_message);
}

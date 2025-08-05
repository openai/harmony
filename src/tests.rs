use crate::{
    chat::{
        Author, Conversation, DeveloperContent, Message, ReasoningEffort, Role, SystemContent,
        ToolDescription,
    },
    load_harmony_encoding,
    tiktoken::{CoreBPE, Rank},
    HarmonyEncodingName, StreamableParser,
};
use pretty_assertions::{assert_eq, Comparison};
use serde_json::json;

fn parse_tokens(text: &str) -> Vec<Rank> {
    text.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

const ENCODINGS: [HarmonyEncodingName; 1] = [HarmonyEncodingName::HarmonyGptOss];

#[test]
fn test_simple_convo() {
    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        let expected_tokens = encoding
            .tokenizer
            .encode(
                include_str!("../test-data/test_simple_convo.txt").trim_end(),
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
            include_str!("../test-data/test_simple_convo_low_effort.txt"),
            true,
        ),
        (
            ReasoningEffort::Medium,
            include_str!("../test-data/test_simple_convo_medium_effort.txt"),
            true,
        ),
        (
            ReasoningEffort::High,
            include_str!("../test-data/test_simple_convo_high_effort.txt"),
            true,
        ),
        (
            ReasoningEffort::Low,
            include_str!("../test-data/test_simple_convo_low_effort_no_instruction.txt"),
            false,
        ),
        (
            ReasoningEffort::Medium,
            include_str!("../test-data/test_simple_convo_medium_effort_no_instruction.txt"),
            false,
        ),
        (
            ReasoningEffort::High,
            include_str!("../test-data/test_simple_convo_high_effort_no_instruction.txt"),
            false,
        ),
    ];

    for encoding_name in ENCODINGS {
        let encoding = load_harmony_encoding(encoding_name).unwrap();
        for (effort, expected_text, use_instruction) in test_cases {
            let expected_tokens = encoding
                .tokenizer
                .encode(
                    expected_text.trim_end(),
                    &encoding.tokenizer.special_tokens(),
                )
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
    let expected_tokens = parse_tokens(include_str!(
        "../test-data/test_simple_reasoning_response.txt"
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
                include_str!("../test-data/test_reasoning_system_message.txt").trim_end(),
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
                include_str!("../test-data/test_reasoning_system_message_no_instruction.txt")
                    .trim_end(),
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
                include_str!("../test-data/test_reasoning_system_message_with_dates.txt")
                    .trim_end(),
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
    let expected_output =
        include_str!("../test-data/test_render_functions_with_parameters.txt").trim_end();

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
    let expected_output = include_str!("../test-data/test_browser_and_python_tool.txt").trim_end();

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
    let expected_output = include_str!("../test-data/test_dropping_cot_by_default.txt").trim_end();

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
    let expected_output =
        include_str!("../test-data/test_does_not_drop_if_ongoing_analysis.txt").trim_end();

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
    let expected_output = include_str!("../test-data/test_preserve_cot.txt").trim_end();

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
    let tokens_msg = encoding.render(&msg).unwrap();
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
    let text_tokens = include_str!("../test-data/test_tool_response_parsing.txt").trim_end();
    let tokens = encoding
        .tokenizer
        .encode(text_tokens, &encoding.tokenizer.special_tokens())
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
    let text = include_str!("../test-data/test_streamable_parser.txt").trim_end();
    let tokens = encoding
        .tokenizer
        .encode(text, &encoding.tokenizer.special_tokens())
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

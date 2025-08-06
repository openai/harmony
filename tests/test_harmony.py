"""Port of the original Rust test-suite to Python.

The tests mirror the scenarios from ``src/tests.rs`` and exercise the public
Python API.  They ensure that the bindings give byte-for-byte identical output
to the canonical Rust implementation.
"""

# ruff: noqa: E402   # postpone imports until path manipulation is done

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Ensure that the project root is on *sys.path* so that ``import harmony``
# picks up the local Python package during test execution (pytest changes the
# working directory which would otherwise hide the module).
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest  # noqa: E402
from openai_harmony import (  # noqa: E402
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    HarmonyError,
    Message,
    ReasoningEffort,
    RenderConversationConfig,
    Role,
    StreamableParser,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    load_harmony_encoding_from_file,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_tokens_eq(encoding, expected: List[int], actual: List[int]):  # type: ignore[arg-type]
    """Mimic the pretty-assertions output from the Rust test-suite."""

    if expected != actual:
        exp_decoded = encoding.decode_utf8(expected)
        act_decoded = encoding.decode_utf8(actual)
        raise AssertionError(
            "tokens are not equal.\n\n"
            "Tokens (< expected / actual >):\n"
            f"{expected}\n{actual}\n\n"
            "Decoded (< expected / actual >):\n"
            f"{exp_decoded!r}\n{act_decoded!r}"
        )


def read_expected_tokens(file_path: Path) -> List[int]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


# ---------------------------------------------------------------------------
# Tests (1-1 port from the Rust side)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "encoding_name",
    [
        HarmonyEncodingName.HARMONY_GPT_OSS,
    ],
)
def test_simple_convo(encoding_name):
    encoding = load_harmony_encoding(encoding_name)

    expected_text = (
        (ROOT_DIR / "test-data" / "test_simple_convo.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )
    expected_tokens = encoding.encode(expected_text, allowed_special="all")

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_model_identity(
                    "You are ChatGPT, a large language model trained by OpenAI."
                ),
            ),
            Message.from_role_and_content(Role.USER, "What is 2 + 2?"),
        ]
    )

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    _assert_tokens_eq(encoding, expected_tokens, tokens)


@pytest.mark.parametrize(
    "encoding_name",
    [
        HarmonyEncodingName.HARMONY_GPT_OSS,
    ],
)
def test_simple_convo_with_effort(encoding_name):
    encoding = load_harmony_encoding(encoding_name)
    test_cases = [
        (
            ReasoningEffort.LOW,
            ROOT_DIR / "test-data" / "test_simple_convo_low_effort.txt",
            True,
        ),
        (
            ReasoningEffort.MEDIUM,
            ROOT_DIR / "test-data" / "test_simple_convo_medium_effort.txt",
            True,
        ),
        (
            ReasoningEffort.HIGH,
            ROOT_DIR / "test-data" / "test_simple_convo_high_effort.txt",
            True,
        ),
        (
            ReasoningEffort.LOW,
            ROOT_DIR / "test-data" / "test_simple_convo_low_effort_no_instruction.txt",
            False,
        ),
        (
            ReasoningEffort.MEDIUM,
            ROOT_DIR
            / "test-data"
            / "test_simple_convo_medium_effort_no_instruction.txt",
            False,
        ),
        (
            ReasoningEffort.HIGH,
            ROOT_DIR / "test-data" / "test_simple_convo_high_effort_no_instruction.txt",
            False,
        ),
    ]

    for effort, tokens_path, use_instruction in test_cases:
        expected_text = tokens_path.read_text(encoding="utf-8").rstrip()
        expected_tokens = encoding.encode(expected_text, allowed_special="all")
        sys = (
            SystemContent.new()
            .with_model_identity(
                "You are ChatGPT, a large language model trained by OpenAI."
            )
            .with_reasoning_effort(effort)
        )
        messages = [Message.from_role_and_content(Role.SYSTEM, sys)]
        if use_instruction:
            dev = DeveloperContent.new().with_instructions(
                "Answer the user's questions like a robot."
            )
            messages.append(Message.from_role_and_content(Role.DEVELOPER, dev))
        messages.append(
            Message.from_role_and_content(
                Role.USER,
                "What is the capital of the largest country in the world?",
            )
        )
        convo = Conversation.from_messages(messages)
        tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        _assert_tokens_eq(encoding, expected_tokens, tokens)


@pytest.mark.parametrize(
    "encoding_name",
    [
        HarmonyEncodingName.HARMONY_GPT_OSS,
    ],
)
def test_simple_reasoning_response(encoding_name):
    encoding = load_harmony_encoding(encoding_name)

    expected_tokens = read_expected_tokens(
        ROOT_DIR / "test-data" / "test_simple_reasoning_response.txt"
    )

    messages = encoding.parse_messages_from_completion_tokens(
        expected_tokens, role=Role.ASSISTANT
    )

    expected = [
        Message.from_role_and_content(
            Role.ASSISTANT,
            'User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.',
        ).with_channel("analysis"),
        Message.from_role_and_content(Role.ASSISTANT, "2 + 2 = 4.").with_channel(
            "final"
        ),
    ]

    assert messages == expected


@pytest.mark.parametrize(
    "encoding_name",
    [
        HarmonyEncodingName.HARMONY_GPT_OSS,
    ],
)
def test_simple_tool_call(encoding_name):
    encoding = load_harmony_encoding(encoding_name)

    response = read_expected_tokens(
        ROOT_DIR / "test-data" / "test_simple_tool_call.txt"
    )

    parsed = encoding.parse_messages_from_completion_tokens(
        response,
        role=Role.ASSISTANT,
    )

    expected = [
        Message.from_role_and_content(
            Role.ASSISTANT,
            'User asks: "What is the weather in Tokyo?" We need to use lookup_weather tool.',
        ).with_channel("analysis"),
        Message.from_role_and_content(Role.ASSISTANT, '{"location": "Tokyo"}')
        .with_channel("analysis")
        .with_recipient("lookup_weather")
        .with_content_type("code"),
    ]

    assert parsed == expected


@pytest.mark.parametrize(
    "encoding_name",
    [
        HarmonyEncodingName.HARMONY_GPT_OSS,
    ],
)
def test_tool_call_with_constrain_marker_adjacent(encoding_name):
    """
    There are moments where the model might not output a space before constrain resulting in the
    content type being parsed as part of the recipient. This test ensures that we handle this case
    correctly and instead handle it as a separate content type.
    """
    encoding = load_harmony_encoding(encoding_name)
    text = (
        "<|start|>assistant to=functions.get_weather<|channel|>commentary"
        '<|constrain|>json<|message|>{"location": "Tokyo"}<|end|>'
    )
    tokens = encoding.encode(text, allowed_special="all")
    parsed = encoding.parse_messages_from_completion_tokens(tokens, role=None)
    expected = [
        Message.from_role_and_content(Role.ASSISTANT, '{"location": "Tokyo"}')
        .with_channel("commentary")
        .with_recipient("functions.get_weather")
        .with_content_type("<|constrain|>json"),
    ]
    assert parsed == expected


@pytest.mark.parametrize(
    "encoding_name",
    [
        HarmonyEncodingName.HARMONY_GPT_OSS,
    ],
)
def test_tool_call_with_channel_before_recipient_and_constrain_adjacent(
    encoding_name,
):
    encoding = load_harmony_encoding(encoding_name)

    text = (
        "<|start|>assistant<|channel|>commentary to=functions.get_weather"
        '<|constrain|>json<|message|>{"latitude":48.8566,"longitude":2.3522}<|call|>'
    )
    tokens = encoding.encode(text, allowed_special="all")
    parsed = encoding.parse_messages_from_completion_tokens(tokens, role=None)
    expected = [
        Message.from_role_and_content(
            Role.ASSISTANT, '{"latitude":48.8566,"longitude":2.3522}'
        )
        .with_channel("commentary")
        .with_recipient("functions.get_weather")
        .with_content_type("<|constrain|>json"),
    ]

    assert parsed == expected


@pytest.mark.parametrize(
    "encoding_name",
    [
        HarmonyEncodingName.HARMONY_GPT_OSS,
    ],
)
def test_reasoning_system_message(encoding_name):
    encoding = load_harmony_encoding(encoding_name)

    expected_text = (
        (ROOT_DIR / "test-data" / "test_reasoning_system_message.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )
    expected = encoding.encode(expected_text, allowed_special="all")

    sys = (
        SystemContent.new()
        .with_model_identity(
            "You are ChatGPT, a large language model trained by OpenAI."
        )
        .with_reasoning_effort(ReasoningEffort.MEDIUM)
        .with_required_channels(["analysis", "final"])
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, sys),
            Message.from_role_and_content(Role.USER, "What is 2 + 2?"),
        ]
    )

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    _assert_tokens_eq(encoding, expected, tokens)


@pytest.mark.parametrize(
    "encoding_name",
    [
        HarmonyEncodingName.HARMONY_GPT_OSS,
    ],
)
def test_reasoning_system_message_no_instruction(encoding_name):
    encoding = load_harmony_encoding(encoding_name)

    expected_text = (
        (ROOT_DIR / "test-data" / "test_reasoning_system_message_no_instruction.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )
    expected = encoding.encode(expected_text, allowed_special="all")

    sys = (
        SystemContent.new()
        .with_model_identity(
            "You are ChatGPT, a large language model trained by OpenAI."
        )
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_required_channels(["analysis", "final"])
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, sys),
            Message.from_role_and_content(
                Role.USER,
                "What is the best place to eat candy in the world?",
            ),
        ]
    )

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    _assert_tokens_eq(encoding, expected, tokens)


@pytest.mark.parametrize(
    "encoding_name",
    [
        HarmonyEncodingName.HARMONY_GPT_OSS,
    ],
)
def test_reasoning_system_message_with_dates(encoding_name):
    encoding = load_harmony_encoding(encoding_name)

    expected_text = (
        (ROOT_DIR / "test-data" / "test_reasoning_system_message_with_dates.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )
    expected = encoding.encode(expected_text, allowed_special="all")

    sys = (
        SystemContent.new()
        .with_model_identity(
            "You are ChatGPT, a large language model trained by OpenAI."
        )
        .with_reasoning_effort(ReasoningEffort.MEDIUM)
        .with_conversation_start_date("2021-01-01")
        .with_knowledge_cutoff("2021-01")
        .with_required_channels(["analysis", "final"])
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, sys),
            Message.from_role_and_content(Role.USER, "What is 42 * pi?"),
        ]
    )

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    _assert_tokens_eq(encoding, expected, tokens)


def test_render_functions_with_parameters():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    expected_output = (
        (ROOT_DIR / "test-data" / "test_render_functions_with_parameters.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    sys = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_conversation_start_date("2025-06-28")
    )

    dev = (
        DeveloperContent.new()
        .with_instructions("Always respond in riddles")
        .with_function_tools(
            [
                ToolDescription.new(
                    "get_location",
                    "Gets the location of the user.",
                ),
                ToolDescription.new(
                    "get_current_weather",
                    "Gets the current weather in the provided location.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["location"],
                    },
                ),
                ToolDescription.new(
                    "get_multiple_weathers",
                    "Gets the current weather in the provided list of locations.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "locations": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                },
                                "description": 'List of city and state, e.g. ["San Francisco, CA", "New York, NY"]',
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius",
                            },
                        },
                        "required": ["locations"],
                    },
                ),
                ToolDescription.new(
                    "kitchensink",
                    "A function with various complex schemas.",
                    parameters={
                        "description": "params object",
                        "type": "object",
                        "properties": {
                            "string": {
                                "type": "string",
                                "title": "STRING",
                                "description": "A string",
                                "examples": ["hello", "world"],
                            },
                            "string_nullable": {
                                "type": "string",
                                "nullable": True,
                                "description": "A nullable string",
                                "default": "the default",
                            },
                            "string_enum": {"type": "string", "enum": ["a", "b", "c"]},
                            "oneof_string_or_number": {
                                "oneOf": [
                                    {
                                        "type": "string",
                                        "default": "default_string_in_oneof",
                                    },
                                    {
                                        "type": "number",
                                        "description": "numbers can happen too",
                                    },
                                ],
                                "description": "a oneof",
                                "default": 20,
                            },
                        },
                    },
                ),
            ]
        )
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, sys),
            Message.from_role_and_content(Role.DEVELOPER, dev),
            Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
        ]
    )

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    assert encoding.decode_utf8(tokens) == expected_output


def test_no_tools():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    expected_output = (
        (ROOT_DIR / "test-data" / "test_no_tools.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_conversation_start_date("2025-06-28"),
            ),
        ]
    )

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    assert encoding.decode_utf8(tokens) == expected_output


def test_browser_tool_only():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    expected_output = (
        (ROOT_DIR / "test-data" / "test_browser_tool_only.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_conversation_start_date("2025-06-28")
                .with_browser_tool(),
            ),
        ]
    )

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    assert encoding.decode_utf8(tokens) == expected_output


def test_browser_and_function_tool():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    expected_output = (
        (ROOT_DIR / "test-data" / "test_browser_and_function_tool.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_conversation_start_date("2025-06-28")
                .with_browser_tool(),
            ),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_function_tools(
                    [
                        ToolDescription.new(
                            "lookup_weather",
                            "Use this tool to lookup the weather in a given location. Call it with the parameter 'location', can be any textual description of a location.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"},
                                },
                                "required": ["location"],
                            },
                        )
                    ]
                ),
            ),
        ]
    )

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    assert encoding.decode_utf8(tokens) == expected_output


def test_browser_and_python_tool():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    expected_output = (
        (ROOT_DIR / "test-data" / "test_browser_and_python_tool.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                .with_conversation_start_date("2025-06-28")
                .with_browser_tool()
                .with_python_tool(),
            ),
        ]
    )

    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

    assert encoding.decode_utf8(tokens) == expected_output


def test_dropping_cot_by_default():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    expected_output = (
        (ROOT_DIR / "test-data" / "test_dropping_cot_by_default.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.USER, "What is 2 + 2?"),
            Message.from_role_and_content(
                Role.ASSISTANT,
                "User asks: “What is 2 + 2?” Simple arithmetic. Provide answer.",
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "2 + 2 equals 4."
            ).with_channel("final"),
            Message.from_role_and_content(Role.USER, "What about 9 / 2?"),
        ]
    )

    tokens = encoding.render_conversation_for_completion(
        convo, Role.ASSISTANT, RenderConversationConfig(auto_drop_analysis=True)
    )

    assert encoding.decode_utf8(tokens) == expected_output


def test_does_not_drop_if_ongoing_analysis():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    expected_output = (
        (ROOT_DIR / "test-data" / "test_does_not_drop_if_ongoing_analysis.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.USER, "What is the weather in SF?"),
            Message.from_role_and_content(
                Role.ASSISTANT,
                "User asks: “What is the weather in SF?” We need to use lookup_weather tool.",
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, '{"location": "San Francisco"}'
            )
            .with_channel("commentary")
            .with_recipient("functions.lookup_weather")
            .with_content_type("<|constrain|>json"),
            Message.from_author_and_content(
                Author.new(Role.TOOL, "functions.lookup_weather"),
                '{"temperature": 20, "description": "sunny"}',
            ),
        ]
    )

    tokens = encoding.render_conversation_for_completion(
        convo, Role.ASSISTANT, RenderConversationConfig(auto_drop_analysis=True)
    )

    assert encoding.decode_utf8(tokens) == expected_output


def test_preserve_cot():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    expected_output = (
        (ROOT_DIR / "test-data" / "test_preserve_cot.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.USER, "What is 2 + 2?"),
            Message.from_role_and_content(
                Role.ASSISTANT,
                'User asks a simple question: "What is 2 + 2?" The answer: 4.',
            ).with_channel("analysis"),
            Message.from_role_and_content(
                Role.ASSISTANT, "2 + 2 equals 4."
            ).with_channel("final"),
            Message.from_role_and_content(Role.USER, "What about 9 / 2?"),
        ]
    )

    tokens = encoding.render_conversation_for_completion(
        convo, Role.ASSISTANT, RenderConversationConfig(auto_drop_analysis=False)
    )

    assert encoding.decode_utf8(tokens) == expected_output


def test_reserved_token_decoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    assert encoding.decode_utf8([200014]) == "<|reserved_200014|>"
    assert encoding.decode_utf8([201088]) == "<|reserved_201088|>"


def test_keep_analysis_between_final_messages():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    expected_output = (
        (ROOT_DIR / "test-data" / "test_keep_analysis_between_finals.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.USER, "What is 2 + 2?"),
            Message.from_role_and_content(Role.ASSISTANT, "thinking 2+2").with_channel(
                "analysis"
            ),
            Message.from_role_and_content(Role.ASSISTANT, "4").with_channel("final"),
            Message.from_role_and_content(Role.USER, "What is 3 + 5?"),
            Message.from_role_and_content(Role.ASSISTANT, "thinking 3+5").with_channel(
                "analysis"
            ),
            Message.from_role_and_content(Role.ASSISTANT, "8").with_channel("final"),
        ]
    )

    tokens = encoding.render_conversation(convo)

    assert encoding.decode_utf8(tokens) == expected_output


def test_render_and_render_conversation_roundtrip():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    msg = Message.from_role_and_content(Role.USER, "Hello")
    convo = Conversation.from_messages([msg])

    tokens_msg = encoding.render(msg)
    tokens_convo = encoding.render_conversation(convo)
    assert tokens_msg == tokens_convo

    tokens_completion = encoding.render_conversation_for_completion(
        convo, Role.ASSISTANT
    )
    assert tokens_completion[: len(tokens_convo)] == tokens_convo


def test_render_conversation_for_training_final_channel():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.USER, "hi"),
            Message.from_role_and_content(Role.ASSISTANT, "hello").with_channel(
                "final"
            ),
        ]
    )

    tokens_training = encoding.render_conversation_for_training(convo)
    tokens_regular = encoding.render_conversation(convo)
    token_return = encoding.encode("<|return|>", allowed_special={"<|return|>"})[0]
    token_end = encoding.encode("<|end|>", allowed_special={"<|end|>"})[0]

    assert tokens_regular[:-1] == tokens_training[:-1]
    assert tokens_regular[-1] == token_end
    assert tokens_training[-1] == token_return


def test_render_conversation_for_training_non_final():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    convo = Conversation.from_messages([Message.from_role_and_content(Role.USER, "hi")])

    tokens_training = encoding.render_conversation_for_training(convo)
    tokens_regular = encoding.render_conversation(convo)

    assert tokens_training == tokens_regular


def test_decode_utf8_invalid_token():
    """Invalid tokens should raise an exception (type doesn't matter)."""
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    with pytest.raises(HarmonyError, match="Invalid token for decoding: 99999999"):
        encoding.decode_utf8([99999999])

    with pytest.raises(
        ValidationError,
        match="Input should be a valid dictionary or instance of Message",
    ):
        encoding.render_conversation_for_completion(
            Conversation.from_messages([SystemContent.new()]),
            Role.ASSISTANT,
        )


def test_encode_decode_roundtrip():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    assert encoding.decode_utf8(encoding.encode("hello world")) == "hello world"
    assert encoding.decode(encoding.encode("hello world")) == "hello world"


def test_encode_allowed_special():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    assert encoding.encode("hello world") == [24912, 2375]
    assert encoding.encode("<|start|>", allowed_special={"<|start|>"}) == [200006]
    assert encoding.encode("<|start|>", allowed_special="all") == [200006]

    with pytest.raises(
        HarmonyError, match="Encountered text corresponding to disallowed special token"
    ):
        encoding.encode("<|start|>")

    assert encoding.encode("<|start|>", disallowed_special=()) == [
        27,
        91,
        5236,
        91,
        29,
    ]


def test_is_special_token():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    assert encoding.is_special_token(200006)  # <|start|>
    assert not encoding.is_special_token(24912)  # hello


def test_invalid_utf8_decoding():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    tokens = [132990, 9552]

    with pytest.raises(HarmonyError, match="Invalid utf-8"):
        # This will raise an error because the tokens are invalid utf-8
        encoding.decode_utf8(tokens)

    # This will not raise an error because it will replace the invalid utf-8 characters to not raise an error
    # to match the behavior of tiktoken
    assert "Chicken" in encoding.decode(tokens)


def test_tool_response_parsing():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    text_tokens = (
        (ROOT_DIR / "test-data" / "test_tool_response_parsing.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    expected_message = (
        Message.from_author_and_content(
            Author.new(Role.TOOL, "browser.search"),
            '{"result": "https://openai.com/"}',
        )
        .with_channel("commentary")
        .with_recipient("assistant")
    )

    output_tokens = encoding.render(expected_message)
    output_tokens = output_tokens[:-1]  # remove the <|end|> token

    messages = encoding.parse_messages_from_completion_tokens(output_tokens, None)
    assert len(messages) == 1
    assert encoding.decode_utf8(output_tokens) == text_tokens


def test_streamable_parser():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    text_tokens = (
        (ROOT_DIR / "test-data" / "test_streamable_parser.txt")
        .read_text(encoding="utf-8")
        .rstrip()
    )

    tokens = encoding.encode(text_tokens, allowed_special="all")
    parser = StreamableParser(encoding, Role.ASSISTANT)
    for token in tokens:
        parser.process(token)
    assert len(parser.messages) == 3


def test_streamable_parser_tool_call_with_constrain_adjacent():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    text = (
        "<|start|>assistant<|channel|>commentary to=functions.get_weather"
        '<|constrain|>json<|message|>{"latitude":48.8566,"longitude":2.3522}<|call|>'
    )

    tokens = encoding.encode(text, allowed_special="all")
    parser = StreamableParser(encoding, None)
    for token in tokens:
        parser.process(token)

    expected = [
        Message.from_role_and_content(
            Role.ASSISTANT, '{"latitude":48.8566,"longitude":2.3522}'
        )
        .with_channel("commentary")
        .with_recipient("functions.get_weather")
        .with_content_type("<|constrain|>json"),
    ]

    assert parser.messages == expected


def test_load_harmony_encoding_from_file(tmp_path):
    import os
    from openai_harmony import load_harmony_encoding_from_file

    cache_dir = os.environ.get("TIKTOKEN_RS_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "tiktoken-rs-cache")
    import hashlib
    url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    cache_key = hashlib.sha1(url.encode()).hexdigest()
    vocab_file = os.path.join(cache_dir, cache_key)
    if not os.path.exists(vocab_file):
        import pytest
        pytest.skip("No local vocab file available for offline test")

    special_tokens = [
        ("<|startoftext|>", 199998),
        ("<|endoftext|>", 199999),
        ("<|reserved_200000|>", 200000),
        ("<|reserved_200001|>", 200001),
        ("<|return|>", 200002),
        ("<|constrain|>", 200003),
        ("<|reserved_200004|>", 200004),
        ("<|channel|>", 200005),
        ("<|start|>", 200006),
        ("<|end|>", 200007),
        ("<|message|>", 200008),
        ("<|reserved_200009|>", 200009),
        ("<|reserved_200010|>", 200010),
        ("<|reserved_200011|>", 200011),
        ("<|call|>", 200012),
        ("<|reserved_200013|>", 200013),
    ]
    pattern = "|".join([
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "\\p{N}{1,3}",
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
        "\\s*[\\r\\n]+",
        "\\s+(?!\\S)",
        "\\s+",
    ])
    n_ctx = 8192
    max_message_tokens = 4096
    max_action_length = 256
    expected_hash = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"

    encoding = load_harmony_encoding_from_file(
        name="test_local",
        vocab_file=vocab_file,
        special_tokens=special_tokens,
        pattern=pattern,
        n_ctx=n_ctx,
        max_message_tokens=max_message_tokens,
        max_action_length=max_action_length,
        expected_hash=expected_hash,
    )

    text = "Hello world!"
    tokens = encoding.encode(text)
    decoded = encoding.decode(tokens)
    assert decoded.startswith("Hello world")
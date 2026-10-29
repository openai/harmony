import pytest

from openai_harmony import (
    HarmonyEncodingName,
    HarmonyError,
    Role,
    StreamableParser,
    load_harmony_encoding,
)


@pytest.fixture(scope="module")
def encoding():
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def feed(parser, tokens):
    observations = []
    for token in tokens:
        parser.process(token)
        observations.append(
            (
                parser.last_content_delta,
                parser.current_role,
                parser.current_channel,
                parser.current_recipient,
                parser.current_content_type,
                parser.message_count,
            )
        )
    return observations


def test_delta_only_discards_history_but_preserves_deltas(encoding):
    body = "hello world " * 5461
    prefix = "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>"
    tokens = encoding.encode(prefix + body + "<|call|>", allowed_special="all")
    full = StreamableParser(encoding, None)
    delta = StreamableParser(encoding, None, retain_history=False)
    assert feed(delta, tokens) == feed(full, tokens)
    assert full.tokens == tokens
    assert len(full.messages) == full.message_count == delta.message_count == 1
    assert delta.tokens == []
    assert delta.messages == []
    assert not delta.retains_history
    with pytest.raises(HarmonyError, match="history"):
        _ = delta.current_content


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("retain_history", [True, False])
@pytest.mark.parametrize(
    "wire",
    [
        '<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{"q":"🙂漢字𐍈"}<|call|>',
        '<|start|>assistant<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>done<|return|>',
        '<|start|>assistant<|channel|>commentary to=functions.lookup<|constrain|>json<|message|>{"nested":[1,{"x":"\\\\\\\""}]}<|call|>',
    ],
)
def test_fork_matches_every_token_split(encoding, strict, retain_history, wire):
    tokens = encoding.encode(wire, allowed_special="all")
    for split in range(len(tokens) + 1):
        full = StreamableParser(encoding, None, strict=strict)
        base = StreamableParser(
            encoding, None, strict=strict, retain_history=retain_history
        )
        assert feed(base, tokens[:split]) == feed(full, tokens[:split])
        fork = base.fork_continuation()
        assert fork.last_content_delta is None
        assert fork.last_completion_delta is None
        assert fork.tokens == fork.messages == []
        assert fork.message_count == base.message_count
        assert feed(fork, tokens[split:]) == feed(full, tokens[split:])
        assert fork.message_count == len(full.messages)


def test_forks_keep_divergent_suffixes_independent(encoding):
    prefix = '<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{"q":"'
    base = StreamableParser(encoding, None)
    tokens = encoding.encode(prefix, allowed_special="all")
    feed(base, tokens)
    left = base.fork_continuation()
    right = base.fork_continuation()
    left_result = feed(left, encoding.encode('left"}<|call|>', allowed_special="all"))
    right_result = feed(right, encoding.encode('right"}<|call|>', allowed_special="all"))
    assert "".join(row[0] or "" for row in left_result) == 'left"}'
    assert "".join(row[0] or "" for row in right_result) == 'right"}'
    assert left.message_count == right.message_count == 1
    assert base.message_count == 0
    assert base.tokens == tokens
    assert base.current_content == '{"q":"'


def test_fork_preserves_partial_utf8_and_eof(encoding):
    wire = "<|start|>assistant<|channel|>final<|message|>𐍈🙂"
    tokens = encoding.encode(wire, allowed_special="all")
    fragments = 0
    for token in encoding.encode("𐍈", allowed_special="all"):
        try:
            encoding.decode_utf8([token])
        except HarmonyError:
            fragments += 1
    assert fragments > 0, "fixture must contain a split UTF-8 code point"
    for split in range(len(tokens) + 1):
        full = StreamableParser(encoding, None)
        base = StreamableParser(encoding, None, retain_history=False)
        feed(full, tokens[:split])
        feed(base, tokens[:split])
        fork = base.fork_continuation()
        assert feed(fork, tokens[split:]) == feed(full, tokens[split:])
        fork.process_eos()
        full.process_eos()
        assert fork.message_count == full.message_count == 1
        assert fork.last_content_delta == full.last_content_delta


def test_fork_preserves_incomplete_header_error(encoding):
    base = StreamableParser(encoding, Role.ASSISTANT, retain_history=False)
    feed(base, encoding.encode(" to=functions.", allowed_special="all"))
    fork = base.fork_continuation()
    with pytest.raises(HarmonyError):
        base.process_eos()
    with pytest.raises(HarmonyError):
        fork.process_eos()


def test_delta_completion_preserves_eof_at_every_utf8_token(encoding):
    header = "<|start|>assistant<|channel|>final<|message|>"
    header_tokens = encoding.encode(header, allowed_special="all")
    body_tokens = encoding.encode("hello 𐍈🙂", allowed_special="all")
    for split in range(len(body_tokens) + 1):
        full = StreamableParser(encoding, None)
        delta = StreamableParser(encoding, None, retain_history=False)
        tokens = header_tokens + body_tokens[:split]
        observations = feed(delta, tokens)
        feed(full, tokens)
        full.process_eos()
        delta.process_eos()
        body = "".join(row[0] or "" for row in observations)
        body += delta.last_completion_delta or ""
        assert body == full.messages[0].content[0].text
        assert delta.message_count == full.message_count == 1
        delta.process_eos()
        assert delta.message_count == 1
        assert delta.last_completion_delta is None


@pytest.mark.parametrize("role", [Role.ASSISTANT, None])
@pytest.mark.parametrize("text", ["", "I must refuse", "<|channel|>final hello"])
def test_delta_completion_preserves_permissive_header_text(encoding, role, text):
    start = "<|start|>assistant " if role is None else ""
    tokens = encoding.encode(start + text + "<|end|>", allowed_special="all")
    full = StreamableParser(encoding, role, strict=False)
    delta = StreamableParser(encoding, role, strict=False, retain_history=False)
    feed(full, tokens)
    observations = feed(delta, tokens)
    body = "".join(row[0] or "" for row in observations)
    body += delta.last_completion_delta or ""
    assert body == "".join(message.content[0].text for message in full.messages)
    assert delta.message_count == len(full.messages)


def test_delta_completion_preserves_invalid_token_replacement(encoding):
    header = encoding.encode(
        "<|start|>assistant<|channel|>final<|message|>", allowed_special="all"
    )
    full = StreamableParser(encoding, None)
    delta = StreamableParser(encoding, None, retain_history=False)
    feed(full, header + [2**32 - 1])
    feed(delta, header + [2**32 - 1])
    full.process_eos()
    delta.process_eos()
    assert delta.last_completion_delta == full.messages[0].content[0].text == "�"


@pytest.mark.parametrize("invalid", [False, True])
def test_fork_immediate_eof_preserves_pending_decoding(encoding, invalid):
    header = encoding.encode(
        "<|start|>assistant<|channel|>final<|message|>", allowed_special="all"
    )
    body = [2**32 - 1] if invalid else encoding.encode("𐍈🙂", allowed_special="all")
    for split in range(len(body) + 1):
        base = StreamableParser(encoding, None, retain_history=False)
        feed(base, header + body[:split])
        forks = [base.fork_continuation(), base.fork_continuation()]
        base.process_eos()
        for fork in forks:
            fork.process_eos()
            assert fork.last_completion_delta == base.last_completion_delta
            assert fork.message_count == base.message_count == 1


def test_discard_history_at_every_boundary_is_idempotent(encoding):
    wire = (
        "<|start|>assistant<|channel|>final<|message|>first<|end|>"
        "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>"
        '{"q":"𐍈🙂"}<|call|>'
    )
    tokens = encoding.encode(wire, allowed_special="all")
    for split in range(len(tokens) + 1):
        converted = StreamableParser(encoding, None)
        delta = StreamableParser(encoding, None, retain_history=False)
        assert feed(converted, tokens[:split]) == feed(delta, tokens[:split])
        count = converted.message_count
        converted.discard_history()
        converted.discard_history()
        assert converted.tokens == converted.messages == []
        assert converted.message_count == count
        assert not converted.retains_history
        assert feed(converted, tokens[split:]) == feed(delta, tokens[split:])
        converted.process_eos()
        delta.process_eos()
        converted.discard_history()
        assert converted.message_count == delta.message_count == 2
        assert converted.last_completion_delta == delta.last_completion_delta

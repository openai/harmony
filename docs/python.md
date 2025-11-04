# Python API Reference

The `openai-harmony` package exposes the Harmony renderer to Python via thin bindings generated with `PyO3`.  It installs a module named `openai_harmony` which re‑exports a set of dataclasses mirroring the structures from the Rust crate together with helper classes for encoding and parsing.

## Installation

Install the package from PyPI:

```bash
pip install openai-harmony
```

Typical imports look like:

```python
from openai_harmony import Message, Conversation, load_harmony_encoding
```

## Enumerations

### `Role`
Represents the author of a message.  Possible values are `USER`, `ASSISTANT`, `SYSTEM`, `DEVELOPER` and `TOOL`.

### `ReasoningEffort`
Defines how much reasoning the assistant should apply.  Values: `LOW`, `MEDIUM`, `HIGH`.

### `StreamState`
State of an incremental parser: `EXPECT_START`, `HEADER` or `CONTENT`.

## Dataclasses

### `Author`
```python
Author(role: Role, name: Optional[str] = None)
```
Helper for specifying the author of a message.  Use `Author.new(role, name)` to create a named author.

### `TextContent`
```python
TextContent(text: str)
```
Simple text payload implementing `Content`.

### `ToolDescription`
```python
ToolDescription(name: str, description: str, parameters: Optional[dict] = None)
```
Describes an individual callable tool.

### `ToolNamespaceConfig`
```python
ToolNamespaceConfig(name: str, description: Optional[str], tools: List[ToolDescription])
```
Namespace for grouping tools.  Convenience constructors `browser()` and `python()` return the built‑in configurations.

### `ChannelConfig`
```python
ChannelConfig(valid_channels: List[str], channel_required: bool)
```
Configuration for valid message channels.  Use `ChannelConfig.require_channels(channels)` to demand that a specific set of channels is present.

### `SystemContent`
```python
SystemContent(
    model_identity: Optional[str] = "You are ChatGPT, a large language model trained by OpenAI.",
    reasoning_effort: Optional[ReasoningEffort] = ReasoningEffort.MEDIUM,
    conversation_start_date: Optional[str] = None,
    knowledge_cutoff: Optional[str] = "2024-06",
    channel_config: Optional[ChannelConfig] = ChannelConfig.require_channels(["analysis", "commentary", "final"]),
    tools: Optional[dict[str, ToolNamespaceConfig]] = None,
)
```
Represents a system message.  Provides fluent helpers like `with_model_identity()`, `with_reasoning_effort()`, `with_required_channels()`, `with_browser_tool()` and `with_python_tool()`.

### `DeveloperContent`
```python
DeveloperContent(instructions: Optional[str] = None, tools: Optional[dict[str, ToolNamespaceConfig]] = None)
```
Content of a developer message.  Helper methods include `with_instructions()`, `with_function_tools()` and the tool helpers mirroring those from `SystemContent`.

### `Message`
```python
Message(author: Author, content: List[Content], channel: Optional[str] = None, recipient: Optional[str] = None, content_type: Optional[str] = None)
```
A single chat message.  Convenience constructors `from_role_and_content()` and `from_role_and_contents()` mirror the Rust API.  Builder methods allow adding content or setting channel/recipient information.  `to_dict()` and `from_dict()` serialise to/from the canonical JSON format.

### `Conversation`
```python
Conversation(messages: List[Message])
```
Sequence of messages.  Create a conversation using `Conversation.from_messages()`; serialisation helpers `to_json()` and `from_json()` mirror the Rust crate.

### `RenderConversationConfig`
```python
RenderConversationConfig(auto_drop_analysis: bool = True)
```
Optional configuration when rendering a conversation.

## Encoding helpers

### `HarmonyEncoding`
Wrapper around the low level bindings.  Obtain an instance with `load_harmony_encoding()`.

Methods:
- `name` – name of the encoding.
- `render_conversation_for_completion(conversation, next_turn_role, config=None)` – render a conversation into tokens.
- `render_conversation_for_training(conversation, config=None)` – render a conversation for training.
- `render_conversation(conversation, config=None)` – render a conversation without appending a new role.
- `render(message)` – render a single message into tokens.
- `parse_messages_from_completion_tokens(tokens, role=None, strict=True)` – parse tokens back into `Message` objects (set `strict=False` to enable permissive parsing).
- `decode_utf8(tokens)` – decode tokens with the underlying tokenizer.
- `stop_tokens()` / `stop_tokens_for_assistant_actions()` – lists of stop tokens.

Use `strict=False` when you need the parser to recover from malformed model output that omits markers such as `<|message|>`.

### `StreamableParser`
Incremental parser built on top of an encoding. Construct with `StreamableParser(encoding, role)` and feed tokens via `process(token)`.  Inspect state via properties like `current_content`, `current_role`, `tokens` and `state`. Pass `strict=False` to enable permissive parsing (mirrors `ParseOptions { strict: false }` on the Rust side).

### `load_harmony_encoding(name)`
Return a `HarmonyEncoding` by name.  Accepts either the string name or a value from the `HarmonyEncodingName` enum (`HARMONY_GPT_OSS`).

## Exports
The package re‑exports the above classes through `__all__` so they are available via:
```python
from openai_harmony import *
```

## Usage Examples

Create a conversation with a system and user message, render it to tokens and
parse them back again:

```python
from openai_harmony import (
    Role,
    Message,
    Conversation,
    SystemContent,
    load_harmony_encoding,
    HarmonyEncodingName,
)

# Build messages
system = Message.from_role_and_content(Role.SYSTEM, SystemContent.new())
user = Message.from_role_and_content(Role.USER, "What is 2 + 2?")

# Assemble a conversation
convo = Conversation.from_messages([system, user])

# Render to tokens using the OSS encoding
enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
print(tokens)

# Decode and roundtrip
print(enc.decode_utf8(tokens))
parsed = enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT)
for m in parsed:
    print(m)
```

## Exceptions

The bindings raise plain Python exceptions.  The most common ones are:

- `RuntimeError` – returned for rendering or parsing failures (for example if a
  token sequence is malformed or decoding fails).
- `ValueError` – raised when an argument is invalid, e.g. an unknown
  `Role` is provided to `load_harmony_encoding` or `StreamableParser`.
- `ModuleNotFoundError` – accessing the package without building the compiled
  extension results in this error.

In typical code you would wrap encoding operations in a `try`/`except` block:

```python
try:
    tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
    parsed = enc.parse_messages_from_completion_tokens(tokens, Role.ASSISTANT)
except RuntimeError as err:
    print(f"Failed to handle conversation: {err}")
```

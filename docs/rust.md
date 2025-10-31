# Rust API Reference

The Rust crate provides the core rendering and parsing logic. It is organised into a few modules which are re‑exported at the crate root for convenience.

## Crate Setup

Add the crate to your `Cargo.toml`:

```toml
openai-harmony = { git = "https://github.com/openai/harmony" }
```

and import the items you need:

```rust
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};
use openai_harmony::chat::{Message, Role, Conversation};
```

## Crate Layout

- `chat` – data structures representing messages and conversations.
- `encoding` – tokenisation and rendering implementation.
- `registry` – helper for loading predefined encodings.
- `tiktoken_ext` – extensions for `tiktoken`.

Key items are re‑exported so you can simply:

```rust
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use openai_harmony::chat::{Message, Role, Conversation};
```

## chat module

### `Role`

```rust
enum Role { User, Assistant, System, Developer, Tool }
```

Represents the author of a message.

### `Author`

```rust
struct Author { role: Role, name: Option<String> }
```

Helper for identifying the message author. `Author::new(role, name)` creates a named author and `Author::from(role)` creates an anonymous one.

### `TextContent`

Simple text payload. Implements `Into<Content>` so it can be passed directly to message constructors.

### `SystemContent` and `DeveloperContent`

Structures used for the respective message types. They offer builder‑style methods (`with_model_identity`, `with_instructions`, `with_tools`, …) to configure the message payload.

### `Message`

```rust
struct Message { author: Author, recipient: Option<String>, content: Vec<Content>, channel: Option<String>, content_type: Option<String> }
```

Convenience constructors mirror those exposed in Python (`from_role_and_content`, `adding_content`, etc.).

### `Conversation`

```rust
struct Conversation { messages: Vec<Message> }
```

Created via `Conversation::from_messages`.

## encoding module

### `HarmonyEncoding`

Represents an encoding instance. Obtainable through `load_harmony_encoding`.

Important methods:

- `name()` – name of the encoding.
- `tokenizer_name()` – name of the underlying tokenizer.
- `max_message_tokens()` – maximum number of tokens a single message may use.
- `render_conversation_for_completion(conversation, next_role, config)` – convert a conversation into tokens ready for inference.
- `render_conversation_for_training(conversation, config)` – render a conversation for training data.
- `render_conversation(conversation, config)` – render a conversation without appending a new role.
- `render(message)` – render a single message into tokens.
- `parse_messages_from_completion_tokens(tokens, role)` – parse a list of tokens back into messages using strict validation.
- `parse_messages_from_completion_tokens_with_options(tokens, role, options)` – parse tokens with custom `ParseOptions` (e.g. to disable strict validation).
- `stop_tokens()` and `stop_tokens_for_assistant_actions()` – sets of stop tokens for sampling.

`ParseOptions` currently exposes a single field, `strict`, which defaults to `true`. Set it to `false` when you need to recover from malformed model output in downstream systems.

### `StreamableParser`

Incremental parser that consumes tokens one by one. Create with `StreamableParser::new(encoding, role)` and feed tokens via `process`. Access information via getters like `current_content`, `current_role`, `messages`, `tokens` and `state_json`. Use `StreamableParser::new_with_options(encoding, role, options)` when you need to override defaults such as `ParseOptions { strict: false }`.

## registry module

### `load_harmony_encoding`

```rust
fn load_harmony_encoding(name: HarmonyEncodingName) -> Result<HarmonyEncoding>
```

Load a predefined encoding by name.

### `HarmonyEncodingName`

```rust
enum HarmonyEncodingName { HarmonyGptOss }
```

Enum of the available encodings. It implements `FromStr` and `Display`.

## Feature flags

If the `python-binding` feature is enabled, the crate exposes a Python module via `pyo3` (see `src/py_module.rs`). This module is used by the accompanying Python package but can be ignored when using the crate purely from Rust.

## Usage Examples

Below is a minimal program that builds a conversation, renders it using the
`HarmonyEncoding`, and parses the tokens back into messages:

```rust
use openai_harmony::chat::{Conversation, Message, Role, SystemContent};
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some messages
    let convo = Conversation::from_messages([
        Message::from_role_and_content(Role::System, SystemContent::new()),
        Message::from_role_and_content(Role::User, "What is 2 + 2?"),
    ]);

    // Render tokens for completion
    let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let tokens = enc.render_conversation_for_completion(&convo, Role::Assistant, None)?;
    println!("{:?}", tokens);

    // Decode & parse back
    println!("{}", enc.decode_utf8(&tokens)?);
    let parsed = enc.parse_messages_from_completion_tokens(tokens, Some(Role::Assistant))?;
    for m in parsed { println!("{:?}", m); }
    Ok(())
}
```

## Error Handling

Most functions return `anyhow::Result<T>` (a type alias for
`Result<T, anyhow::Error>`). Errors may originate from loading encodings
(`LoadError`), rendering problems (`RenderFormattingTokenError`) or parsing
failures when the token stream is malformed.

Typical applications propagate errors with the `?` operator:

```rust
let tokens = enc.render_conversation_for_completion(&convo, Role::Assistant, None)?;
let parsed = enc.parse_messages_from_completion_tokens(tokens, Some(Role::Assistant))?;
```

You can also match on the underlying error kinds via `anyhow::Error` if you
need more specific handling.

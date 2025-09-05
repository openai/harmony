# OpenAI Harmony C++ Library

This is a complete C++ conversion of the OpenAI Harmony library, which provides response formatting for OpenAI's open-weight model series gpt-oss. The library offers consistent formatting for rendering and parsing conversation structures, generating reasoning output, and structuring function calls.

## Features

- **Consistent formatting** – Shared implementation for rendering and parsing keeps token sequences loss-free
- **High performance** – Optimized C++20 implementation with modern features
- **Thread-safe** – Safe for concurrent use across multiple threads
- **Complete API** – Full feature parity with the original Rust/Python implementation

## Architecture Overview

The C++ conversion maintains the same architectural principles as the original:

```
┌──────────────────┐      ┌───────────────────────────┐
│  Application     │      │  C++ Library (this repo)  │
│  (your code)     │────► │  • chat / encoding logic  │
│                  │      │  • tokenizer (tiktoken)   │
└──────────────────┘      └───────────────────────────┘
```

### Key Components

- **Chat Module** (`chat.hpp/cpp`) - High-level data structures (Role, Message, Conversation, etc.)
- **Encoding Module** (`encoding.hpp/cpp`) - Rendering & parsing implementation
- **Tiktoken Module** (`tiktoken.hpp/cpp`) - Tokenization using BPE algorithm
- **Registry Module** (`registry.hpp/cpp`) - Built-in encodings management
- **Utils Module** (`utils.hpp/cpp`) - Utility functions for base64, SHA, string operations

## Requirements

### System Requirements
- **C++ Standard**: C++20 or later
- **Platform**: Linux (primary target)
- **Build System**: CMake 3.20+

### Dependencies

#### Required Libraries
- **nlohmann/json** - JSON parsing and serialization
- **OpenSSL** - Cryptographic functions (SHA1, SHA256)
- **libcurl** - HTTP client functionality
- **pthread** - Threading support

#### Optional Dependencies (for testing and examples)
- **Google Test** - Unit testing framework
- **Google Benchmark** - Performance benchmarking

## Installation

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev

# For testing (optional)
sudo apt install -y libgtest-dev libgmock-dev

# Clone and build
git clone <repository-url>
cd harmony-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install system-wide (optional)
sudo make install
```

### CentOS/RHEL/Fedora

```bash
# CentOS/RHEL 8+
sudo dnf install -y \
    gcc-c++ \
    cmake \
    pkgconfig \
    openssl-devel \
    libcurl-devel \
    json-devel

# Fedora
sudo dnf install -y \
    gcc-c++ \
    cmake \
    pkgconfig \
    openssl-devel \
    libcurl-devel \
    nlohmann-json-devel

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### From Source (Manual Dependency Installation)

If your distribution doesn't have nlohmann/json packaged:

```bash
# Install nlohmann/json from source
git clone https://github.com/nlohmann/json.git
cd json
mkdir build && cd build
cmake .. -DJSON_BuildTests=OFF
make -j$(nproc)
sudo make install

# Then build harmony
cd /path/to/harmony-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### Basic Example

```cpp
#include <openai_harmony/harmony.hpp>
#include <iostream>

using namespace openai_harmony;

int main() {
    // Load the harmony encoding
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    
    // Create a conversation
    auto conversation = Conversation::from_messages({
        Message::from_role_and_content(
            Role::System,
            SystemContent::new_system_content()
                .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
        ),
        Message::from_role_and_content(Role::User, TextContent("Hello there!"))
    });
    
    // Render for completion
    auto tokens = encoding->render_conversation_for_completion(conversation, Role::Assistant);
    
    // Print tokens
    std::cout << "Tokens: ";
    for (auto token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

### Advanced Usage with Tools

```cpp
#include <openai_harmony/harmony.hpp>

using namespace openai_harmony;

int main() {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    
    // Create system content with tools
    auto system_content = SystemContent::new_system_content()
        .with_reasoning_effort(ReasoningEffort::High)
        .with_browser_tool()
        .with_python_tool();
    
    // Create developer content with custom functions
    auto dev_content = DeveloperContent::new_developer_content()
        .with_instructions("Always respond in riddles")
        .with_function_tools({
            ToolDescription::new_tool(
                "get_weather",
                "Gets the current weather for a location",
                nlohmann::json{
                    {"type", "object"},
                    {"properties", {
                        {"location", {{"type", "string"}}},
                        {"format", {{"type", "string"}, {"enum", {"celsius", "fahrenheit"}}, {"default", "celsius"}}}
                    }},
                    {"required", {"location"}}
                }
            )
        });
    
    auto conversation = Conversation::from_messages({
        Message::from_role_and_content(Role::System, system_content),
        Message::from_role_and_content(Role::Developer, dev_content),
        Message::from_role_and_content(Role::User, TextContent("What's the weather like in SF?"))
    });
    
    auto tokens = encoding->render_conversation_for_completion(conversation, Role::Assistant);
    
    // Decode back to text to see the formatted output
    std::string formatted = encoding->tokenizer().decode_utf8(tokens);
    std::cout << "Formatted conversation:\n" << formatted << std::endl;
    
    return 0;
}
```

### Streaming Parser Example

```cpp
#include <openai_harmony/harmony.hpp>

using namespace openai_harmony;

int main() {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    
    // Create a streaming parser
    StreamableParser parser(*encoding, Role::Assistant);
    
    // Simulate receiving tokens one by one
    std::vector<Rank> incoming_tokens = {200005, 35644, 200008, /* ... more tokens ... */};
    
    for (auto token : incoming_tokens) {
        parser.process(token);
        
        // Check current state
        if (auto content = parser.current_content(); !content.empty()) {
            std::cout << "Current content: " << content << std::endl;
        }
        
        if (auto role = parser.current_role()) {
            std::cout << "Current role: " << role_to_string(*role) << std::endl;
        }
    }
    
    // Finalize parsing
    parser.process_eos();
    
    // Get all parsed messages
    for (const auto& message : parser.messages()) {
        std::cout << "Message from " << role_to_string(message.author.role) << std::endl;
    }
    
    return 0;
}
```

## Building and Testing

### Build Options

```bash
# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release build with optimizations
cmake .. -DCMAKE_BUILD_TYPE=Release

# Disable tests
cmake .. -DBUILD_TESTS=OFF

# Disable examples
cmake .. -DBUILD_EXAMPLES=OFF

# Custom install prefix
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
```

### Running Tests

```bash
# Build with tests enabled
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)

# Run tests
ctest --verbose

# Or run directly
./harmony_tests
```

### Performance Benchmarks

```bash
# Build with benchmarks
cmake .. -DBUILD_BENCHMARKS=ON
make -j$(nproc)

# Run benchmarks
./harmony_benchmarks
```

## API Reference

### Core Classes

#### `HarmonyEncoding`
Main class for encoding and decoding conversations.

**Key Methods:**
- `render_conversation_for_completion()` - Render conversation for model completion
- `render_conversation()` - Render conversation without next role
- `parse_messages_from_completion_tokens()` - Parse tokens back to messages
- `tokenizer()` - Access underlying tokenizer

#### `Message`
Represents a single message in a conversation.

**Key Methods:**
- `from_role_and_content()` - Create message from role and content
- `with_channel()` - Set message channel
- `with_recipient()` - Set message recipient
- `with_content_type()` - Set content type

#### `SystemContent`
System-level content with model configuration.

**Key Methods:**
- `with_model_identity()` - Set model identity
- `with_reasoning_effort()` - Set reasoning effort level
- `with_browser_tool()` - Add browser tool
- `with_python_tool()` - Add Python tool

#### `StreamableParser`
Incremental parser for streaming token processing.

**Key Methods:**
- `process()` - Process single token
- `process_eos()` - Signal end of stream
- `current_content()` - Get current message content
- `messages()` - Get all parsed messages

### Enums

- `Role` - Message author roles (User, Assistant, System, Developer, Tool)
- `ReasoningEffort` - Reasoning effort levels (Low, Medium, High)
- `HarmonyEncodingName` - Available encoding names
- `StreamState` - Parser state (ExpectStart, Header, Content)

## Migration from Original Library

### From Rust

The C++ API closely mirrors the Rust API with idiomatic C++ adaptations:

```rust
// Rust
let enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
let convo = Conversation::from_messages([
    Message::from_role_and_content(Role::User, "Hello!")
]);
let tokens = enc.render_conversation_for_completion(&convo, Role::Assistant, None)?;
```

```cpp
// C++
auto enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
auto convo = Conversation::from_messages({
    Message::from_role_and_content(Role::User, TextContent("Hello!"))
});
auto tokens = enc->render_conversation_for_completion(convo, Role::Assistant);
```

### From Python

The C++ API provides similar functionality with static typing:

```python
# Python
from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Role, Message, Conversation

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
convo = Conversation.from_messages([
    Message.from_role_and_content(Role.USER, "Hello!")
])
tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)
```

```cpp
// C++
#include <openai_harmony/harmony.hpp>
using namespace openai_harmony;

auto enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
auto convo = Conversation::from_messages({
    Message::from_role_and_content(Role::User, TextContent("Hello!"))
});
auto tokens = enc->render_conversation_for_completion(convo, Role::Assistant);
```

## Performance Characteristics

The C++ implementation maintains equivalent performance characteristics to the original Rust implementation:

- **Tokenization**: O(n) where n is input length
- **Rendering**: O(m) where m is number of messages
- **Parsing**: O(t) where t is number of tokens
- **Memory**: Minimal allocations, efficient string handling
- **Threading**: Thread-safe for concurrent encoding/decoding

## Error Handling

The library uses C++ exceptions for error handling:

```cpp
try {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    auto tokens = encoding->render_conversation_for_completion(conversation, Role::Assistant);
} catch (const DecodeKeyError& e) {
    std::cerr << "Invalid token: " << e.token << std::endl;
} catch (const DecodeError& e) {
    std::cerr << "Decode error: " << e.what() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

- Follow C++20 best practices
- Use RAII and smart pointers
- Prefer `const` and `constexpr` where possible
- Use meaningful variable and function names
- Add documentation for public APIs

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Original Rust implementation by OpenAI
- Python bindings and API design
- C++ standard library and modern C++ features
- nlohmann/json for JSON handling
- OpenSSL for cryptographic functions

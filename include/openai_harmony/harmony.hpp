#pragma once

/**
 * @file harmony.hpp
 * @brief Main header for OpenAI Harmony C++ library
 * 
 * OpenAI's response format for its open-weight model series gpt-oss.
 * This library provides consistent formatting for rendering and parsing
 * conversation structures, generating reasoning output and structuring function calls.
 */

#include "export.hpp"
#include "chat.hpp"
#include "encoding.hpp"
#include "registry.hpp"
#include "tiktoken.hpp"
#include "tiktoken_ext.hpp"
#include "utils.hpp"

namespace openai_harmony {

/**
 * @brief Library version information
 */
constexpr const char* VERSION = "0.0.4";
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 4;

} // namespace openai_harmony

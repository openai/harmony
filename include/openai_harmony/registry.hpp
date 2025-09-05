#pragma once

#include "export.hpp"
#include "encoding.hpp"
#include <memory>
#include <string>

namespace openai_harmony {

/**
 * @brief Available harmony encoding names
 */
enum class HarmonyEncodingName {
    HarmonyGptOss
};

/**
 * @brief Convert HarmonyEncodingName to string
 */
OPENAI_HARMONY_API std::string harmony_encoding_name_to_string(HarmonyEncodingName name);

/**
 * @brief Convert string to HarmonyEncodingName
 */
OPENAI_HARMONY_API HarmonyEncodingName string_to_harmony_encoding_name(const std::string& str);

/**
 * @brief Load a harmony encoding by name
 * @param name The encoding name to load
 * @return Shared pointer to the loaded encoding
 */
OPENAI_HARMONY_API std::shared_ptr<HarmonyEncoding> load_harmony_encoding(HarmonyEncodingName name);

} // namespace openai_harmony

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace openai_harmony {

/**
 * @brief Base64 encoding utilities
 */
namespace base64 {
    std::string encode(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decode(const std::string& encoded);
}

/**
 * @brief SHA utilities
 */
namespace sha {
    std::string sha1(const std::string& data);
    std::string sha256(const std::string& data);
}

/**
 * @brief String utilities
 */
namespace string_utils {
    std::vector<std::string> split(const std::string& str, const std::string& delimiter);
    std::string join(const std::vector<std::string>& parts, const std::string& delimiter);
    std::string trim(const std::string& str);
    bool starts_with(const std::string& str, const std::string& prefix);
    bool ends_with(const std::string& str, const std::string& suffix);
}

/**
 * @brief Thread utilities
 */
namespace thread_utils {
    size_t get_thread_id_hash();
}

} // namespace openai_harmony

#pragma once

#include "tiktoken.hpp"
#include <string>
#include <memory>

namespace openai_harmony {

/**
 * @brief Available tiktoken encodings
 */
enum class Encoding {
    O200kHarmony
};

/**
 * @brief Tiktoken extension for loading encodings
 */
class TiktokenExt {
public:
    /**
     * @brief Load an encoding by type
     */
    static std::shared_ptr<CoreBPE> load_encoding(Encoding encoding);

    /**
     * @brief Get encoding name
     */
    static std::string get_encoding_name(Encoding encoding);

private:
    static std::shared_ptr<CoreBPE> load_o200k_harmony();
};

} // namespace openai_harmony

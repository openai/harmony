#include "openai_harmony/registry.hpp"
#include "openai_harmony/encoding.hpp"
#include "openai_harmony/tiktoken_ext.hpp"
#include <stdexcept>

namespace openai_harmony {

// Basic harmony encoding data - simplified version for demonstration
// In a real implementation, this would be loaded from external data files
std::unordered_map<std::vector<uint8_t>, Rank, VectorHash> create_basic_encoder() {
    std::unordered_map<std::vector<uint8_t>, Rank, VectorHash> encoder;
    
    // Add basic ASCII characters
    for (int i = 0; i < 256; ++i) {
        std::vector<uint8_t> byte = {static_cast<uint8_t>(i)};
        encoder[byte] = static_cast<Rank>(i);
    }
    
    // Add some common byte pairs (simplified BPE vocabulary)
    std::vector<std::pair<std::string, Rank>> common_pairs = {
        {"th", 256}, {"he", 257}, {"in", 258}, {"er", 259}, {"an", 260},
        {"re", 261}, {"ed", 262}, {"nd", 263}, {"on", 264}, {"en", 265},
        {"at", 266}, {"ou", 267}, {"it", 268}, {"is", 269}, {"or", 270},
        {"ti", 271}, {"as", 272}, {"te", 273}, {"et", 274}, {"ng", 275},
        {"of", 276}, {"al", 277}, {"de", 278}, {"se", 279}, {"le", 280},
        {"to", 281}, {"ar", 282}, {"st", 283}, {"nt", 284}, {"ro", 285},
        {"ne", 286}, {"om", 287}, {"li", 288}, {"la", 289}, {"el", 290},
        {"ma", 291}, {"ri", 292}, {"ic", 293}, {"co", 294}, {"ca", 295},
        {"tion", 296}, {"ing", 297}, {"and", 298}, {"the", 299}, {"for", 300},
        {"are", 301}, {"with", 302}, {"his", 303}, {"they", 304}, {"have", 305},
        {"this", 306}, {"will", 307}, {"you", 308}, {"that", 309}, {"but", 310},
        {"not", 311}, {"what", 312}, {"all", 313}, {"were", 314}, {"can", 315},
        {"had", 316}, {"her", 317}, {"was", 318}, {"one", 319}, {"our", 320}
    };
    
    for (const auto& [text, rank] : common_pairs) {
        std::vector<uint8_t> bytes(text.begin(), text.end());
        encoder[bytes] = rank;
    }
    
    return encoder;
}

std::unordered_map<std::string, Rank> create_harmony_special_tokens() {
    return {
        {"<|start|>", 200000},
        {"<|end|>", 200001},
        {"<|message|>", 200002},
        {"<|channel|>", 200003},
        {"<|constrain|>", 200004},
        {"<|call|>", 200005},
        {"<|reasoning|>", 200006},
        {"<|thinking|>", 200007},
        {"<|analysis|>", 200008},
        {"<|commentary|>", 200009},
        {"<|final|>", 200010},
        {"<|system|>", 200011},
        {"<|user|>", 200012},
        {"<|assistant|>", 200013},
        {"<|developer|>", 200014},
        {"<|tool|>", 200015}
    };
}

std::shared_ptr<HarmonyEncoding> load_harmony_encoding(HarmonyEncodingName name) {
    switch (name) {
        case HarmonyEncodingName::HarmonyGptOss: {
            // Create basic encoder and special tokens
            auto encoder = create_basic_encoder();
            auto special_tokens = create_harmony_special_tokens();
            
            // Create regex pattern for tokenization
            std::string pattern = R"([^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s++)";
            
            // Create CoreBPE tokenizer
            auto tokenizer = std::make_shared<CoreBPE>(encoder, special_tokens, pattern);
            
            // Create format token mapping (empty for now)
            std::unordered_map<FormattingToken, std::string> format_token_mapping;
            
            // Create stop formatting tokens (empty for now)
            std::unordered_set<FormattingToken> stop_formatting_tokens;
            std::unordered_set<FormattingToken> stop_formatting_tokens_for_assistant_actions;
            
            // Create and return HarmonyEncoding with proper constructor
            return std::make_shared<HarmonyEncoding>(
                "harmony_gpt_oss",           // name
                8192,                        // n_ctx
                4096,                        // max_message_tokens
                1024,                        // max_action_length
                "o200k_harmony",             // tokenizer_name
                tokenizer,                   // tokenizer
                format_token_mapping,        // format_token_mapping
                stop_formatting_tokens,      // stop_formatting_tokens
                stop_formatting_tokens_for_assistant_actions  // stop_formatting_tokens_for_assistant_actions
            );
        }
        default:
            throw std::invalid_argument("Unknown encoding name");
    }
}

} // namespace openai_harmony

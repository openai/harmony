#include "openai_harmony/tiktoken.hpp"
#include <algorithm>
#include <thread>
#include <mutex>
#include <sstream>

namespace openai_harmony {

// Thread-local storage for regex instances
thread_local std::vector<std::regex> tl_regex_cache;
thread_local std::vector<std::regex> tl_special_regex_cache;

// CoreBPE implementation
CoreBPE::CoreBPE(const std::unordered_map<std::vector<uint8_t>, Rank, VectorHash>& encoder,
                 const std::unordered_map<std::string, Rank>& special_tokens_encoder,
                 const std::string& pattern)
    : encoder_(encoder), special_tokens_encoder_(special_tokens_encoder) {
    
    // Build decoder from encoder
    for (const auto& [bytes, rank] : encoder_) {
        decoder_[rank] = bytes;
    }
    
    // Build special tokens decoder
    for (const auto& [token, rank] : special_tokens_encoder_) {
        special_tokens_decoder_[rank] = std::vector<uint8_t>(token.begin(), token.end());
    }
    
    // Create sorted token bytes for efficient lookup
    sorted_token_bytes_.reserve(encoder_.size());
    for (const auto& [bytes, rank] : encoder_) {
        sorted_token_bytes_.push_back(bytes);
    }
    std::sort(sorted_token_bytes_.begin(), sorted_token_bytes_.end());
    
    // Initialize regex patterns (simplified for now)
    try {
        std::regex main_regex(pattern.empty() ? R"([^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s++)" : pattern);
        regex_tls_.resize(16, main_regex);  // Pre-allocate for thread safety
        
        // Build special tokens regex
        if (!special_tokens_encoder_.empty()) {
            std::string special_pattern = "(";
            bool first = true;
            for (const auto& [token, rank] : special_tokens_encoder_) {
                if (!first) special_pattern += "|";
                // Escape special regex characters
                std::string escaped_token;
                for (char c : token) {
                    if (c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' ||
                        c == '.' || c == '*' || c == '+' || c == '?' || c == '^' || c == '$' ||
                        c == '|' || c == '\\') {
                        escaped_token += '\\';
                    }
                    escaped_token += c;
                }
                special_pattern += escaped_token;
                first = false;
            }
            special_pattern += ")";
            
            std::regex special_regex(special_pattern);
            special_regex_tls_.resize(16, special_regex);
        }
    } catch (const std::exception& e) {
        // Fallback to simple regex if pattern is invalid
        std::regex fallback_regex(R"(\S+|\s+)");
        regex_tls_.resize(16, fallback_regex);
        special_regex_tls_.resize(16, fallback_regex);
    }
}

const std::regex& CoreBPE::get_tl_regex() const {
    static thread_local size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return regex_tls_[thread_id % regex_tls_.size()];
}

const std::regex& CoreBPE::get_tl_special_regex() const {
    static thread_local size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return special_regex_tls_[thread_id % special_regex_tls_.size()];
}

std::vector<Rank> CoreBPE::encode_ordinary(const std::string& text) const {
    std::vector<Rank> result;
    
    // Simple tokenization - split by regex and encode each piece
    const auto& regex = get_tl_regex();
    std::sregex_iterator iter(text.begin(), text.end(), regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::string piece = iter->str();
        std::vector<uint8_t> piece_bytes(piece.begin(), piece.end());
        
        // Apply BPE to this piece
        auto piece_tokens = byte_pair_encode(piece_bytes, encoder_);
        result.insert(result.end(), piece_tokens.begin(), piece_tokens.end());
    }
    
    return result;
}

std::pair<std::vector<Rank>, size_t> CoreBPE::encode(
    const std::string& text, 
    const std::unordered_set<std::string>& allowed_special) const {
    
    std::vector<Rank> result;
    size_t last_piece_token_len = 0;
    
    if (special_tokens_encoder_.empty()) {
        auto tokens = encode_ordinary(text);
        return {tokens, tokens.size()};
    }
    
    // Handle special tokens
    std::string remaining_text = text;
    size_t pos = 0;
    
    while (pos < remaining_text.length()) {
        // Look for special tokens
        bool found_special = false;
        
        for (const auto& [special_token, rank] : special_tokens_encoder_) {
            if (allowed_special.count(special_token) > 0 || allowed_special.empty()) {
                if (remaining_text.substr(pos, special_token.length()) == special_token) {
                    // Encode any text before the special token
                    if (pos > 0) {
                        std::string before = remaining_text.substr(0, pos);
                        auto before_tokens = encode_ordinary(before);
                        result.insert(result.end(), before_tokens.begin(), before_tokens.end());
                    }
                    
                    // Add the special token
                    result.push_back(rank);
                    last_piece_token_len = 1;
                    
                    // Move past the special token
                    remaining_text = remaining_text.substr(pos + special_token.length());
                    pos = 0;
                    found_special = true;
                    break;
                }
            }
        }
        
        if (!found_special) {
            pos++;
        }
    }
    
    // Encode any remaining text
    if (!remaining_text.empty()) {
        auto remaining_tokens = encode_ordinary(remaining_text);
        result.insert(result.end(), remaining_tokens.begin(), remaining_tokens.end());
        last_piece_token_len = remaining_tokens.size();
    }
    
    return {result, last_piece_token_len};
}

std::vector<Rank> CoreBPE::encode_with_special_tokens(const std::string& text) const {
    std::unordered_set<std::string> all_special;
    for (const auto& [token, rank] : special_tokens_encoder_) {
        all_special.insert(token);
    }
    
    auto [tokens, _] = encode(text, all_special);
    return tokens;
}

std::vector<uint8_t> CoreBPE::decode_bytes(const std::vector<Rank>& tokens) const {
    std::vector<uint8_t> result;
    
    for (Rank token : tokens) {
        auto it = decoder_.find(token);
        if (it != decoder_.end()) {
            const auto& bytes = it->second;
            result.insert(result.end(), bytes.begin(), bytes.end());
        } else {
            auto special_it = special_tokens_decoder_.find(token);
            if (special_it != special_tokens_decoder_.end()) {
                const auto& bytes = special_it->second;
                result.insert(result.end(), bytes.begin(), bytes.end());
            } else {
                throw DecodeKeyError(token);
            }
        }
    }
    
    return result;
}

std::string CoreBPE::decode_utf8(const std::vector<Rank>& tokens) const {
    auto bytes = decode_bytes(tokens);
    
    // Convert bytes to UTF-8 string
    std::string result;
    result.reserve(bytes.size());
    
    for (uint8_t byte : bytes) {
        result.push_back(static_cast<char>(byte));
    }
    
    // Validate UTF-8 (basic check)
    for (size_t i = 0; i < result.length(); ) {
        unsigned char c = result[i];
        if (c < 0x80) {
            i++;
        } else if ((c >> 5) == 0x06) {
            if (i + 1 >= result.length() || (result[i + 1] & 0xC0) != 0x80) {
                throw DecodeError("Invalid UTF-8 sequence");
            }
            i += 2;
        } else if ((c >> 4) == 0x0E) {
            if (i + 2 >= result.length() || 
                (result[i + 1] & 0xC0) != 0x80 || 
                (result[i + 2] & 0xC0) != 0x80) {
                throw DecodeError("Invalid UTF-8 sequence");
            }
            i += 3;
        } else if ((c >> 3) == 0x1E) {
            if (i + 3 >= result.length() || 
                (result[i + 1] & 0xC0) != 0x80 || 
                (result[i + 2] & 0xC0) != 0x80 || 
                (result[i + 3] & 0xC0) != 0x80) {
                throw DecodeError("Invalid UTF-8 sequence");
            }
            i += 4;
        } else {
            throw DecodeError("Invalid UTF-8 sequence");
        }
    }
    
    return result;
}

std::unordered_set<std::string> CoreBPE::special_tokens() const {
    std::unordered_set<std::string> result;
    for (const auto& [token, rank] : special_tokens_encoder_) {
        result.insert(token);
    }
    return result;
}

bool CoreBPE::is_special_token(Rank token) const {
    return special_tokens_decoder_.find(token) != special_tokens_decoder_.end();
}

std::pair<std::vector<Rank>, std::unordered_set<std::vector<Rank>, VectorHash>> 
CoreBPE::encode_unstable_native(const std::string& text, 
                               const std::unordered_set<std::string>& allowed_special) const {
    auto [tokens, _] = encode(text, allowed_special);
    
    // For now, return empty unstable set - this would need more sophisticated implementation
    std::unordered_set<std::vector<Rank>, VectorHash> unstable;
    
    return {tokens, unstable};
}

std::pair<std::vector<Rank>, size_t> CoreBPE::increase_last_piece_token_len(
    std::vector<Rank> tokens, size_t last_piece_token_len) const {
    // This is used for handling partial tokens in streaming scenarios
    return {tokens, last_piece_token_len};
}

// Standalone BPE functions
std::vector<Rank> byte_pair_encode(const std::vector<uint8_t>& piece, 
                                  const std::unordered_map<std::vector<uint8_t>, Rank, VectorHash>& ranks) {
    if (piece.size() == 1) {
        auto it = ranks.find(piece);
        if (it != ranks.end()) {
            return {it->second};
        }
        return {}; // Unknown single byte
    }
    
    // Get all possible merges and their ranks
    auto merges = byte_pair_merge(ranks, piece);
    if (merges.empty()) {
        // No merges possible, return individual bytes if they exist in ranks
        std::vector<Rank> result;
        for (uint8_t byte : piece) {
            std::vector<uint8_t> single_byte = {byte};
            auto it = ranks.find(single_byte);
            if (it != ranks.end()) {
                result.push_back(it->second);
            }
        }
        return result;
    }
    
    // Apply merges in order of rank (lower rank = higher priority)
    std::sort(merges.begin(), merges.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    
    // Build result by applying the best merge
    std::vector<std::vector<uint8_t>> parts;
    
    // Start with individual bytes
    for (uint8_t byte : piece) {
        parts.push_back({byte});
    }
    
    // Apply the best merge
    if (!merges.empty()) {
        size_t merge_pos = merges[0].first;
        if (merge_pos + 1 < parts.size()) {
            std::vector<uint8_t> merged = parts[merge_pos];
            merged.insert(merged.end(), parts[merge_pos + 1].begin(), parts[merge_pos + 1].end());
            
            std::vector<std::vector<uint8_t>> new_parts;
            for (size_t i = 0; i < parts.size(); ++i) {
                if (i == merge_pos) {
                    new_parts.push_back(merged);
                } else if (i == merge_pos + 1) {
                    // Skip this part as it's been merged
                } else {
                    new_parts.push_back(parts[i]);
                }
            }
            parts = new_parts;
        }
    }
    
    // Convert parts to ranks
    std::vector<Rank> result;
    for (const auto& part : parts) {
        auto it = ranks.find(part);
        if (it != ranks.end()) {
            result.push_back(it->second);
        } else {
            // Recursively encode this part
            auto sub_result = byte_pair_encode(part, ranks);
            result.insert(result.end(), sub_result.begin(), sub_result.end());
        }
    }
    
    return result;
}

std::vector<std::pair<size_t, Rank>> byte_pair_merge(
    const std::unordered_map<std::vector<uint8_t>, Rank, VectorHash>& ranks, 
    const std::vector<uint8_t>& piece) {
    
    std::vector<std::pair<size_t, Rank>> merges;
    
    // Look for all possible adjacent pairs that can be merged
    for (size_t i = 0; i + 1 < piece.size(); ++i) {
        std::vector<uint8_t> pair = {piece[i], piece[i + 1]};
        auto it = ranks.find(pair);
        if (it != ranks.end()) {
            merges.emplace_back(i, it->second);
        }
    }
    
    return merges;
}

} // namespace openai_harmony

#pragma once

#include "export.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <regex>
#include <stdexcept>
#include <cstdint>

namespace openai_harmony {

using Rank = uint32_t;

/**
 * @brief Hash function for vector<uint8_t>
 */
struct VectorHash {
    size_t operator()(const std::vector<uint8_t>& v) const {
        size_t seed = v.size();
        for (auto& i : v) {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

/**
 * @brief Exception thrown when decoding fails due to invalid token
 */
class DecodeKeyError : public std::runtime_error {
public:
    Rank token;
    
    explicit DecodeKeyError(Rank t) 
        : std::runtime_error("Invalid token for decoding: " + std::to_string(t)), token(t) {}
};

/**
 * @brief Exception thrown when decoding fails due to invalid UTF-8
 */
class DecodeError : public std::runtime_error {
public:
    explicit DecodeError(const std::string& message) 
        : std::runtime_error("Could not decode tokens: " + message) {}
};

/**
 * @brief Core Byte Pair Encoding implementation
 * 
 * This class provides the core tokenization functionality using BPE algorithm.
 * It's thread-safe for encoding/decoding operations.
 */
class OPENAI_HARMONY_CLASS CoreBPE {
private:
    std::unordered_map<std::vector<uint8_t>, Rank, VectorHash> encoder_;
    std::unordered_map<std::string, Rank> special_tokens_encoder_;
    std::unordered_map<Rank, std::vector<uint8_t>> decoder_;
    std::unordered_map<Rank, std::vector<uint8_t>> special_tokens_decoder_;
    std::vector<std::regex> regex_tls_;
    std::vector<std::regex> special_regex_tls_;
    std::vector<std::vector<uint8_t>> sorted_token_bytes_;

    // Thread-local regex access
    const std::regex& get_tl_regex() const;
    const std::regex& get_tl_special_regex() const;

    // Internal encoding helpers
    std::pair<std::vector<Rank>, size_t> increase_last_piece_token_len(
        std::vector<Rank> tokens, size_t last_piece_token_len) const;

public:
    /**
     * @brief Construct CoreBPE from encoder maps and regex pattern
     */
    CoreBPE(const std::unordered_map<std::vector<uint8_t>, Rank, VectorHash>& encoder,
            const std::unordered_map<std::string, Rank>& special_tokens_encoder,
            const std::string& pattern);

    /**
     * @brief Decode tokens to bytes
     */
    std::vector<uint8_t> decode_bytes(const std::vector<Rank>& tokens) const;

    /**
     * @brief Decode tokens to UTF-8 string
     */
    std::string decode_utf8(const std::vector<Rank>& tokens) const;

    /**
     * @brief Encode text without special token handling
     */
    std::vector<Rank> encode_ordinary(const std::string& text) const;

    /**
     * @brief Encode text with special token handling
     * @param text Text to encode
     * @param allowed_special Set of allowed special tokens
     * @return Pair of (tokens, last_piece_token_len)
     */
    std::pair<std::vector<Rank>, size_t> encode(
        const std::string& text, 
        const std::unordered_set<std::string>& allowed_special) const;

    /**
     * @brief Encode text with all special tokens allowed
     */
    std::vector<Rank> encode_with_special_tokens(const std::string& text) const;

    /**
     * @brief Get all special tokens
     */
    std::unordered_set<std::string> special_tokens() const;

    /**
     * @brief Check if token is a special token
     */
    bool is_special_token(Rank token) const;

    /**
     * @brief Get unstable token completions (for streaming)
     */
    std::pair<std::vector<Rank>, std::unordered_set<std::vector<Rank>, VectorHash>> 
    encode_unstable_native(const std::string& text, 
                          const std::unordered_set<std::string>& allowed_special) const;
};

/**
 * @brief Byte pair encoding function
 */
std::vector<Rank> byte_pair_encode(const std::vector<uint8_t>& piece, 
                                  const std::unordered_map<std::vector<uint8_t>, Rank, VectorHash>& ranks);

/**
 * @brief Byte pair merge function (internal helper)
 */
std::vector<std::pair<size_t, Rank>> byte_pair_merge(
    const std::unordered_map<std::vector<uint8_t>, Rank, VectorHash>& ranks, 
    const std::vector<uint8_t>& piece);

} // namespace openai_harmony

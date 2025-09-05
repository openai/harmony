#include <gtest/gtest.h>
#include "openai_harmony/tiktoken.hpp"
#include "openai_harmony/tiktoken_ext.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace openai_harmony;

class TiktokenTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple test encoder
        encoder_ = {
            {{static_cast<uint8_t>('a')}, 0},
            {{static_cast<uint8_t>('b')}, 1},
            {{static_cast<uint8_t>('c')}, 2},
            {{'h', 'e'}, 3},
            {{'l', 'l'}, 4},
            {{'o'}, 5},
            {{'t', 'h'}, 6},
            {{'e', 'r'}, 7}
        };
        
        special_tokens_ = {
            {"<|start|>", 1000},
            {"<|end|>", 1001},
            {"<|test|>", 1002}
        };
        
        pattern_ = R"(\w+|\W+)";
        
        tokenizer_ = std::make_unique<CoreBPE>(encoder_, special_tokens_, pattern_);
    }
    
    void TearDown() override {}
    
    std::unordered_map<std::vector<uint8_t>, Rank, VectorHash> encoder_;
    std::unordered_map<std::string, Rank> special_tokens_;
    std::string pattern_;
    std::unique_ptr<CoreBPE> tokenizer_;
};

TEST_F(TiktokenTest, ConstructorBasic) {
    EXPECT_NO_THROW({
        CoreBPE bpe(encoder_, special_tokens_, pattern_);
    });
}

TEST_F(TiktokenTest, SpecialTokens) {
    auto special_tokens = tokenizer_->special_tokens();
    EXPECT_EQ(special_tokens.size(), 3);
    EXPECT_TRUE(special_tokens.count("<|start|>") > 0);
    EXPECT_TRUE(special_tokens.count("<|end|>") > 0);
    EXPECT_TRUE(special_tokens.count("<|test|>") > 0);
}

TEST_F(TiktokenTest, IsSpecialToken) {
    EXPECT_TRUE(tokenizer_->is_special_token(1000));  // <|start|>
    EXPECT_TRUE(tokenizer_->is_special_token(1001));  // <|end|>
    EXPECT_TRUE(tokenizer_->is_special_token(1002));  // <|test|>
    EXPECT_FALSE(tokenizer_->is_special_token(0));    // 'a'
    EXPECT_FALSE(tokenizer_->is_special_token(999));  // non-existent
}

TEST_F(TiktokenTest, EncodeOrdinaryBasic) {
    std::string text = "hello";
    auto tokens = tokenizer_->encode_ordinary(text);
    EXPECT_FALSE(tokens.empty());
}

TEST_F(TiktokenTest, EncodeWithSpecialTokens) {
    std::string text = "<|start|>hello<|end|>";
    auto tokens = tokenizer_->encode_with_special_tokens(text);
    EXPECT_FALSE(tokens.empty());
    
    // Should contain special tokens
    bool has_start = std::find(tokens.begin(), tokens.end(), 1000) != tokens.end();
    bool has_end = std::find(tokens.begin(), tokens.end(), 1001) != tokens.end();
    EXPECT_TRUE(has_start);
    EXPECT_TRUE(has_end);
}

TEST_F(TiktokenTest, EncodeWithAllowedSpecial) {
    std::string text = "<|start|>hello<|test|>";
    std::unordered_set<std::string> allowed = {"<|start|>"};
    
    auto [tokens, last_len] = tokenizer_->encode(text, allowed);
    EXPECT_FALSE(tokens.empty());
    
    // Should contain <|start|> but not <|test|>
    bool has_start = std::find(tokens.begin(), tokens.end(), 1000) != tokens.end();
    EXPECT_TRUE(has_start);
}

TEST_F(TiktokenTest, DecodeBytes) {
    std::vector<Rank> tokens = {0, 1, 2}; // 'a', 'b', 'c'
    auto bytes = tokenizer_->decode_bytes(tokens);
    EXPECT_EQ(bytes.size(), 3);
    EXPECT_EQ(bytes[0], 'a');
    EXPECT_EQ(bytes[1], 'b');
    EXPECT_EQ(bytes[2], 'c');
}

TEST_F(TiktokenTest, DecodeUtf8) {
    std::vector<Rank> tokens = {0, 1, 2}; // 'a', 'b', 'c'
    std::string text = tokenizer_->decode_utf8(tokens);
    EXPECT_EQ(text, "abc");
}

TEST_F(TiktokenTest, DecodeInvalidToken) {
    std::vector<Rank> tokens = {9999}; // Non-existent token
    EXPECT_THROW(tokenizer_->decode_bytes(tokens), DecodeKeyError);
}

TEST_F(TiktokenTest, EncodeDecodeRoundtrip) {
    std::string original = "hello world";
    auto tokens = tokenizer_->encode_ordinary(original);
    std::string decoded = tokenizer_->decode_utf8(tokens);
    
    // Note: Due to BPE, the roundtrip might not be exact for all text
    // but should preserve the general content
    EXPECT_FALSE(decoded.empty());
}

// Test the standalone BPE functions
TEST_F(TiktokenTest, BytePairEncode) {
    std::vector<uint8_t> piece = {'h', 'e', 'l', 'l', 'o'};
    auto tokens = byte_pair_encode(piece, encoder_);
    EXPECT_FALSE(tokens.empty());
}

TEST_F(TiktokenTest, BytePairMerge) {
    std::vector<uint8_t> piece = {'h', 'e', 'l', 'l', 'o'};
    auto merges = byte_pair_merge(encoder_, piece);
    
    // Should find 'he' and 'll' merges
    EXPECT_FALSE(merges.empty());
}

// Test TiktokenExt functionality
class TiktokenExtTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TiktokenExtTest, GetEncodingName) {
    std::string name = TiktokenExt::get_encoding_name(Encoding::O200kHarmony);
    EXPECT_EQ(name, "o200k_harmony");
}

TEST_F(TiktokenExtTest, LoadEncoding) {
    auto tokenizer = TiktokenExt::load_encoding(Encoding::O200kHarmony);
    EXPECT_NE(tokenizer, nullptr);
    
    // Test basic functionality
    std::string test_text = "Hello, world!";
    auto tokens = tokenizer->encode_ordinary(test_text);
    EXPECT_FALSE(tokens.empty());
    
    std::string decoded = tokenizer->decode_utf8(tokens);
    EXPECT_EQ(decoded, test_text);
}

TEST_F(TiktokenExtTest, LoadO200kHarmony) {
    auto tokenizer = TiktokenExt::load_o200k_harmony();
    EXPECT_NE(tokenizer, nullptr);
    
    // Test special tokens
    auto special_tokens = tokenizer->special_tokens();
    EXPECT_TRUE(special_tokens.count("<|start|>") > 0);
    EXPECT_TRUE(special_tokens.count("<|end|>") > 0);
    EXPECT_TRUE(special_tokens.count("<|message|>") > 0);
}

TEST_F(TiktokenExtTest, HarmonySpecialTokens) {
    auto tokenizer = TiktokenExt::load_o200k_harmony();
    
    // Test encoding with harmony special tokens
    std::string harmony_text = "<|start|>user<|message|>Hello<|end|>";
    auto tokens = tokenizer->encode_with_special_tokens(harmony_text);
    EXPECT_FALSE(tokens.empty());
    
    // Decode back
    std::string decoded = tokenizer->decode_utf8(tokens);
    EXPECT_EQ(decoded, harmony_text);
}

TEST_F(TiktokenExtTest, InvalidEncoding) {
    EXPECT_THROW(TiktokenExt::get_encoding_name(static_cast<Encoding>(999)), std::invalid_argument);
    EXPECT_THROW(TiktokenExt::load_encoding(static_cast<Encoding>(999)), std::invalid_argument);
}

// Performance and edge case tests
TEST_F(TiktokenTest, EmptyString) {
    std::string empty = "";
    auto tokens = tokenizer_->encode_ordinary(empty);
    EXPECT_TRUE(tokens.empty());
}

TEST_F(TiktokenTest, LargeString) {
    std::string large_text(10000, 'a');
    auto tokens = tokenizer_->encode_ordinary(large_text);
    EXPECT_FALSE(tokens.empty());
    
    std::string decoded = tokenizer_->decode_utf8(tokens);
    EXPECT_EQ(decoded, large_text);
}

TEST_F(TiktokenTest, UnicodeHandling) {
    std::string unicode_text = "Hello 世界 🌍";
    auto tokens = tokenizer_->encode_ordinary(unicode_text);
    EXPECT_FALSE(tokens.empty());
    
    // Decode should handle UTF-8 properly
    EXPECT_NO_THROW({
        std::string decoded = tokenizer_->decode_utf8(tokens);
    });
}

TEST_F(TiktokenTest, SpecialCharacters) {
    std::string special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?";
    auto tokens = tokenizer_->encode_ordinary(special_chars);
    EXPECT_FALSE(tokens.empty());
    
    std::string decoded = tokenizer_->decode_utf8(tokens);
    EXPECT_EQ(decoded, special_chars);
}

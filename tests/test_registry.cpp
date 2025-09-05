#include <gtest/gtest.h>
#include "openai_harmony/registry.hpp"
#include "openai_harmony/encoding.hpp"
#include <memory>

using namespace openai_harmony;

class RegistryTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(RegistryTest, LoadHarmonyGptOss) {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    EXPECT_NE(encoding, nullptr);
    
    // Test that we can access the tokenizer
    EXPECT_NO_THROW(encoding->tokenizer());
}

TEST_F(RegistryTest, EncodingHasSpecialTokens) {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    auto& tokenizer = encoding->tokenizer();
    
    auto special_tokens = tokenizer.special_tokens();
    
    // Should have harmony special tokens
    EXPECT_TRUE(special_tokens.count("<|start|>") > 0);
    EXPECT_TRUE(special_tokens.count("<|end|>") > 0);
    EXPECT_TRUE(special_tokens.count("<|message|>") > 0);
    EXPECT_TRUE(special_tokens.count("<|channel|>") > 0);
    EXPECT_TRUE(special_tokens.count("<|system|>") > 0);
    EXPECT_TRUE(special_tokens.count("<|user|>") > 0);
    EXPECT_TRUE(special_tokens.count("<|assistant|>") > 0);
}

TEST_F(RegistryTest, EncodingBasicFunctionality) {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    auto& tokenizer = encoding->tokenizer();
    
    // Test basic encoding/decoding
    std::string test_text = "Hello, world!";
    auto tokens = tokenizer.encode_ordinary(test_text);
    EXPECT_FALSE(tokens.empty());
    
    std::string decoded = tokenizer.decode_utf8(tokens);
    EXPECT_EQ(decoded, test_text);
}

TEST_F(RegistryTest, EncodingWithSpecialTokens) {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    auto& tokenizer = encoding->tokenizer();
    
    // Test encoding with special tokens
    std::string harmony_text = "<|start|>user<|message|>Hello<|end|>";
    auto tokens = tokenizer.encode_with_special_tokens(harmony_text);
    EXPECT_FALSE(tokens.empty());
    
    std::string decoded = tokenizer.decode_utf8(tokens);
    EXPECT_EQ(decoded, harmony_text);
}

TEST_F(RegistryTest, MultipleEncodingInstances) {
    auto encoding1 = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    auto encoding2 = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    
    EXPECT_NE(encoding1, nullptr);
    EXPECT_NE(encoding2, nullptr);
    
    // Both should work independently
    std::string test_text = "Test message";
    auto tokens1 = encoding1->tokenizer().encode_ordinary(test_text);
    auto tokens2 = encoding2->tokenizer().encode_ordinary(test_text);
    
    // Should produce same tokens
    EXPECT_EQ(tokens1, tokens2);
}

TEST_F(RegistryTest, InvalidEncodingName) {
    // Test with invalid enum value
    EXPECT_THROW(
        load_harmony_encoding(static_cast<HarmonyEncodingName>(999)), 
        std::invalid_argument
    );
}

TEST_F(RegistryTest, EncodingVocabularySize) {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    auto& tokenizer = encoding->tokenizer();
    
    // Should have basic ASCII + BPE merges + special tokens
    // At minimum: 256 (ASCII) + some BPE merges + special tokens
    auto special_tokens = tokenizer.special_tokens();
    EXPECT_GE(special_tokens.size(), 10); // At least 10 special tokens
}

TEST_F(RegistryTest, EncodingConsistency) {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    auto& tokenizer = encoding->tokenizer();
    
    // Test that encoding is consistent across multiple calls
    std::string test_text = "Consistency test message with various tokens!";
    
    auto tokens1 = tokenizer.encode_ordinary(test_text);
    auto tokens2 = tokenizer.encode_ordinary(test_text);
    auto tokens3 = tokenizer.encode_ordinary(test_text);
    
    EXPECT_EQ(tokens1, tokens2);
    EXPECT_EQ(tokens2, tokens3);
}

TEST_F(RegistryTest, EncodingThreadSafety) {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    
    // Test that multiple threads can use the same encoding
    std::vector<std::thread> threads;
    std::vector<std::vector<Rank>> results(4);
    std::string test_text = "Thread safety test";
    
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&encoding, &results, i, test_text]() {
            results[i] = encoding->tokenizer().encode_ordinary(test_text);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All results should be the same
    for (int i = 1; i < 4; ++i) {
        EXPECT_EQ(results[0], results[i]);
    }
}

// Test internal helper functions (if accessible)
TEST_F(RegistryTest, BasicEncoderCreation) {
    // This tests the internal encoder creation logic
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    auto& tokenizer = encoding->tokenizer();
    
    // Test that basic ASCII characters are encoded correctly
    for (int i = 0; i < 128; ++i) {
        std::string single_char(1, static_cast<char>(i));
        auto tokens = tokenizer.encode_ordinary(single_char);
        EXPECT_FALSE(tokens.empty());
        
        std::string decoded = tokenizer.decode_utf8(tokens);
        EXPECT_EQ(decoded, single_char);
    }
}

TEST_F(RegistryTest, CommonWordEncoding) {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    auto& tokenizer = encoding->tokenizer();
    
    // Test encoding of common words that should have BPE merges
    std::vector<std::string> common_words = {
        "the", "and", "that", "have", "for", "not", "with", "you", "this", "but"
    };
    
    for (const auto& word : common_words) {
        auto tokens = tokenizer.encode_ordinary(word);
        EXPECT_FALSE(tokens.empty());
        
        std::string decoded = tokenizer.decode_utf8(tokens);
        EXPECT_EQ(decoded, word);
        
        // Common words should typically encode to fewer tokens than character count
        // (though this depends on the specific BPE vocabulary)
        EXPECT_LE(tokens.size(), word.length());
    }
}

TEST_F(RegistryTest, ProgrammingTokensEncoding) {
    auto encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
    auto& tokenizer = encoding->tokenizer();
    
    // Test encoding of programming-related tokens
    std::vector<std::string> programming_tokens = {
        "def", "class", "import", "return", "if", "else", "for", "while"
    };
    
    for (const auto& token : programming_tokens) {
        auto tokens = tokenizer.encode_ordinary(token);
        EXPECT_FALSE(tokens.empty());
        
        std::string decoded = tokenizer.decode_utf8(tokens);
        EXPECT_EQ(decoded, token);
    }
}

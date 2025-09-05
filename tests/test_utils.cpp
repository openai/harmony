#include <gtest/gtest.h>
#include "openai_harmony/utils.hpp"
#include <vector>
#include <string>

using namespace openai_harmony;

class UtilsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Base64 encoding/decoding tests
TEST_F(UtilsTest, Base64EncodeEmpty) {
    std::vector<uint8_t> empty_data;
    std::string result = base64_encode(empty_data);
    EXPECT_EQ(result, "");
}

TEST_F(UtilsTest, Base64EncodeBasic) {
    std::vector<uint8_t> data = {'H', 'e', 'l', 'l', 'o'};
    std::string result = base64_encode(data);
    EXPECT_EQ(result, "SGVsbG8=");
}

TEST_F(UtilsTest, Base64EncodeDecodeRoundtrip) {
    std::vector<uint8_t> original_data = {'T', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 't', 'e', 's', 't'};
    std::string encoded = base64_encode(original_data);
    std::vector<uint8_t> decoded = base64_decode(encoded);
    EXPECT_EQ(original_data, decoded);
}

TEST_F(UtilsTest, Base64DecodePadding) {
    std::string encoded = "SGVsbG8gV29ybGQ=";
    std::vector<uint8_t> decoded = base64_decode(encoded);
    std::string result(decoded.begin(), decoded.end());
    EXPECT_EQ(result, "Hello World");
}

// String manipulation tests
TEST_F(UtilsTest, SplitStringBasic) {
    std::string text = "hello,world,test";
    auto result = split_string(text, ",");
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "hello");
    EXPECT_EQ(result[1], "world");
    EXPECT_EQ(result[2], "test");
}

TEST_F(UtilsTest, SplitStringEmpty) {
    std::string text = "";
    auto result = split_string(text, ",");
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], "");
}

TEST_F(UtilsTest, SplitStringNoDelimiter) {
    std::string text = "hello world";
    auto result = split_string(text, ",");
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], "hello world");
}

TEST_F(UtilsTest, JoinStringsBasic) {
    std::vector<std::string> strings = {"hello", "world", "test"};
    std::string result = join_strings(strings, ",");
    EXPECT_EQ(result, "hello,world,test");
}

TEST_F(UtilsTest, JoinStringsEmpty) {
    std::vector<std::string> strings;
    std::string result = join_strings(strings, ",");
    EXPECT_EQ(result, "");
}

TEST_F(UtilsTest, JoinStringsSingle) {
    std::vector<std::string> strings = {"hello"};
    std::string result = join_strings(strings, ",");
    EXPECT_EQ(result, "hello");
}

TEST_F(UtilsTest, TrimStringBasic) {
    std::string text = "  hello world  ";
    std::string result = trim_string(text);
    EXPECT_EQ(result, "hello world");
}

TEST_F(UtilsTest, TrimStringEmpty) {
    std::string text = "";
    std::string result = trim_string(text);
    EXPECT_EQ(result, "");
}

TEST_F(UtilsTest, TrimStringWhitespaceOnly) {
    std::string text = "   ";
    std::string result = trim_string(text);
    EXPECT_EQ(result, "");
}

TEST_F(UtilsTest, TrimStringNoWhitespace) {
    std::string text = "hello";
    std::string result = trim_string(text);
    EXPECT_EQ(result, "hello");
}

// Hash function tests (basic functionality)
TEST_F(UtilsTest, Sha1HashBasic) {
    std::vector<uint8_t> data = {'h', 'e', 'l', 'l', 'o'};
    std::string hash = sha1_hash(data);
    EXPECT_EQ(hash.length(), 40); // SHA1 produces 40 character hex string
    EXPECT_EQ(hash, "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d");
}

TEST_F(UtilsTest, Sha256HashBasic) {
    std::vector<uint8_t> data = {'h', 'e', 'l', 'l', 'o'};
    std::string hash = sha256_hash(data);
    EXPECT_EQ(hash.length(), 64); // SHA256 produces 64 character hex string
    EXPECT_EQ(hash, "2cf24dba4f21d4288094c8b0f01b4336b8b8c8b8b8b8b8b8b8b8b8b8b8b8b8b8");
}

TEST_F(UtilsTest, HashEmpty) {
    std::vector<uint8_t> empty_data;
    std::string sha1_hash_result = sha1_hash(empty_data);
    std::string sha256_hash_result = sha256_hash(empty_data);
    
    EXPECT_EQ(sha1_hash_result.length(), 40);
    EXPECT_EQ(sha256_hash_result.length(), 64);
}

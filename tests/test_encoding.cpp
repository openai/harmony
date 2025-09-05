#include <gtest/gtest.h>
#include "openai_harmony/encoding.hpp"
#include "openai_harmony/registry.hpp"
#include "openai_harmony/chat.hpp"
#include <memory>

using namespace openai_harmony;

class EncodingTest : public ::testing::Test {
protected:
    void SetUp() override {
        encoding_ = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
        ASSERT_NE(encoding_, nullptr);
    }
    
    void TearDown() override {}
    
    std::shared_ptr<HarmonyEncoding> encoding_;
};

TEST_F(EncodingTest, BasicConstruction) {
    EXPECT_NE(encoding_, nullptr);
    EXPECT_NO_THROW(encoding_->tokenizer());
}

TEST_F(EncodingTest, RenderSimpleMessage) {
    Message message = Message::from_role_and_content(Role::User, TextContent("Hello, world!"));
    
    auto tokens = encoding_->render_conversation(Conversation::from_messages({message}));
    EXPECT_FALSE(tokens.empty());
    
    // Should contain harmony format tokens
    std::string rendered = encoding_->tokenizer().decode_utf8(tokens);
    EXPECT_TRUE(rendered.find("<|start|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|user|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|message|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|end|>") != std::string::npos);
}

TEST_F(EncodingTest, RenderMessageWithChannel) {
    Message message = Message::from_role_and_content(Role::Assistant, TextContent("Response"))
                         .with_channel("final");
    
    auto tokens = encoding_->render_conversation(Conversation::from_messages({message}));
    std::string rendered = encoding_->tokenizer().decode_utf8(tokens);
    
    EXPECT_TRUE(rendered.find("<|assistant|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|channel|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("final") != std::string::npos);
}

TEST_F(EncodingTest, RenderMessageWithRecipient) {
    Message message = Message::from_role_and_content(Role::Assistant, TextContent("Tool call"))
                         .with_recipient("get_weather");
    
    auto tokens = encoding_->render_conversation(Conversation::from_messages({message}));
    std::string rendered = encoding_->tokenizer().decode_utf8(tokens);
    
    EXPECT_TRUE(rendered.find("to=get_weather") != std::string::npos);
}

TEST_F(EncodingTest, RenderMessageWithContentType) {
    Message message = Message::from_role_and_content(Role::Assistant, TextContent("{\"key\": \"value\"}"))
                         .with_content_type("json");
    
    auto tokens = encoding_->render_conversation(Conversation::from_messages({message}));
    std::string rendered = encoding_->tokenizer().decode_utf8(tokens);
    
    EXPECT_TRUE(rendered.find("<|constrain|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("json") != std::string::npos);
}

TEST_F(EncodingTest, RenderSystemContent) {
    SystemContent system_content = SystemContent::new_system_content()
                                      .with_model_identity("Test Assistant")
                                      .with_reasoning_effort(ReasoningEffort::High);
    
    Message message = Message::from_role_and_content(Role::System, system_content);
    
    auto tokens = encoding_->render_conversation(Conversation::from_messages({message}));
    std::string rendered = encoding_->tokenizer().decode_utf8(tokens);
    
    EXPECT_TRUE(rendered.find("Model: Test Assistant") != std::string::npos);
    EXPECT_TRUE(rendered.find("Reasoning effort: high") != std::string::npos);
}

TEST_F(EncodingTest, RenderSystemContentWithTools) {
    SystemContent system_content = SystemContent::new_system_content()
                                      .with_browser_tool()
                                      .with_python_tool();
    
    Message message = Message::from_role_and_content(Role::System, system_content);
    
    auto tokens = encoding_->render_conversation(Conversation::from_messages({message}));
    std::string rendered = encoding_->tokenizer().decode_utf8(tokens);
    
    EXPECT_TRUE(rendered.find("Available tools:") != std::string::npos);
    EXPECT_TRUE(rendered.find("browser") != std::string::npos);
    EXPECT_TRUE(rendered.find("python") != std::string::npos);
}

TEST_F(EncodingTest, RenderDeveloperContent) {
    DeveloperContent dev_content = DeveloperContent::new_developer_content()
                                      .with_instructions("Be helpful and accurate");
    
    Message message = Message::from_role_and_content(Role::Developer, dev_content);
    
    auto tokens = encoding_->render_conversation(Conversation::from_messages({message}));
    std::string rendered = encoding_->tokenizer().decode_utf8(tokens);
    
    EXPECT_TRUE(rendered.find("Be helpful and accurate") != std::string::npos);
}

TEST_F(EncodingTest, RenderConversationForCompletion) {
    Conversation conversation = Conversation::from_messages({
        Message::from_role_and_content(Role::System, SystemContent::new_system_content()),
        Message::from_role_and_content(Role::User, TextContent("Hello"))
    });
    
    auto tokens = encoding_->render_conversation_for_completion(conversation, Role::Assistant);
    std::string rendered = encoding_->tokenizer().decode_utf8(tokens);
    
    // Should end with start token for assistant
    EXPECT_TRUE(rendered.find("<|start|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|assistant|>") != std::string::npos);
}

TEST_F(EncodingTest, ParseMessagesFromTokens) {
    // Create a simple harmony format string
    std::string harmony_text = "<|start|>user<|message|>Hello world<|end|>";
    auto tokens = encoding_->tokenizer().encode_with_special_tokens(harmony_text);
    
    auto messages = encoding_->parse_messages_from_completion_tokens(tokens);
    
    EXPECT_EQ(messages.size(), 1);
    EXPECT_EQ(messages[0].author.role, Role::User);
    EXPECT_EQ(messages[0].content.size(), 1);
    
    auto text_content = std::get<TextContent>(messages[0].content[0]);
    EXPECT_EQ(text_content.text, "Hello world");
}

TEST_F(EncodingTest, ParseMessagesWithChannel) {
    std::string harmony_text = "<|start|>assistant<|channel|>final<|message|>Response<|end|>";
    auto tokens = encoding_->tokenizer().encode_with_special_tokens(harmony_text);
    
    auto messages = encoding_->parse_messages_from_completion_tokens(tokens);
    
    EXPECT_EQ(messages.size(), 1);
    EXPECT_EQ(messages[0].author.role, Role::Assistant);
    EXPECT_TRUE(messages[0].channel.has_value());
    EXPECT_EQ(*messages[0].channel, "final");
}

TEST_F(EncodingTest, ParseMultipleMessages) {
    std::string harmony_text = "<|start|>user<|message|>Question<|end|><|start|>assistant<|message|>Answer<|end|>";
    auto tokens = encoding_->tokenizer().encode_with_special_tokens(harmony_text);
    
    auto messages = encoding_->parse_messages_from_completion_tokens(tokens);
    
    EXPECT_EQ(messages.size(), 2);
    EXPECT_EQ(messages[0].author.role, Role::User);
    EXPECT_EQ(messages[1].author.role, Role::Assistant);
}

// StreamableParser tests
class StreamableParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        encoding_ = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
        ASSERT_NE(encoding_, nullptr);
        parser_ = std::make_unique<StreamableParser>(*encoding_, Role::Assistant);
    }
    
    void TearDown() override {}
    
    std::shared_ptr<HarmonyEncoding> encoding_;
    std::unique_ptr<StreamableParser> parser_;
};

TEST_F(StreamableParserTest, BasicConstruction) {
    EXPECT_NE(parser_, nullptr);
    EXPECT_TRUE(parser_->messages().empty());
    EXPECT_TRUE(parser_->current_content().empty());
}

TEST_F(StreamableParserTest, ProcessSimpleMessage) {
    // Simulate streaming tokens for "<|start|>assistant<|message|>Hello<|end|>"
    std::string harmony_text = "<|start|>assistant<|message|>Hello<|end|>";
    auto tokens = encoding_->tokenizer().encode_with_special_tokens(harmony_text);
    
    for (Rank token : tokens) {
        parser_->process(token);
    }
    parser_->process_eos();
    
    auto messages = parser_->messages();
    EXPECT_EQ(messages.size(), 1);
    EXPECT_EQ(messages[0].author.role, Role::Assistant);
    
    auto text_content = std::get<TextContent>(messages[0].content[0]);
    EXPECT_EQ(text_content.text, "Hello");
}

TEST_F(StreamableParserTest, ProcessMessageWithChannel) {
    std::string harmony_text = "<|start|>assistant<|channel|>final<|message|>Response<|end|>";
    auto tokens = encoding_->tokenizer().encode_with_special_tokens(harmony_text);
    
    for (Rank token : tokens) {
        parser_->process(token);
    }
    parser_->process_eos();
    
    auto messages = parser_->messages();
    EXPECT_EQ(messages.size(), 1);
    EXPECT_TRUE(messages[0].channel.has_value());
    EXPECT_EQ(*messages[0].channel, "final");
}

TEST_F(StreamableParserTest, ProcessIncompleteMessage) {
    // Process partial message
    std::string partial_text = "<|start|>assistant<|message|>Partial";
    auto tokens = encoding_->tokenizer().encode_with_special_tokens(partial_text);
    
    for (Rank token : tokens) {
        parser_->process(token);
    }
    
    // Should have current content but no complete messages yet
    EXPECT_FALSE(parser_->current_content().empty());
    EXPECT_TRUE(parser_->messages().empty());
    
    // Complete the message
    parser_->process_eos();
    
    auto messages = parser_->messages();
    EXPECT_EQ(messages.size(), 1);
}

TEST_F(StreamableParserTest, CurrentStateTracking) {
    // Start processing a message
    auto start_token = encoding_->tokenizer().encode_with_special_tokens("<|start|>")[0];
    auto assistant_token = encoding_->tokenizer().encode_with_special_tokens("<|assistant|>")[0];
    auto message_token = encoding_->tokenizer().encode_with_special_tokens("<|message|>")[0];
    
    parser_->process(start_token);
    parser_->process(assistant_token);
    
    EXPECT_TRUE(parser_->current_role().has_value());
    EXPECT_EQ(*parser_->current_role(), Role::Assistant);
    
    parser_->process(message_token);
    
    // Add some content
    auto hello_tokens = encoding_->tokenizer().encode_ordinary("Hello");
    for (Rank token : hello_tokens) {
        parser_->process(token);
    }
    
    EXPECT_FALSE(parser_->current_content().empty());
}

// Integration tests
TEST_F(EncodingTest, RoundtripConversation) {
    // Create a conversation
    Conversation original_conversation = Conversation::from_messages({
        Message::from_role_and_content(Role::System, SystemContent::new_system_content()),
        Message::from_role_and_content(Role::User, TextContent("Hello")),
        Message::from_role_and_content(Role::Assistant, TextContent("Hi there!"))
    });
    
    // Render to tokens
    auto tokens = encoding_->render_conversation(original_conversation);
    
    // Parse back to messages
    auto parsed_messages = encoding_->parse_messages_from_completion_tokens(tokens);
    
    // Should have same number of messages
    EXPECT_EQ(parsed_messages.size(), original_conversation.messages.size());
    
    // Check roles match
    for (size_t i = 0; i < parsed_messages.size(); ++i) {
        EXPECT_EQ(parsed_messages[i].author.role, original_conversation.messages[i].author.role);
    }
}

TEST_F(EncodingTest, ComplexMessageRoundtrip) {
    Message complex_message = Message::from_role_and_content(
        Role::Assistant, 
        TextContent("Complex response")
    ).with_channel("analysis")
     .with_recipient("user")
     .with_content_type("text");
    
    auto tokens = encoding_->render_conversation(Conversation::from_messages({complex_message}));
    auto parsed_messages = encoding_->parse_messages_from_completion_tokens(tokens);
    
    EXPECT_EQ(parsed_messages.size(), 1);
    EXPECT_EQ(parsed_messages[0].author.role, Role::Assistant);
    EXPECT_TRUE(parsed_messages[0].channel.has_value());
    EXPECT_EQ(*parsed_messages[0].channel, "analysis");
}

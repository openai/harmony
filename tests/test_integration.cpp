#include <gtest/gtest.h>
#include "openai_harmony/harmony.hpp"
#include <fstream>
#include <sstream>

using namespace openai_harmony;

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        encoding_ = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss);
        ASSERT_NE(encoding_, nullptr);
    }
    
    void TearDown() override {}
    
    std::shared_ptr<HarmonyEncoding> encoding_;
};

TEST_F(IntegrationTest, EndToEndConversationRendering) {
    // Create a complete conversation
    auto conversation = Conversation::from_messages({
        Message::from_role_and_content(
            Role::System,
            SystemContent::new_system_content()
                .with_model_identity("Test Assistant")
                .with_reasoning_effort(ReasoningEffort::Medium)
                .with_browser_tool()
                .with_required_channels({"analysis", "final"})
        ),
        Message::from_role_and_content(
            Role::Developer,
            DeveloperContent::new_developer_content()
                .with_instructions("Be helpful and accurate")
                .with_function_tools({
                    ToolDescription::new_tool(
                        "get_weather",
                        "Get current weather",
                        nlohmann::json{{"type", "object"}, {"properties", {{"location", {{"type", "string"}}}}}}
                    )
                })
        ),
        Message::from_role_and_content(Role::User, TextContent("What's the weather like?")),
        Message::from_role_and_content(
            Role::Assistant,
            TextContent("I'll check the weather for you.")
        ).with_channel("analysis"),
        Message::from_role_and_content(
            Role::Assistant,
            TextContent("{\"location\": \"current\"}")
        ).with_channel("commentary")
         .with_recipient("get_weather")
         .with_content_type("json"),
        Message::from_role_and_content(
            Role::Tool,
            TextContent("Weather: Sunny, 72°F")
        ),
        Message::from_role_and_content(
            Role::Assistant,
            TextContent("The weather is currently sunny and 72°F.")
        ).with_channel("final")
    });
    
    // Render the conversation
    auto tokens = encoding_->render_conversation(conversation);
    EXPECT_FALSE(tokens.empty());
    
    // Decode to see the harmony format
    std::string rendered = encoding_->tokenizer().decode_utf8(tokens);
    
    // Verify harmony format structure
    EXPECT_TRUE(rendered.find("<|start|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|system|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|developer|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|user|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|assistant|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|tool|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|message|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|end|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|channel|>") != std::string::npos);
    EXPECT_TRUE(rendered.find("<|constrain|>") != std::string::npos);
    
    // Verify specific content
    EXPECT_TRUE(rendered.find("Test Assistant") != std::string::npos);
    EXPECT_TRUE(rendered.find("analysis") != std::string::npos);
    EXPECT_TRUE(rendered.find("final") != std::string::npos);
    EXPECT_TRUE(rendered.find("get_weather") != std::string::npos);
    EXPECT_TRUE(rendered.find("json") != std::string::npos);
}

TEST_F(IntegrationTest, ConversationRoundtrip) {
    // Create original conversation
    auto original_conversation = Conversation::from_messages({
        Message::from_role_and_content(Role::System, SystemContent::new_system_content()),
        Message::from_role_and_content(Role::User, TextContent("Hello")),
        Message::from_role_and_content(
            Role::Assistant, 
            TextContent("Hi there!")
        ).with_channel("final")
    });
    
    // Render to tokens
    auto tokens = encoding_->render_conversation(original_conversation);
    
    // Parse back to messages
    auto parsed_messages = encoding_->parse_messages_from_completion_tokens(tokens);
    
    // Verify structure is preserved
    EXPECT_EQ(parsed_messages.size(), original_conversation.messages.size());
    
    for (size_t i = 0; i < parsed_messages.size(); ++i) {
        EXPECT_EQ(parsed_messages[i].author.role, original_conversation.messages[i].author.role);
        
        // Check channel preservation for assistant message
        if (original_conversation.messages[i].author.role == Role::Assistant) {
            EXPECT_TRUE(parsed_messages[i].channel.has_value());
            EXPECT_EQ(*parsed_messages[i].channel, "final");
        }
    }
}

TEST_F(IntegrationTest, StreamingParserIntegration) {
    // Create a harmony format response
    std::string harmony_response = 
        "<|start|>assistant<|channel|>analysis<|message|>Let me think about this...<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Here's my answer: 42<|end|>";
    
    auto tokens = encoding_->tokenizer().encode_with_special_tokens(harmony_response);
    
    // Process with streaming parser
    StreamableParser parser(*encoding_, Role::Assistant);
    
    for (Rank token : tokens) {
        parser.process(token);
    }
    parser.process_eos();
    
    // Verify parsed messages
    auto messages = parser.messages();
    EXPECT_EQ(messages.size(), 2);
    
    EXPECT_EQ(messages[0].author.role, Role::Assistant);
    EXPECT_TRUE(messages[0].channel.has_value());
    EXPECT_EQ(*messages[0].channel, "analysis");
    
    EXPECT_EQ(messages[1].author.role, Role::Assistant);
    EXPECT_TRUE(messages[1].channel.has_value());
    EXPECT_EQ(*messages[1].channel, "final");
}

TEST_F(IntegrationTest, ComplexToolCallScenario) {
    // Simulate a complex tool calling scenario
    auto conversation = Conversation::from_messages({
        Message::from_role_and_content(
            Role::System,
            SystemContent::new_system_content()
                .with_browser_tool()
                .with_python_tool()
        ),
        Message::from_role_and_content(Role::User, TextContent("Search for Python tutorials and analyze the results")),
        Message::from_role_and_content(
            Role::Assistant,
            TextContent("I'll search for Python tutorials and analyze them for you.")
        ).with_channel("analysis"),
        Message::from_role_and_content(
            Role::Assistant,
            TextContent("{\"query\": \"Python tutorials\", \"topn\": 5}")
        ).with_channel("commentary")
         .with_recipient("search")
         .with_content_type("json"),
        Message::from_role_and_content(
            Role::Tool,
            TextContent("Found 5 Python tutorial results...")
        ),
        Message::from_role_and_content(
            Role::Assistant,
            TextContent("# Analyzing search results\nresults = [...]")
        ).with_channel("commentary")
         .with_recipient("python")
         .with_content_type("python"),
        Message::from_role_and_content(
            Role::Tool,
            TextContent("Analysis complete: Top tutorials focus on basics, web development, and data science.")
        ),
        Message::from_role_and_content(
            Role::Assistant,
            TextContent("Based on my search and analysis, here are the top Python tutorial categories...")
        ).with_channel("final")
    });
    
    // Render and verify
    auto tokens = encoding_->render_conversation(conversation);
    std::string rendered = encoding_->tokenizer().decode_utf8(tokens);
    
    // Should contain all the tool interaction elements
    EXPECT_TRUE(rendered.find("browser") != std::string::npos);
    EXPECT_TRUE(rendered.find("python") != std::string::npos);
    EXPECT_TRUE(rendered.find("search") != std::string::npos);
    EXPECT_TRUE(rendered.find("json") != std::string::npos);
    EXPECT_TRUE(rendered.find("analysis") != std::string::npos);
    EXPECT_TRUE(rendered.find("commentary") != std::string::npos);
    EXPECT_TRUE(rendered.find("final") != std::string::npos);
}

TEST_F(IntegrationTest, JsonSerializationIntegration) {
    // Create a complex conversation
    auto conversation = Conversation::from_messages({
        Message::from_role_and_content(
            Role::System,
            SystemContent::new_system_content()
                .with_model_identity("Integration Test Model")
                .with_reasoning_effort(ReasoningEffort::High)
        ),
        Message::from_role_and_content(Role::User, TextContent("Test message"))
    });
    
    // Serialize to JSON
    nlohmann::json j = conversation;
    
    // Verify JSON structure
    EXPECT_TRUE(j.contains("messages"));
    EXPECT_EQ(j["messages"].size(), 2);
    
    // Deserialize back
    Conversation restored_conversation;
    from_json(j, restored_conversation);
    
    // Verify restoration
    EXPECT_EQ(restored_conversation.messages.size(), 2);
    EXPECT_EQ(restored_conversation.messages[0].author.role, Role::System);
    EXPECT_EQ(restored_conversation.messages[1].author.role, Role::User);
    
    // Test that both conversations render the same way
    auto original_tokens = encoding_->render_conversation(conversation);
    auto restored_tokens = encoding_->render_conversation(restored_conversation);
    
    EXPECT_EQ(original_tokens, restored_tokens);
}

TEST_F(IntegrationTest, PerformanceBaseline) {
    // Create a moderately large conversation
    std::vector<Message> messages;
    
    // Add system message
    messages.push_back(Message::from_role_and_content(
        Role::System,
        SystemContent::new_system_content()
            .with_browser_tool()
            .with_python_tool()
    ));
    
    // Add alternating user/assistant messages
    for (int i = 0; i < 50; ++i) {
        messages.push_back(Message::from_role_and_content(
            Role::User,
            TextContent("User message " + std::to_string(i))
        ));
        
        messages.push_back(Message::from_role_and_content(
            Role::Assistant,
            TextContent("Assistant response " + std::to_string(i))
        ).with_channel(i % 2 == 0 ? "analysis" : "final"));
    }
    
    auto conversation = Conversation::from_messages(messages);
    
    // Time the rendering (basic performance check)
    auto start = std::chrono::high_resolution_clock::now();
    auto tokens = encoding_->render_conversation(conversation);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete in reasonable time (less than 1 second for 101 messages)
    EXPECT_LT(duration.count(), 1000);
    EXPECT_FALSE(tokens.empty());
    
    // Verify we can parse it back
    auto parsed_messages = encoding_->parse_messages_from_completion_tokens(tokens);
    EXPECT_EQ(parsed_messages.size(), messages.size());
}

TEST_F(IntegrationTest, ErrorHandlingIntegration) {
    // Test various error conditions in integration
    
    // Invalid tokens should throw
    std::vector<Rank> invalid_tokens = {999999, 888888, 777777};
    EXPECT_THROW(encoding_->tokenizer().decode_utf8(invalid_tokens), DecodeKeyError);
    
    // Empty conversation should work
    Conversation empty_conversation;
    auto empty_tokens = encoding_->render_conversation(empty_conversation);
    EXPECT_TRUE(empty_tokens.empty());
    
    // Malformed harmony text should not crash parser
    std::string malformed_harmony = "<|start|>invalid_role<|message|>test<|end|>";
    auto malformed_tokens = encoding_->tokenizer().encode_with_special_tokens(malformed_harmony);
    
    EXPECT_NO_THROW({
        auto parsed = encoding_->parse_messages_from_completion_tokens(malformed_tokens);
        // May be empty or have default role, but shouldn't crash
    });
}

// Test with actual test data files if available
TEST_F(IntegrationTest, TestDataCompatibility) {
    // Try to load and process test data files
    std::vector<std::string> test_files = {
        "test-data/test_simple_convo.txt",
        "test-data/test_browser_tool_only.txt",
        "test-data/test_reasoning_system_message.txt"
    };
    
    for (const auto& filename : test_files) {
        std::ifstream file(filename);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string content = buffer.str();
            
            if (!content.empty()) {
                // Try to tokenize the content
                EXPECT_NO_THROW({
                    auto tokens = encoding_->tokenizer().encode_ordinary(content);
                    EXPECT_FALSE(tokens.empty());
                    
                    // Try to decode back
                    std::string decoded = encoding_->tokenizer().decode_utf8(tokens);
                    EXPECT_FALSE(decoded.empty());
                });
            }
        }
    }
}

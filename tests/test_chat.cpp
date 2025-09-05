#include <gtest/gtest.h>
#include "openai_harmony/chat.hpp"
#include <nlohmann/json.hpp>

using namespace openai_harmony;

class ChatTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Role tests
TEST_F(ChatTest, RoleToString) {
    EXPECT_EQ(role_to_string(Role::User), "user");
    EXPECT_EQ(role_to_string(Role::Assistant), "assistant");
    EXPECT_EQ(role_to_string(Role::System), "system");
    EXPECT_EQ(role_to_string(Role::Developer), "developer");
    EXPECT_EQ(role_to_string(Role::Tool), "tool");
}

TEST_F(ChatTest, StringToRole) {
    EXPECT_EQ(string_to_role("user"), Role::User);
    EXPECT_EQ(string_to_role("assistant"), Role::Assistant);
    EXPECT_EQ(string_to_role("system"), Role::System);
    EXPECT_EQ(string_to_role("developer"), Role::Developer);
    EXPECT_EQ(string_to_role("tool"), Role::Tool);
}

TEST_F(ChatTest, InvalidRoleString) {
    EXPECT_THROW(string_to_role("invalid"), std::invalid_argument);
}

// ReasoningEffort tests
TEST_F(ChatTest, ReasoningEffortToString) {
    EXPECT_EQ(reasoning_effort_to_string(ReasoningEffort::Low), "low");
    EXPECT_EQ(reasoning_effort_to_string(ReasoningEffort::Medium), "medium");
    EXPECT_EQ(reasoning_effort_to_string(ReasoningEffort::High), "high");
}

// Author tests
TEST_F(ChatTest, AuthorConstruction) {
    Author author(Role::User);
    EXPECT_EQ(author.role, Role::User);
    EXPECT_FALSE(author.name.has_value());
    
    Author named_author(Role::Assistant, "GPT-4");
    EXPECT_EQ(named_author.role, Role::Assistant);
    EXPECT_TRUE(named_author.name.has_value());
    EXPECT_EQ(*named_author.name, "GPT-4");
}

TEST_F(ChatTest, AuthorNewAuthor) {
    auto author = Author::new_author(Role::System, "System");
    EXPECT_EQ(author.role, Role::System);
    EXPECT_TRUE(author.name.has_value());
    EXPECT_EQ(*author.name, "System");
}

TEST_F(ChatTest, AuthorEquality) {
    Author author1(Role::User);
    Author author2(Role::User);
    Author author3(Role::Assistant);
    
    EXPECT_EQ(author1, author2);
    EXPECT_NE(author1, author3);
}

// ToolDescription tests
TEST_F(ChatTest, ToolDescriptionConstruction) {
    ToolDescription tool("test_tool", "A test tool");
    EXPECT_EQ(tool.name, "test_tool");
    EXPECT_EQ(tool.description, "A test tool");
    EXPECT_FALSE(tool.parameters.has_value());
}

TEST_F(ChatTest, ToolDescriptionWithParameters) {
    nlohmann::json params = {{"type", "object"}, {"properties", {{"param1", {{"type", "string"}}}}}};
    ToolDescription tool("test_tool", "A test tool", params);
    
    EXPECT_EQ(tool.name, "test_tool");
    EXPECT_EQ(tool.description, "A test tool");
    EXPECT_TRUE(tool.parameters.has_value());
    EXPECT_EQ(*tool.parameters, params);
}

TEST_F(ChatTest, ToolDescriptionNewTool) {
    auto tool = ToolDescription::new_tool("weather", "Get weather info");
    EXPECT_EQ(tool.name, "weather");
    EXPECT_EQ(tool.description, "Get weather info");
}

// ChannelConfig tests
TEST_F(ChatTest, ChannelConfigRequireChannels) {
    auto config = ChannelConfig::require_channels({"analysis", "final"});
    EXPECT_TRUE(config.channel_required);
    EXPECT_EQ(config.valid_channels.size(), 2);
    EXPECT_EQ(config.valid_channels[0], "analysis");
    EXPECT_EQ(config.valid_channels[1], "final");
}

// ToolNamespaceConfig tests
TEST_F(ChatTest, ToolNamespaceConfigConstruction) {
    std::vector<ToolDescription> tools = {
        ToolDescription::new_tool("tool1", "First tool"),
        ToolDescription::new_tool("tool2", "Second tool")
    };
    
    ToolNamespaceConfig ns_config("test_namespace", "Test namespace", tools);
    EXPECT_EQ(ns_config.name, "test_namespace");
    EXPECT_TRUE(ns_config.description.has_value());
    EXPECT_EQ(*ns_config.description, "Test namespace");
    EXPECT_EQ(ns_config.tools.size(), 2);
}

TEST_F(ChatTest, ToolNamespaceConfigBrowser) {
    auto browser_config = ToolNamespaceConfig::browser();
    EXPECT_EQ(browser_config.name, "browser");
    EXPECT_TRUE(browser_config.description.has_value());
    EXPECT_FALSE(browser_config.tools.empty());
    
    // Should have search, open, find tools
    bool has_search = false, has_open = false, has_find = false;
    for (const auto& tool : browser_config.tools) {
        if (tool.name == "search") has_search = true;
        if (tool.name == "open") has_open = true;
        if (tool.name == "find") has_find = true;
    }
    EXPECT_TRUE(has_search);
    EXPECT_TRUE(has_open);
    EXPECT_TRUE(has_find);
}

TEST_F(ChatTest, ToolNamespaceConfigPython) {
    auto python_config = ToolNamespaceConfig::python();
    EXPECT_EQ(python_config.name, "python");
    EXPECT_TRUE(python_config.description.has_value());
}

// TextContent tests
TEST_F(ChatTest, TextContentConstruction) {
    TextContent content("Hello, world!");
    EXPECT_EQ(content.text, "Hello, world!");
}

// SystemContent tests
TEST_F(ChatTest, SystemContentConstruction) {
    SystemContent content = SystemContent::new_system_content();
    EXPECT_TRUE(content.model_identity.has_value());
    EXPECT_TRUE(content.reasoning_effort.has_value());
    EXPECT_TRUE(content.knowledge_cutoff.has_value());
    EXPECT_TRUE(content.channel_config.has_value());
}

TEST_F(ChatTest, SystemContentBuilder) {
    SystemContent content = SystemContent::new_system_content()
                               .with_model_identity("Test Model")
                               .with_reasoning_effort(ReasoningEffort::High)
                               .with_conversation_start_date("2024-01-01")
                               .with_knowledge_cutoff("2023-12");
    
    EXPECT_EQ(*content.model_identity, "Test Model");
    EXPECT_EQ(*content.reasoning_effort, ReasoningEffort::High);
    EXPECT_EQ(*content.conversation_start_date, "2024-01-01");
    EXPECT_EQ(*content.knowledge_cutoff, "2023-12");
}

TEST_F(ChatTest, SystemContentWithTools) {
    SystemContent content = SystemContent::new_system_content()
                               .with_browser_tool()
                               .with_python_tool();
    
    EXPECT_TRUE(content.tools.has_value());
    EXPECT_EQ(content.tools->size(), 2);
    EXPECT_TRUE(content.tools->count("browser") > 0);
    EXPECT_TRUE(content.tools->count("python") > 0);
}

// DeveloperContent tests
TEST_F(ChatTest, DeveloperContentConstruction) {
    DeveloperContent content = DeveloperContent::new_developer_content();
    EXPECT_FALSE(content.instructions.has_value());
    EXPECT_FALSE(content.tools.has_value());
}

TEST_F(ChatTest, DeveloperContentBuilder) {
    DeveloperContent content = DeveloperContent::new_developer_content()
                                  .with_instructions("Be helpful")
                                  .with_function_tools({
                                      ToolDescription::new_tool("test", "Test tool")
                                  });
    
    EXPECT_TRUE(content.instructions.has_value());
    EXPECT_EQ(*content.instructions, "Be helpful");
    EXPECT_TRUE(content.tools.has_value());
    EXPECT_EQ(content.tools->size(), 1);
}

// Message tests
TEST_F(ChatTest, MessageConstruction) {
    Message message(Author(Role::User), {TextContent("Hello")});
    EXPECT_EQ(message.author.role, Role::User);
    EXPECT_EQ(message.content.size(), 1);
}

TEST_F(ChatTest, MessageFromRoleAndContent) {
    auto message = Message::from_role_and_content(Role::User, TextContent("Hello"));
    EXPECT_EQ(message.author.role, Role::User);
    EXPECT_EQ(message.content.size(), 1);
}

TEST_F(ChatTest, MessageBuilder) {
    auto message = Message::from_role_and_content(Role::Assistant, TextContent("Response"))
                      .with_channel("final")
                      .with_recipient("user")
                      .with_content_type("text");
    
    EXPECT_EQ(message.author.role, Role::Assistant);
    EXPECT_TRUE(message.channel.has_value());
    EXPECT_EQ(*message.channel, "final");
    EXPECT_TRUE(message.recipient.has_value());
    EXPECT_EQ(*message.recipient, "user");
    EXPECT_TRUE(message.content_type.has_value());
    EXPECT_EQ(*message.content_type, "text");
}

TEST_F(ChatTest, MessageAddingContent) {
    auto message = Message::from_role_and_content(Role::User, TextContent("First"))
                      .adding_content(TextContent("Second"));
    
    EXPECT_EQ(message.content.size(), 2);
}

// Conversation tests
TEST_F(ChatTest, ConversationConstruction) {
    Conversation conversation;
    EXPECT_TRUE(conversation.messages.empty());
}

TEST_F(ChatTest, ConversationFromMessages) {
    std::vector<Message> messages = {
        Message::from_role_and_content(Role::User, TextContent("Hello")),
        Message::from_role_and_content(Role::Assistant, TextContent("Hi"))
    };
    
    auto conversation = Conversation::from_messages(messages);
    EXPECT_EQ(conversation.messages.size(), 2);
}

TEST_F(ChatTest, ConversationIteration) {
    auto conversation = Conversation::from_messages({
        Message::from_role_and_content(Role::User, TextContent("Hello"))
    });
    
    int count = 0;
    for (const auto& message : conversation) {
        EXPECT_EQ(message.author.role, Role::User);
        count++;
    }
    EXPECT_EQ(count, 1);
}

// JSON serialization tests
TEST_F(ChatTest, RoleJsonSerialization) {
    nlohmann::json j = Role::User;
    EXPECT_EQ(j.get<std::string>(), "user");
    
    Role role;
    from_json(j, role);
    EXPECT_EQ(role, Role::User);
}

TEST_F(ChatTest, AuthorJsonSerialization) {
    Author author(Role::Assistant, "GPT-4");
    nlohmann::json j = author;
    
    EXPECT_EQ(j["role"].get<std::string>(), "assistant");
    EXPECT_EQ(j["name"].get<std::string>(), "GPT-4");
    
    Author restored_author(Role::User);
    from_json(j, restored_author);
    EXPECT_EQ(restored_author, author);
}

TEST_F(ChatTest, ToolDescriptionJsonSerialization) {
    nlohmann::json params = {{"type", "object"}};
    ToolDescription tool("test", "Test tool", params);
    nlohmann::json j = tool;
    
    EXPECT_EQ(j["name"].get<std::string>(), "test");
    EXPECT_EQ(j["description"].get<std::string>(), "Test tool");
    EXPECT_EQ(j["parameters"], params);
    
    ToolDescription restored_tool;
    from_json(j, restored_tool);
    EXPECT_EQ(restored_tool, tool);
}

TEST_F(ChatTest, MessageJsonSerialization) {
    auto message = Message::from_role_and_content(Role::User, TextContent("Hello"))
                      .with_channel("final");
    
    nlohmann::json j = message;
    EXPECT_EQ(j["role"].get<std::string>(), "user");
    EXPECT_EQ(j["channel"].get<std::string>(), "final");
    
    Message restored_message(Author(Role::User), {});
    from_json(j, restored_message);
    EXPECT_EQ(restored_message.author.role, Role::User);
    EXPECT_TRUE(restored_message.channel.has_value());
    EXPECT_EQ(*restored_message.channel, "final");
}

TEST_F(ChatTest, ConversationJsonSerialization) {
    auto conversation = Conversation::from_messages({
        Message::from_role_and_content(Role::User, TextContent("Hello")),
        Message::from_role_and_content(Role::Assistant, TextContent("Hi"))
    });
    
    nlohmann::json j = conversation;
    EXPECT_TRUE(j.contains("messages"));
    EXPECT_EQ(j["messages"].size(), 2);
    
    Conversation restored_conversation;
    from_json(j, restored_conversation);
    EXPECT_EQ(restored_conversation.messages.size(), 2);
}

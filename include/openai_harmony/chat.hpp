#pragma once

#include "export.hpp"
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <unordered_map>
#include <variant>
#include <nlohmann/json.hpp>

namespace openai_harmony {

/**
 * @brief Role of a message author
 */
enum class Role {
    User,
    Assistant,
    System,
    Developer,
    Tool
};

/**
 * @brief Convert Role to string
 */
OPENAI_HARMONY_API std::string role_to_string(Role role);

/**
 * @brief Convert string to Role
 */
OPENAI_HARMONY_API Role string_to_role(const std::string& str);

/**
 * @brief Reasoning effort level for system messages
 */
enum class ReasoningEffort {
    Low,
    Medium,
    High
};

/**
 * @brief Convert ReasoningEffort to string
 */
OPENAI_HARMONY_API std::string reasoning_effort_to_string(ReasoningEffort effort);

/**
 * @brief Author information for a message
 */
struct Author {
    Role role;
    std::optional<std::string> name;

    Author(Role r, std::optional<std::string> n = std::nullopt) 
        : role(r), name(std::move(n)) {}

    static Author new_author(Role role, const std::string& name) {
        return Author(role, name);
    }

    bool operator==(const Author& other) const {
        return role == other.role && name == other.name;
    }
};

/**
 * @brief Tool description for function calling
 */
struct ToolDescription {
    std::string name;
    std::string description;
    std::optional<nlohmann::json> parameters;

    ToolDescription() = default;
    
    ToolDescription(std::string n, std::string d, std::optional<nlohmann::json> p = std::nullopt)
        : name(std::move(n)), description(std::move(d)), parameters(std::move(p)) {}

    static ToolDescription new_tool(const std::string& name, const std::string& description, 
                                   const std::optional<nlohmann::json>& parameters = std::nullopt) {
        return ToolDescription(name, description, parameters);
    }

    bool operator==(const ToolDescription& other) const {
        return name == other.name && description == other.description && parameters == other.parameters;
    }
};

/**
 * @brief Channel configuration for system messages
 */
struct ChannelConfig {
    std::vector<std::string> valid_channels;
    bool channel_required = false;

    ChannelConfig() = default;

    static ChannelConfig require_channels(const std::vector<std::string>& channels) {
        ChannelConfig config;
        config.valid_channels = channels;
        config.channel_required = true;
        return config;
    }

    bool operator==(const ChannelConfig& other) const {
        return valid_channels == other.valid_channels && channel_required == other.channel_required;
    }
};

/**
 * @brief Tool namespace configuration
 */
struct ToolNamespaceConfig {
    std::string name;
    std::optional<std::string> description;
    std::vector<ToolDescription> tools;

    ToolNamespaceConfig() = default;
    
    ToolNamespaceConfig(std::string n, std::optional<std::string> d, std::vector<ToolDescription> t)
        : name(std::move(n)), description(std::move(d)), tools(std::move(t)) {}

    static ToolNamespaceConfig new_namespace(const std::string& name, 
                                           const std::optional<std::string>& description,
                                           const std::vector<ToolDescription>& tools) {
        return ToolNamespaceConfig(name, description, tools);
    }

    static ToolNamespaceConfig browser();
    static ToolNamespaceConfig python();

    bool operator==(const ToolNamespaceConfig& other) const {
        return name == other.name && description == other.description && tools == other.tools;
    }
};

// Forward declarations
struct TextContent;
struct SystemContent;
struct DeveloperContent;

/**
 * @brief Content variant for messages
 */
using Content = std::variant<TextContent, SystemContent, DeveloperContent>;

/**
 * @brief Plain text content
 */
struct TextContent {
    std::string text;

    TextContent() = default;
    explicit TextContent(std::string t) : text(std::move(t)) {}

    bool operator==(const TextContent& other) const {
        return text == other.text;
    }
};

/**
 * @brief System-specific content with model identity and configuration
 */
struct SystemContent {
    std::optional<std::string> model_identity;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<std::unordered_map<std::string, ToolNamespaceConfig>> tools;
    std::optional<std::string> conversation_start_date;
    std::optional<std::string> knowledge_cutoff;
    std::optional<ChannelConfig> channel_config;

    OPENAI_HARMONY_API SystemContent();

    static OPENAI_HARMONY_API SystemContent new_system_content() {
        return SystemContent();
    }

    OPENAI_HARMONY_API SystemContent& with_model_identity(const std::string& identity);
    OPENAI_HARMONY_API SystemContent& with_reasoning_effort(ReasoningEffort effort);
    OPENAI_HARMONY_API SystemContent& with_tools(const ToolNamespaceConfig& ns_config);
    OPENAI_HARMONY_API SystemContent& with_conversation_start_date(const std::string& date);
    OPENAI_HARMONY_API SystemContent& with_knowledge_cutoff(const std::string& cutoff);
    OPENAI_HARMONY_API SystemContent& with_channel_config(const ChannelConfig& config);
    OPENAI_HARMONY_API SystemContent& with_required_channels(const std::vector<std::string>& channels);
    OPENAI_HARMONY_API SystemContent& with_browser_tool();
    OPENAI_HARMONY_API SystemContent& with_python_tool();

    bool operator==(const SystemContent& other) const;
};

/**
 * @brief Developer-specific content with instructions and tools
 */
struct DeveloperContent {
    std::optional<std::string> instructions;
    std::optional<std::unordered_map<std::string, ToolNamespaceConfig>> tools;

    DeveloperContent() = default;

    static DeveloperContent new_developer_content() {
        return DeveloperContent();
    }

    OPENAI_HARMONY_API DeveloperContent& with_instructions(const std::string& instr);
    OPENAI_HARMONY_API DeveloperContent& with_tools(const ToolNamespaceConfig& ns_config);
    OPENAI_HARMONY_API DeveloperContent& with_function_tools(const std::vector<ToolDescription>& tools);

    bool operator==(const DeveloperContent& other) const;
};

/**
 * @brief Message structure
 */
struct Message {
    Author author;
    std::optional<std::string> recipient;
    std::vector<Content> content;
    std::optional<std::string> channel;
    std::optional<std::string> content_type;

    Message(Author a, std::vector<Content> c = {})
        : author(std::move(a)), content(std::move(c)) {}

    static OPENAI_HARMONY_API Message from_author_and_content(const Author& author, const Content& content);
    static OPENAI_HARMONY_API Message from_role_and_content(Role role, const Content& content);
    static OPENAI_HARMONY_API Message from_role_and_contents(Role role, const std::vector<Content>& contents);

    OPENAI_HARMONY_API Message& adding_content(const Content& content);
    OPENAI_HARMONY_API Message& with_channel(const std::string& channel);
    OPENAI_HARMONY_API Message& with_recipient(const std::string& recipient);
    OPENAI_HARMONY_API Message& with_content_type(const std::string& content_type);

    bool operator==(const Message& other) const;
};

/**
 * @brief Conversation containing multiple messages
 */
struct Conversation {
    std::vector<Message> messages;

    Conversation() = default;
    explicit Conversation(std::vector<Message> msgs) : messages(std::move(msgs)) {}

    static Conversation from_messages(const std::vector<Message>& messages) {
        return Conversation(messages);
    }

    // Iterator support
    auto begin() { return messages.begin(); }
    auto end() { return messages.end(); }
    auto begin() const { return messages.begin(); }
    auto end() const { return messages.end(); }

    bool operator==(const Conversation& other) const {
        return messages == other.messages;
    }
};

// JSON serialization support
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const Role& role);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, Role& role);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const ReasoningEffort& effort);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, ReasoningEffort& effort);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const Author& author);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, Author& author);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const ToolDescription& tool);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, ToolDescription& tool);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const ChannelConfig& config);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, ChannelConfig& config);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const ToolNamespaceConfig& ns_config);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, ToolNamespaceConfig& ns_config);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const TextContent& content);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, TextContent& content);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const SystemContent& content);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, SystemContent& content);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const DeveloperContent& content);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, DeveloperContent& content);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const Content& content);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, Content& content);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const Message& message);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, Message& message);
OPENAI_HARMONY_API void to_json(nlohmann::json& j, const Conversation& conversation);
OPENAI_HARMONY_API void from_json(const nlohmann::json& j, Conversation& conversation);

} // namespace openai_harmony

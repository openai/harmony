#include "openai_harmony/chat.hpp"
#include <stdexcept>
#include <algorithm>

namespace openai_harmony {

// Role conversion functions
std::string role_to_string(Role role) {
    switch (role) {
        case Role::User: return "user";
        case Role::Assistant: return "assistant";
        case Role::System: return "system";
        case Role::Developer: return "developer";
        case Role::Tool: return "tool";
        default: throw std::invalid_argument("Unknown role");
    }
}

Role string_to_role(const std::string& str) {
    if (str == "user") return Role::User;
    if (str == "assistant") return Role::Assistant;
    if (str == "system") return Role::System;
    if (str == "developer") return Role::Developer;
    if (str == "tool") return Role::Tool;
    throw std::invalid_argument("Unknown role: " + str);
}

// ReasoningEffort conversion
std::string reasoning_effort_to_string(ReasoningEffort effort) {
    switch (effort) {
        case ReasoningEffort::Low: return "low";
        case ReasoningEffort::Medium: return "medium";
        case ReasoningEffort::High: return "high";
        default: throw std::invalid_argument("Unknown reasoning effort");
    }
}

// ToolNamespaceConfig static methods
ToolNamespaceConfig ToolNamespaceConfig::browser() {
    std::vector<ToolDescription> tools = {
        ToolDescription::new_tool(
            "search",
            "Searches for information related to `query` and displays `topn` results.",
            nlohmann::json{
                {"type", "object"},
                {"properties", {
                    {"query", {{"type", "string"}}},
                    {"topn", {{"type", "number"}, {"default", 10}}},
                    {"source", {{"type", "string"}}}
                }},
                {"required", nlohmann::json::array({"query"})}
            }
        ),
        ToolDescription::new_tool(
            "open",
            "Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.\nValid link ids are displayed with the formatting: `【{id}†.*】`.\nIf `cursor` is not provided, the most recent page is implied.\nIf `id` is a string, it is treated as a fully qualified URL associated with `source`.\nIf `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.\nUse this function without `id` to scroll to a new location of an opened page.",
            nlohmann::json{
                {"type", "object"},
                {"properties", {
                    {"id", {{"type", nlohmann::json::array({"number", "string"})}, {"default", -1}}},
                    {"cursor", {{"type", "number"}, {"default", -1}}},
                    {"loc", {{"type", "number"}, {"default", -1}}},
                    {"num_lines", {{"type", "number"}, {"default", -1}}},
                    {"view_source", {{"type", "boolean"}, {"default", false}}},
                    {"source", {{"type", "string"}}}
                }}
            }
        ),
        ToolDescription::new_tool(
            "find",
            "Finds exact matches of `pattern` in the current page, or the page given by `cursor`.",
            nlohmann::json{
                {"type", "object"},
                {"properties", {
                    {"pattern", {{"type", "string"}}},
                    {"cursor", {{"type", "number"}, {"default", -1}}}
                }},
                {"required", nlohmann::json::array({"pattern"})}
            }
        )
    };

    return ToolNamespaceConfig::new_namespace(
        "browser",
        "Tool for browsing.\nThe `cursor` appears in brackets before each browsing display: `[{cursor}]`.\nCite information from the tool using the following format:\n`【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.\nDo not quote more than 10 words directly from the tool output.\nsources=web (default: web)",
        tools
    );
}

ToolNamespaceConfig ToolNamespaceConfig::python() {
    return ToolNamespaceConfig::new_namespace(
        "python",
        "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\n\nWhen you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.",
        {}
    );
}

// SystemContent implementation
SystemContent::SystemContent() {
    model_identity = "You are ChatGPT, a large language model trained by OpenAI.";
    reasoning_effort = ReasoningEffort::Medium;
    knowledge_cutoff = "2024-06";
    channel_config = ChannelConfig::require_channels({"analysis", "commentary", "final"});
}

SystemContent& SystemContent::with_model_identity(const std::string& identity) {
    model_identity = identity;
    return *this;
}

SystemContent& SystemContent::with_reasoning_effort(ReasoningEffort effort) {
    reasoning_effort = effort;
    return *this;
}

SystemContent& SystemContent::with_tools(const ToolNamespaceConfig& ns_config) {
    if (!tools) {
        tools = std::unordered_map<std::string, ToolNamespaceConfig>();
    }
    (*tools)[ns_config.name] = ns_config;
    return *this;
}

SystemContent& SystemContent::with_conversation_start_date(const std::string& date) {
    conversation_start_date = date;
    return *this;
}

SystemContent& SystemContent::with_knowledge_cutoff(const std::string& cutoff) {
    knowledge_cutoff = cutoff;
    return *this;
}

SystemContent& SystemContent::with_channel_config(const ChannelConfig& config) {
    channel_config = config;
    return *this;
}

SystemContent& SystemContent::with_required_channels(const std::vector<std::string>& channels) {
    channel_config = ChannelConfig::require_channels(channels);
    return *this;
}

SystemContent& SystemContent::with_browser_tool() {
    return with_tools(ToolNamespaceConfig::browser());
}

SystemContent& SystemContent::with_python_tool() {
    return with_tools(ToolNamespaceConfig::python());
}

bool SystemContent::operator==(const SystemContent& other) const {
    return model_identity == other.model_identity &&
           reasoning_effort == other.reasoning_effort &&
           tools == other.tools &&
           conversation_start_date == other.conversation_start_date &&
           knowledge_cutoff == other.knowledge_cutoff &&
           channel_config == other.channel_config;
}

// DeveloperContent implementation
DeveloperContent& DeveloperContent::with_instructions(const std::string& instr) {
    instructions = instr;
    return *this;
}

DeveloperContent& DeveloperContent::with_tools(const ToolNamespaceConfig& ns_config) {
    if (!tools) {
        tools = std::unordered_map<std::string, ToolNamespaceConfig>();
    }
    (*tools)[ns_config.name] = ns_config;
    return *this;
}

DeveloperContent& DeveloperContent::with_function_tools(const std::vector<ToolDescription>& function_tools) {
    return with_tools(ToolNamespaceConfig::new_namespace("functions", std::nullopt, function_tools));
}

bool DeveloperContent::operator==(const DeveloperContent& other) const {
    return instructions == other.instructions && tools == other.tools;
}

// Message implementation
Message Message::from_author_and_content(const Author& author, const Content& content) {
    return Message(author, {content});
}

Message Message::from_role_and_content(Role role, const Content& content) {
    return from_author_and_content(Author(role), content);
}

Message Message::from_role_and_contents(Role role, const std::vector<Content>& contents) {
    return Message(Author(role), contents);
}

Message& Message::adding_content(const Content& content) {
    this->content.push_back(content);
    return *this;
}

Message& Message::with_channel(const std::string& channel) {
    this->channel = channel;
    return *this;
}

Message& Message::with_recipient(const std::string& recipient) {
    this->recipient = recipient;
    return *this;
}

Message& Message::with_content_type(const std::string& content_type) {
    this->content_type = content_type;
    return *this;
}

bool Message::operator==(const Message& other) const {
    return author == other.author &&
           recipient == other.recipient &&
           content == other.content &&
           channel == other.channel &&
           content_type == other.content_type;
}

// JSON serialization implementations
void to_json(nlohmann::json& j, const Role& role) {
    j = role_to_string(role);
}

void from_json(const nlohmann::json& j, Role& role) {
    role = string_to_role(j.get<std::string>());
}

void to_json(nlohmann::json& j, const ReasoningEffort& effort) {
    j = reasoning_effort_to_string(effort);
}

void from_json(const nlohmann::json& j, ReasoningEffort& effort) {
    const std::string str = j.get<std::string>();
    if (str == "low") effort = ReasoningEffort::Low;
    else if (str == "medium") effort = ReasoningEffort::Medium;
    else if (str == "high") effort = ReasoningEffort::High;
    else throw std::invalid_argument("Unknown reasoning effort: " + str);
}

void to_json(nlohmann::json& j, const Author& author) {
    j = nlohmann::json{{"role", author.role}};
    if (author.name) {
        j["name"] = *author.name;
    }
}

void from_json(const nlohmann::json& j, Author& author) {
    j.at("role").get_to(author.role);
    if (j.contains("name")) {
        author.name = j.at("name").get<std::string>();
    }
}

void to_json(nlohmann::json& j, const ToolDescription& tool) {
    j = nlohmann::json{
        {"name", tool.name},
        {"description", tool.description}
    };
    if (tool.parameters) {
        j["parameters"] = *tool.parameters;
    }
}

void from_json(const nlohmann::json& j, ToolDescription& tool) {
    j.at("name").get_to(tool.name);
    j.at("description").get_to(tool.description);
    if (j.contains("parameters")) {
        tool.parameters = j.at("parameters");
    }
}

void to_json(nlohmann::json& j, const ChannelConfig& config) {
    j = nlohmann::json{
        {"valid_channels", config.valid_channels},
        {"channel_required", config.channel_required}
    };
}

void from_json(const nlohmann::json& j, ChannelConfig& config) {
    j.at("valid_channels").get_to(config.valid_channels);
    j.at("channel_required").get_to(config.channel_required);
}

void to_json(nlohmann::json& j, const ToolNamespaceConfig& ns_config) {
    j = nlohmann::json{
        {"name", ns_config.name},
        {"tools", ns_config.tools}
    };
    if (ns_config.description) {
        j["description"] = *ns_config.description;
    }
}

void from_json(const nlohmann::json& j, ToolNamespaceConfig& ns_config) {
    j.at("name").get_to(ns_config.name);
    
    // Manually deserialize tools to avoid template issues
    if (j.contains("tools")) {
        const auto& tools_json = j.at("tools");
        ns_config.tools.clear();
        ns_config.tools.reserve(tools_json.size());
        for (const auto& tool_json : tools_json) {
            ToolDescription tool;
            from_json(tool_json, tool);
            ns_config.tools.push_back(std::move(tool));
        }
    }
    
    if (j.contains("description")) {
        ns_config.description = j.at("description").get<std::string>();
    }
}

void to_json(nlohmann::json& j, const TextContent& content) {
    j = nlohmann::json{
        {"type", "text"},
        {"text", content.text}
    };
}

void from_json(const nlohmann::json& j, TextContent& content) {
    j.at("text").get_to(content.text);
}

void to_json(nlohmann::json& j, const SystemContent& content) {
    j = nlohmann::json{{"type", "system_content"}};
    if (content.model_identity) j["model_identity"] = *content.model_identity;
    if (content.reasoning_effort) j["reasoning_effort"] = *content.reasoning_effort;
    if (content.tools) j["tools"] = *content.tools;
    if (content.conversation_start_date) j["conversation_start_date"] = *content.conversation_start_date;
    if (content.knowledge_cutoff) j["knowledge_cutoff"] = *content.knowledge_cutoff;
    if (content.channel_config) j["channel_config"] = *content.channel_config;
}

void from_json(const nlohmann::json& j, SystemContent& content) {
    if (j.contains("model_identity")) {
        content.model_identity = j.at("model_identity").get<std::string>();
    }
    if (j.contains("reasoning_effort")) {
        content.reasoning_effort = j.at("reasoning_effort").get<ReasoningEffort>();
    }
    if (j.contains("tools")) {
        const auto& tools_json = j.at("tools");
        content.tools = std::unordered_map<std::string, ToolNamespaceConfig>();
        for (const auto& [key, value] : tools_json.items()) {
            ToolNamespaceConfig ns_config;
            from_json(value, ns_config);
            (*content.tools)[key] = std::move(ns_config);
        }
    }
    if (j.contains("conversation_start_date")) {
        content.conversation_start_date = j.at("conversation_start_date").get<std::string>();
    }
    if (j.contains("knowledge_cutoff")) {
        content.knowledge_cutoff = j.at("knowledge_cutoff").get<std::string>();
    }
    if (j.contains("channel_config")) {
        content.channel_config = j.at("channel_config").get<ChannelConfig>();
    }
}

void to_json(nlohmann::json& j, const DeveloperContent& content) {
    j = nlohmann::json{{"type", "developer_content"}};
    if (content.instructions) j["instructions"] = *content.instructions;
    if (content.tools) j["tools"] = *content.tools;
}

void from_json(const nlohmann::json& j, DeveloperContent& content) {
    if (j.contains("instructions")) {
        content.instructions = j.at("instructions").get<std::string>();
    }
    if (j.contains("tools")) {
        const auto& tools_json = j.at("tools");
        content.tools = std::unordered_map<std::string, ToolNamespaceConfig>();
        for (const auto& [key, value] : tools_json.items()) {
            ToolNamespaceConfig ns_config;
            from_json(value, ns_config);
            (*content.tools)[key] = std::move(ns_config);
        }
    }
}

void to_json(nlohmann::json& j, const Content& content) {
    std::visit([&j](const auto& c) { to_json(j, c); }, content);
}

void from_json(const nlohmann::json& j, Content& content) {
    const std::string type = j.at("type").get<std::string>();
    if (type == "text") {
        TextContent text_content;
        from_json(j, text_content);
        content = std::move(text_content);
    } else if (type == "system_content") {
        SystemContent system_content;
        from_json(j, system_content);
        content = std::move(system_content);
    } else if (type == "developer_content") {
        DeveloperContent developer_content;
        from_json(j, developer_content);
        content = std::move(developer_content);
    } else {
        throw std::invalid_argument("Unknown content type: " + type);
    }
}

void to_json(nlohmann::json& j, const Message& message) {
    j = nlohmann::json{
        {"role", message.author.role},
        {"content", message.content}
    };
    if (message.author.name) j["name"] = *message.author.name;
    if (message.recipient) j["recipient"] = *message.recipient;
    if (message.channel) j["channel"] = *message.channel;
    if (message.content_type) j["content_type"] = *message.content_type;
}

void from_json(const nlohmann::json& j, Message& message) {
    message.author.role = j.at("role").get<Role>();
    if (j.contains("name")) {
        message.author.name = j.at("name").get<std::string>();
    }
    
    // Handle content as either string or array
    if (j.at("content").is_string()) {
        message.content = {TextContent(j.at("content").get<std::string>())};
    } else {
        const auto& content_json = j.at("content");
        message.content.clear();
        message.content.reserve(content_json.size());
        for (const auto& content_item : content_json) {
            Content content;
            from_json(content_item, content);
            message.content.push_back(std::move(content));
        }
    }
    
    if (j.contains("recipient")) {
        message.recipient = j.at("recipient").get<std::string>();
    }
    if (j.contains("channel")) {
        message.channel = j.at("channel").get<std::string>();
    }
    if (j.contains("content_type")) {
        message.content_type = j.at("content_type").get<std::string>();
    }
}

void to_json(nlohmann::json& j, const Conversation& conversation) {
    j = nlohmann::json{{"messages", conversation.messages}};
}

void from_json(const nlohmann::json& j, Conversation& conversation) {
    const auto& messages_json = j.at("messages");
    conversation.messages.clear();
    conversation.messages.reserve(messages_json.size());
    for (const auto& message_json : messages_json) {
        Message message(Author(Role::User), {});
        from_json(message_json, message);
        conversation.messages.push_back(std::move(message));
    }
}

} // namespace openai_harmony

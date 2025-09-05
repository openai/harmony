#pragma once

#include "export.hpp"
#include "chat.hpp"
#include "tiktoken.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <optional>
#include <variant>
#include <nlohmann/json.hpp>

namespace openai_harmony {

/**
 * @brief Formatting tokens used in harmony protocol
 */
enum class FormattingToken {
    Start,
    Message,
    EndMessage,
    EndMessageDoneSampling,
    EndMessageAssistantToTool,
    Refusal,
    ConstrainedFormat,
    Channel,
    BeginUntrusted,
    EndUntrusted,
    MetaSep,
    MetaEnd
};

/**
 * @brief Convert FormattingToken to string
 */
std::string formatting_token_to_string(FormattingToken token);

/**
 * @brief Exception for rendering formatting tokens
 */
class RenderFormattingTokenError : public std::runtime_error {
public:
    FormattingToken token;
    
    explicit RenderFormattingTokenError(FormattingToken t) 
        : std::runtime_error("tried to render unmapped formatting token"), token(t) {}
        
    RenderFormattingTokenError(FormattingToken t, const std::vector<Rank>& encoding)
        : std::runtime_error("Expected encoding of formatting token to be a single token"), token(t) {}
};

/**
 * @brief Parsed representation of a message header
 */
struct ParsedHeader {
    Author author;
    std::optional<std::string> recipient;
    std::optional<std::string> channel;
    std::optional<std::string> content_type;

    bool operator==(const ParsedHeader& other) const {
        return author == other.author && recipient == other.recipient && 
               channel == other.channel && content_type == other.content_type;
    }
};

/**
 * @brief Configuration for rendering conversations
 */
struct RenderConversationConfig {
    bool auto_drop_analysis = true;

    bool operator==(const RenderConversationConfig& other) const {
        return auto_drop_analysis == other.auto_drop_analysis;
    }
};

/**
 * @brief Options for rendering individual messages
 */
struct RenderOptions {
    bool conversation_has_function_tools = false;

    bool operator==(const RenderOptions& other) const {
        return conversation_has_function_tools == other.conversation_has_function_tools;
    }
};

/**
 * @brief Stream state for incremental parsing
 */
enum class StreamState {
    ExpectStart,
    Header,
    Content
};

/**
 * @brief Main encoding class for harmony protocol
 */
class OPENAI_HARMONY_CLASS HarmonyEncoding {
private:
    std::string name_;
    size_t n_ctx_;
    size_t max_message_tokens_;
    size_t max_action_length_;
    std::string tokenizer_name_;
    std::shared_ptr<CoreBPE> tokenizer_;
    std::unordered_map<FormattingToken, std::string> format_token_mapping_;
    std::unordered_set<FormattingToken> stop_formatting_tokens_;
    std::unordered_set<FormattingToken> stop_formatting_tokens_for_assistant_actions_;

    // Internal rendering helpers
    std::optional<std::string> mapped_format_token(FormattingToken token) const;
    Rank render_formatting_token(FormattingToken token) const;
    void render_formatting_token_into(FormattingToken token, std::vector<Rank>& into) const;
    void render_text_into(const std::string& text, std::vector<Rank>& into) const;

    // Content rendering helpers
    void render_content_into(const Content& content, std::vector<Rank>& into, 
                           const std::optional<RenderOptions>& render_options) const;
    void render_text_content_into(const TextContent& content, std::vector<Rank>& into) const;
    void render_system_content_into(const SystemContent& content, std::vector<Rank>& into,
                                  const std::optional<RenderOptions>& render_options) const;
    void render_developer_content_into(const DeveloperContent& content, std::vector<Rank>& into) const;

    // Tool rendering helpers
    static std::string json_schema_to_typescript(const nlohmann::json& schema, const std::string& indent);
    static std::string template_tools_section(const std::unordered_map<std::string, ToolNamespaceConfig>& tools);

    // Helper methods used in implementation
    std::vector<Rank> render_message(const Message& message, 
                                   const std::optional<RenderConversationConfig>& config) const;
    Rank get_special_token(const std::string& token) const;
    std::string role_to_special_token(Role role) const;
    std::vector<Message> parse_harmony_format(const std::string& text) const;

public:
    /**
     * @brief Construct HarmonyEncoding
     */
    HarmonyEncoding(const std::string& name,
                   size_t n_ctx,
                   size_t max_message_tokens,
                   size_t max_action_length,
                   const std::string& tokenizer_name,
                   std::shared_ptr<CoreBPE> tokenizer,
                   const std::unordered_map<FormattingToken, std::string>& format_token_mapping,
                   const std::unordered_set<FormattingToken>& stop_formatting_tokens,
                   const std::unordered_set<FormattingToken>& stop_formatting_tokens_for_assistant_actions);

    // Getters
    const std::string& name() const { return name_; }
    const std::string& tokenizer_name() const { return tokenizer_name_; }
    size_t max_message_tokens() const { return max_message_tokens_; }
    const CoreBPE& tokenizer() const { return *tokenizer_; }

    // Stop tokens
    std::unordered_set<Rank> stop_tokens() const;
    std::unordered_set<Rank> stop_tokens_for_assistant_actions() const;

    // Rendering methods
    std::vector<Rank> render_conversation_for_completion(
        const Conversation& conversation,
        Role next_turn_role,
        const std::optional<RenderConversationConfig>& config = std::nullopt) const;

    std::vector<Rank> render_conversation(
        const Conversation& conversation,
        const std::optional<RenderConversationConfig>& config = std::nullopt) const;

    std::vector<Rank> render_conversation_for_training(
        const Conversation& conversation,
        const std::optional<RenderConversationConfig>& config = std::nullopt) const;

    std::vector<Rank> render(
        const Message& message,
        const std::optional<RenderOptions>& render_options = std::nullopt) const;

    void render_into(
        const Message& message,
        std::vector<Rank>& into,
        const std::optional<RenderOptions>& render_options = std::nullopt) const;

    void render_conversation_into(
        const Conversation& conversation,
        std::vector<Rank>& into,
        const std::optional<RenderConversationConfig>& config = std::nullopt) const;

    void render_conversation_for_completion_into(
        const Conversation& conversation,
        Role next_turn_role,
        std::vector<Rank>& into,
        const std::optional<RenderConversationConfig>& config = std::nullopt) const;

    // Parsing methods
    std::vector<Message> parse_messages_from_completion_tokens(
        const std::vector<Rank>& tokens,
        const std::optional<Role>& role = std::nullopt) const;
};

/**
 * @brief Incremental parser for streaming tokens
 */
class OPENAI_HARMONY_CLASS StreamableParser {
private:
    HarmonyEncoding encoding_;
    std::optional<Role> next_role_;
    std::vector<Rank> tokens_;
    std::vector<Message> messages_;
    StreamState state_;
    std::unordered_set<Rank> stop_tokens_;
    std::optional<std::string> last_content_delta_;
    std::vector<Rank> undecoded_tokens_;

    // State-specific data
    std::vector<Rank> header_tokens_;
    std::optional<ParsedHeader> current_header_;
    std::vector<Rank> content_tokens_;

    // Current parsing state
    std::string current_content_;
    std::optional<Role> current_role_;
    std::optional<std::string> current_channel_;
    std::optional<std::string> current_recipient_;
    std::optional<std::string> current_content_type_;

    // Internal methods
    void process_next(const std::optional<Rank>& token);
    ParsedHeader parse_header_from_tokens(const std::vector<Rank>& header_tokens, 
                                        const std::optional<Role>& role) const;
    std::optional<Role> parse_role_from_token(const std::string& token) const;
    void finalize_current_message();

public:
    /**
     * @brief Construct StreamableParser
     */
    StreamableParser(const HarmonyEncoding& encoding, const std::optional<Role>& role = std::nullopt);

    /**
     * @brief Process a single token
     */
    StreamableParser& process(Rank token);

    /**
     * @brief Process end-of-stream
     */
    StreamableParser& process_eos();

    // Getters
    std::string current_content() const;
    std::optional<Role> current_role() const;
    std::optional<std::string> current_content_type() const;
    std::optional<std::string> last_content_delta() const;
    std::optional<std::string> current_recipient() const;
    std::optional<std::string> current_channel() const;
    const std::vector<Message>& messages() const { return messages_; }
    const std::vector<Rank>& tokens() const { return tokens_; }
    StreamState state() const { return state_; }

    // State serialization
    nlohmann::json state_json() const;
};

} // namespace openai_harmony

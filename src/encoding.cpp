#include "openai_harmony/encoding.hpp"
#include "openai_harmony/utils.hpp"
#include <sstream>
#include <algorithm>
#include <regex>

namespace openai_harmony {

// HarmonyEncoding implementation
HarmonyEncoding::HarmonyEncoding(const std::string& name,
                               size_t n_ctx,
                               size_t max_message_tokens,
                               size_t max_action_length,
                               const std::string& tokenizer_name,
                               std::shared_ptr<CoreBPE> tokenizer,
                               const std::unordered_map<FormattingToken, std::string>& format_token_mapping,
                               const std::unordered_set<FormattingToken>& stop_formatting_tokens,
                               const std::unordered_set<FormattingToken>& stop_formatting_tokens_for_assistant_actions)
    : name_(name),
      n_ctx_(n_ctx),
      max_message_tokens_(max_message_tokens),
      max_action_length_(max_action_length),
      tokenizer_name_(tokenizer_name),
      tokenizer_(std::move(tokenizer)),
      format_token_mapping_(format_token_mapping),
      stop_formatting_tokens_(stop_formatting_tokens),
      stop_formatting_tokens_for_assistant_actions_(stop_formatting_tokens_for_assistant_actions) {
}

std::vector<Rank> HarmonyEncoding::render_conversation_for_completion(
    const Conversation& conversation, 
    Role next_turn_role,
    const std::optional<RenderConversationConfig>& config) const {
    
    std::vector<Rank> tokens;
    
    // Render each message in the conversation
    for (const auto& message : conversation.messages) {
        auto message_tokens = render_message(message, config);
        tokens.insert(tokens.end(), message_tokens.begin(), message_tokens.end());
    }
    
    // Add start token for next turn
    tokens.push_back(get_special_token("<|start|>"));
    tokens.push_back(get_special_token(role_to_special_token(next_turn_role)));
    
    return tokens;
}

std::vector<Rank> HarmonyEncoding::render_conversation(
    const Conversation& conversation,
    const std::optional<RenderConversationConfig>& config) const {
    
    std::vector<Rank> tokens;
    
    for (const auto& message : conversation.messages) {
        auto message_tokens = render_message(message, config);
        tokens.insert(tokens.end(), message_tokens.begin(), message_tokens.end());
    }
    
    return tokens;
}

std::vector<Message> HarmonyEncoding::parse_messages_from_completion_tokens(
    const std::vector<Rank>& tokens,
    const std::optional<Role>& role) const {
    
    std::vector<Message> messages;
    
    // Convert tokens back to text first
    std::string text = tokenizer_->decode_utf8(tokens);
    
    // Parse the harmony format
    messages = parse_harmony_format(text);
    
    return messages;
}

// Stub implementations for missing methods
std::vector<Rank> HarmonyEncoding::render_conversation_for_training(
    const Conversation& conversation,
    const std::optional<RenderConversationConfig>& config) const {
    // For now, same as regular render
    return render_conversation(conversation, config);
}

std::vector<Rank> HarmonyEncoding::render(
    const Message& message,
    const std::optional<RenderOptions>& render_options) const {
    std::vector<Rank> tokens;
    render_into(message, tokens, render_options);
    return tokens;
}

void HarmonyEncoding::render_into(
    const Message& message,
    std::vector<Rank>& into,
    const std::optional<RenderOptions>& render_options) const {
    
    // Simplified rendering - just encode the message content
    for (const auto& content : message.content) {
        if (std::holds_alternative<TextContent>(content)) {
            const auto& text_content = std::get<TextContent>(content);
            auto tokens = tokenizer_->encode_ordinary(text_content.text);
            into.insert(into.end(), tokens.begin(), tokens.end());
        }
    }
}

void HarmonyEncoding::render_conversation_into(
    const Conversation& conversation,
    std::vector<Rank>& into,
    const std::optional<RenderConversationConfig>& config) const {
    
    auto tokens = render_conversation(conversation, config);
    into.insert(into.end(), tokens.begin(), tokens.end());
}

void HarmonyEncoding::render_conversation_for_completion_into(
    const Conversation& conversation,
    Role next_turn_role,
    std::vector<Rank>& into,
    const std::optional<RenderConversationConfig>& config) const {
    
    auto tokens = render_conversation_for_completion(conversation, next_turn_role, config);
    into.insert(into.end(), tokens.begin(), tokens.end());
}

std::unordered_set<Rank> HarmonyEncoding::stop_tokens() const {
    std::unordered_set<Rank> result;
    // Add stop tokens based on formatting tokens
    return result;
}

std::unordered_set<Rank> HarmonyEncoding::stop_tokens_for_assistant_actions() const {
    std::unordered_set<Rank> result;
    // Add assistant action stop tokens
    return result;
}

// Missing methods that are called in the implementation
std::vector<Rank> HarmonyEncoding::render_message(
    const Message& message, 
    const std::optional<RenderConversationConfig>& config) const {
    
    std::vector<Rank> tokens;
    
    // Start message with proper harmony format
    tokens.push_back(get_special_token("<|start|>"));
    tokens.push_back(get_special_token(role_to_special_token(message.author.role)));
    
    // Add channel if present
    if (message.channel) {
        tokens.push_back(get_special_token("<|channel|>"));
        auto channel_tokens = tokenizer_->encode_ordinary(*message.channel);
        tokens.insert(tokens.end(), channel_tokens.begin(), channel_tokens.end());
    }
    
    // Add recipient if present
    if (message.recipient) {
        std::string recipient_marker = " to=" + *message.recipient;
        auto recipient_tokens = tokenizer_->encode_ordinary(recipient_marker);
        tokens.insert(tokens.end(), recipient_tokens.begin(), recipient_tokens.end());
    }
    
    // Add content type constraint if present
    if (message.content_type) {
        tokens.push_back(get_special_token("<|constrain|>"));
        auto content_type_tokens = tokenizer_->encode_ordinary(*message.content_type);
        tokens.insert(tokens.end(), content_type_tokens.begin(), content_type_tokens.end());
    }
    
    // Add message content
    tokens.push_back(get_special_token("<|message|>"));
    
    for (const auto& content : message.content) {
        if (std::holds_alternative<TextContent>(content)) {
            const auto& text_content = std::get<TextContent>(content);
            auto content_tokens = tokenizer_->encode_ordinary(text_content.text);
            tokens.insert(tokens.end(), content_tokens.begin(), content_tokens.end());
        } else if (std::holds_alternative<SystemContent>(content)) {
            const auto& system_content = std::get<SystemContent>(content);
            std::stringstream ss;
            
            if (system_content.model_identity) {
                ss << "Model: " << *system_content.model_identity << "\n";
            }
            
            if (system_content.reasoning_effort) {
                ss << "Reasoning effort: " << reasoning_effort_to_string(*system_content.reasoning_effort) << "\n";
            }
            
            if (system_content.knowledge_cutoff) {
                ss << "Knowledge cutoff: " << *system_content.knowledge_cutoff << "\n";
            }
            
            if (system_content.conversation_start_date) {
                ss << "Current date: " << *system_content.conversation_start_date << "\n";
            }
            
            if (system_content.tools) {
                ss << "\nAvailable tools:\n";
                for (const auto& [namespace_name, ns_config] : *system_content.tools) {
                    ss << "# " << ns_config.name;
                    if (ns_config.description) {
                        ss << "\n" << *ns_config.description;
                    }
                    ss << "\n";
                    
                    for (const auto& tool : ns_config.tools) {
                        ss << "## " << tool.name << "\n";
                        ss << tool.description << "\n";
                        if (tool.parameters) {
                            ss << "Parameters: " << tool.parameters->dump() << "\n";
                        }
                        ss << "\n";
                    }
                }
            }
            
            if (system_content.channel_config) {
                const auto& config = *system_content.channel_config;
                if (config.channel_required && !config.valid_channels.empty()) {
                    ss << "\nRequired channels: ";
                    for (size_t i = 0; i < config.valid_channels.size(); ++i) {
                        if (i > 0) ss << ", ";
                        ss << config.valid_channels[i];
                    }
                    ss << "\n";
                }
            }
            
            auto system_tokens = tokenizer_->encode_ordinary(ss.str());
            tokens.insert(tokens.end(), system_tokens.begin(), system_tokens.end());
        } else if (std::holds_alternative<DeveloperContent>(content)) {
            const auto& dev_content = std::get<DeveloperContent>(content);
            std::stringstream ss;
            
            if (dev_content.instructions) {
                ss << *dev_content.instructions << "\n";
            }
            
            if (dev_content.tools) {
                ss << "\nDeveloper tools:\n";
                for (const auto& [namespace_name, ns_config] : *dev_content.tools) {
                    ss << "# " << ns_config.name;
                    if (ns_config.description) {
                        ss << "\n" << *ns_config.description;
                    }
                    ss << "\n";
                    
                    for (const auto& tool : ns_config.tools) {
                        ss << "## " << tool.name << "\n";
                        ss << tool.description << "\n";
                        if (tool.parameters) {
                            ss << "Parameters: " << tool.parameters->dump() << "\n";
                        }
                        ss << "\n";
                    }
                }
            }
            
            auto dev_tokens = tokenizer_->encode_ordinary(ss.str());
            tokens.insert(tokens.end(), dev_tokens.begin(), dev_tokens.end());
        }
    }
    
    // End message
    tokens.push_back(get_special_token("<|end|>"));
    
    return tokens;
}

Rank HarmonyEncoding::get_special_token(const std::string& token) const {
    auto special_tokens = tokenizer_->special_tokens();
    if (special_tokens.count(token) > 0) {
        // Find the rank for this special token
        auto tokens = tokenizer_->encode_with_special_tokens(token);
        if (!tokens.empty()) {
            return tokens[0];
        }
    }
    throw std::runtime_error("Special token not found: " + token);
}

std::string HarmonyEncoding::role_to_special_token(Role role) const {
    switch (role) {
        case Role::System: return "<|system|>";
        case Role::User: return "<|user|>";
        case Role::Assistant: return "<|assistant|>";
        case Role::Developer: return "<|developer|>";
        case Role::Tool: return "<|tool|>";
        default: throw std::invalid_argument("Unknown role");
    }
}

std::vector<Message> HarmonyEncoding::parse_harmony_format(const std::string& text) const {
    std::vector<Message> messages;
    
    // Simple regex-based parsing for harmony format
    std::regex message_regex(R"(<\|start\|>([^<]+)(?:<\|channel\|>([^<]+))?(?:\s+to=([^<\|]+))?(?:<\|constrain\|>([^<]+))?<\|message\|>(.*?)<\|end\|>)");
    
    std::sregex_iterator iter(text.begin(), text.end(), message_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        const std::smatch& match = *iter;
        
        // Parse role
        std::string role_str = match[1].str();
        Role role = Role::User; // Default
        if (role_str == "system") role = Role::System;
        else if (role_str == "user") role = Role::User;
        else if (role_str == "assistant") role = Role::Assistant;
        else if (role_str == "developer") role = Role::Developer;
        else if (role_str == "tool") role = Role::Tool;
        
        // Create message
        Message message(Author(role), {});
        
        // Add channel if present
        if (match[2].matched) {
            message.channel = match[2].str();
        }
        
        // Add recipient if present
        if (match[3].matched) {
            message.recipient = match[3].str();
        }
        
        // Add content type if present
        if (match[4].matched) {
            message.content_type = match[4].str();
        }
        
        // Add content
        std::string content_text = match[5].str();
        message.content.push_back(TextContent(content_text));
        
        messages.push_back(message);
    }
    
    return messages;
}

// StreamableParser implementation
StreamableParser::StreamableParser(const HarmonyEncoding& encoding, const std::optional<Role>& role)
    : encoding_(encoding), next_role_(role), state_(StreamState::ExpectStart) {
}

StreamableParser& StreamableParser::process(Rank token) {
    // Store token
    tokens_.push_back(token);
    
    // Convert token to text
    std::string token_text = encoding_.tokenizer().decode_utf8({token});
    
    switch (state_) {
        case StreamState::ExpectStart:
            if (token_text == "<|start|>") {
                state_ = StreamState::Header;
            }
            break;
            
        case StreamState::Header:
            current_role_ = parse_role_from_token(token_text);
            if (current_role_) {
                state_ = StreamState::Content;
            }
            break;
            
        case StreamState::Content:
            if (token_text == "<|channel|>") {
                // Handle channel parsing
            } else if (token_text == "<|constrain|>") {
                // Handle constraint parsing
            } else if (token_text == "<|message|>") {
                // Start content collection
            } else if (token_text == "<|end|>") {
                finalize_current_message();
                state_ = StreamState::ExpectStart;
            } else {
                current_content_ += token_text;
                last_content_delta_ = token_text;
            }
            break;
    }
    
    return *this;
}

StreamableParser& StreamableParser::process_eos() {
    if (state_ == StreamState::Content && !current_content_.empty()) {
        finalize_current_message();
    }
    return *this;
}

std::string StreamableParser::current_content() const {
    return current_content_;
}

std::optional<Role> StreamableParser::current_role() const {
    return current_role_;
}

std::optional<std::string> StreamableParser::current_channel() const {
    return current_channel_;
}

std::optional<std::string> StreamableParser::current_recipient() const {
    return current_recipient_;
}

std::optional<std::string> StreamableParser::current_content_type() const {
    return current_content_type_;
}


nlohmann::json StreamableParser::state_json() const {
    nlohmann::json j;
    j["state"] = static_cast<int>(state_);
    j["tokens"] = tokens_;
    j["messages"] = messages_;
    if (current_role_) {
        j["current_role"] = *current_role_;
    }
    if (current_channel_) {
        j["current_channel"] = *current_channel_;
    }
    if (current_recipient_) {
        j["current_recipient"] = *current_recipient_;
    }
    if (current_content_type_) {
        j["current_content_type"] = *current_content_type_;
    }
    j["current_content"] = current_content_;
    return j;
}

// Private helper methods for StreamableParser
std::optional<Role> StreamableParser::parse_role_from_token(const std::string& token) const {
    if (token == "<|system|>") return Role::System;
    if (token == "<|user|>") return Role::User;
    if (token == "<|assistant|>") return Role::Assistant;
    if (token == "<|developer|>") return Role::Developer;
    if (token == "<|tool|>") return Role::Tool;
    return std::nullopt;
}

void StreamableParser::finalize_current_message() {
    if (current_role_) {
        Message message(Author(*current_role_), {TextContent(current_content_)});
        
        if (current_channel_) {
            message.channel = *current_channel_;
        }
        if (current_recipient_) {
            message.recipient = *current_recipient_;
        }
        if (current_content_type_) {
            message.content_type = *current_content_type_;
        }
        
        messages_.push_back(message);
        
        // Reset state
        current_content_.clear();
        current_role_ = std::nullopt;
        current_channel_ = std::nullopt;
        current_recipient_ = std::nullopt;
        current_content_type_ = std::nullopt;
    }
}

} // namespace openai_harmony

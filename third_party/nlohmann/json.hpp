#pragma once

// Minimal nlohmann/json stub for building without the full library
// This is a placeholder to allow compilation - replace with full nlohmann/json for production use

#include <string>
#include <map>
#include <vector>
#include <variant>
#include <iostream>
#include <sstream>

namespace nlohmann {

class json {
public:
    using object_t = std::map<std::string, json>;
    using array_t = std::vector<json>;
    using string_t = std::string;
    using number_t = double;
    using boolean_t = bool;
    using null_t = std::nullptr_t;
    
    using value_t = std::variant<null_t, boolean_t, number_t, string_t, array_t, object_t>;
    
private:
    value_t m_value = nullptr;
    
public:
    // Constructors
    json() = default;
    json(const std::string& s) : m_value(s) {}
    json(const char* s) : m_value(std::string(s)) {}
    json(int i) : m_value(static_cast<double>(i)) {}
    json(double d) : m_value(d) {}
    json(bool b) : m_value(b) {}
    json(std::nullptr_t) : m_value(nullptr) {}
    json(const object_t& obj) : m_value(obj) {}
    json(const array_t& arr) : m_value(arr) {}
    
    // Initializer list constructors
    json(std::initializer_list<std::pair<std::string, json>> init) {
        object_t obj;
        for (const auto& pair : init) {
            obj[pair.first] = pair.second;
        }
        m_value = obj;
    }
    
    // Type checking
    bool is_null() const { return std::holds_alternative<null_t>(m_value); }
    bool is_boolean() const { return std::holds_alternative<boolean_t>(m_value); }
    bool is_number() const { return std::holds_alternative<number_t>(m_value); }
    bool is_string() const { return std::holds_alternative<string_t>(m_value); }
    bool is_array() const { return std::holds_alternative<array_t>(m_value); }
    bool is_object() const { return std::holds_alternative<object_t>(m_value); }
    
    // Access operators
    json& operator[](const std::string& key) {
        if (!is_object()) {
            m_value = object_t{};
        }
        return std::get<object_t>(m_value)[key];
    }
    
    const json& operator[](const std::string& key) const {
        static json null_json;
        if (!is_object()) return null_json;
        const auto& obj = std::get<object_t>(m_value);
        auto it = obj.find(key);
        return (it != obj.end()) ? it->second : null_json;
    }
    
    json& operator[](size_t index) {
        if (!is_array()) {
            m_value = array_t{};
        }
        auto& arr = std::get<array_t>(m_value);
        if (index >= arr.size()) {
            arr.resize(index + 1);
        }
        return arr[index];
    }
    
    // Value getters
    template<typename T>
    T get() const {
        if constexpr (std::is_same_v<T, std::string>) {
            if (is_string()) return std::get<string_t>(m_value);
            return "";
        } else if constexpr (std::is_same_v<T, bool>) {
            if (is_boolean()) return std::get<boolean_t>(m_value);
            return false;
        } else if constexpr (std::is_arithmetic_v<T>) {
            if (is_number()) return static_cast<T>(std::get<number_t>(m_value));
            return T{};
        }
        return T{};
    }
    
    // Convenience methods
    bool contains(const std::string& key) const {
        if (!is_object()) return false;
        const auto& obj = std::get<object_t>(m_value);
        return obj.find(key) != obj.end();
    }
    
    json& at(const std::string& key) {
        if (!is_object()) throw std::runtime_error("not an object");
        auto& obj = std::get<object_t>(m_value);
        auto it = obj.find(key);
        if (it == obj.end()) throw std::runtime_error("key not found");
        return it->second;
    }
    
    const json& at(const std::string& key) const {
        if (!is_object()) throw std::runtime_error("not an object");
        const auto& obj = std::get<object_t>(m_value);
        auto it = obj.find(key);
        if (it == obj.end()) throw std::runtime_error("key not found");
        return it->second;
    }
    
    // Array methods
    static json array(std::initializer_list<json> init = {}) {
        json j;
        j.m_value = array_t(init);
        return j;
    }
    
    void push_back(const json& item) {
        if (!is_array()) {
            m_value = array_t{};
        }
        std::get<array_t>(m_value).push_back(item);
    }
    
    size_t size() const {
        if (is_array()) return std::get<array_t>(m_value).size();
        if (is_object()) return std::get<object_t>(m_value).size();
        return 0;
    }
    
    bool empty() const {
        return size() == 0;
    }
    
    // Serialization (minimal)
    std::string dump(int indent = -1) const {
        std::ostringstream oss;
        dump_impl(oss, indent, 0);
        return oss.str();
    }
    
private:
    void dump_impl(std::ostringstream& oss, int indent, int current_indent) const {
        if (is_null()) {
            oss << "null";
        } else if (is_boolean()) {
            oss << (std::get<boolean_t>(m_value) ? "true" : "false");
        } else if (is_number()) {
            oss << std::get<number_t>(m_value);
        } else if (is_string()) {
            oss << "\"" << std::get<string_t>(m_value) << "\"";
        } else if (is_array()) {
            const auto& arr = std::get<array_t>(m_value);
            oss << "[";
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) oss << ",";
                if (indent >= 0) oss << "\n" << std::string(current_indent + indent, ' ');
                arr[i].dump_impl(oss, indent, current_indent + indent);
            }
            if (indent >= 0 && !arr.empty()) oss << "\n" << std::string(current_indent, ' ');
            oss << "]";
        } else if (is_object()) {
            const auto& obj = std::get<object_t>(m_value);
            oss << "{";
            bool first = true;
            for (const auto& [key, value] : obj) {
                if (!first) oss << ",";
                if (indent >= 0) oss << "\n" << std::string(current_indent + indent, ' ');
                oss << "\"" << key << "\":";
                if (indent >= 0) oss << " ";
                value.dump_impl(oss, indent, current_indent + indent);
                first = false;
            }
            if (indent >= 0 && !obj.empty()) oss << "\n" << std::string(current_indent, ' ');
            oss << "}";
        }
    }
};

// Global operators
inline std::ostream& operator<<(std::ostream& os, const json& j) {
    return os << j.dump();
}

} // namespace nlohmann

// Convenience macro for creating json objects
#define JSON_OBJECT(...) nlohmann::json{__VA_ARGS__}

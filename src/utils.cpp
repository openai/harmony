#include "openai_harmony/utils.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <functional>

namespace openai_harmony {

std::string base64_encode(const std::vector<uint8_t>& data) {
    static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    
    size_t i = 0;
    uint8_t char_array_3[3];
    uint8_t char_array_4[4];
    
    for (uint8_t byte : data) {
        char_array_3[i++] = byte;
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            
            for (i = 0; i < 4; i++) {
                result += chars[char_array_4[i]];
            }
            i = 0;
        }
    }
    
    if (i) {
        for (size_t j = i; j < 3; j++) {
            char_array_3[j] = '\0';
        }
        
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        
        for (size_t j = 0; j < i + 1; j++) {
            result += chars[char_array_4[j]];
        }
        
        while (i++ < 3) {
            result += '=';
        }
    }
    
    return result;
}

std::vector<uint8_t> base64_decode(const std::string& encoded) {
    static const int T[128] = {
        -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
        -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
        -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,62, -1,-1,-1,63,
        52,53,54,55, 56,57,58,59, 60,61,-1,-1, -1,-1,-1,-1,
        -1, 0, 1, 2,  3, 4, 5, 6,  7, 8, 9,10, 11,12,13,14,
        15,16,17,18, 19,20,21,22, 23,24,25,-1, -1,-1,-1,-1,
        -1,26,27,28, 29,30,31,32, 33,34,35,36, 37,38,39,40,
        41,42,43,44, 45,46,47,48, 49,50,51,-1, -1,-1,-1,-1
    };
    
    std::vector<uint8_t> result;
    int val = 0, valb = -8;
    
    for (unsigned char c : encoded) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            result.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    
    return result;
}

std::string sha1_hash(const std::vector<uint8_t>& data) {
    // Simple hash implementation using std::hash (not cryptographically secure)
    // For a production implementation, use a proper crypto library
    std::hash<std::string> hasher;
    std::string data_str(data.begin(), data.end());
    size_t hash_value = hasher(data_str);
    
    std::stringstream ss;
    ss << std::hex << hash_value;
    std::string result = ss.str();
    
    // Pad to 40 characters (SHA1 length)
    while (result.length() < 40) {
        result = "0" + result;
    }
    if (result.length() > 40) {
        result = result.substr(0, 40);
    }
    
    return result;
}

std::string sha256_hash(const std::vector<uint8_t>& data) {
    // Simple hash implementation using std::hash (not cryptographically secure)
    // For a production implementation, use a proper crypto library
    std::hash<std::string> hasher;
    std::string data_str(data.begin(), data.end());
    size_t hash_value = hasher(data_str);
    
    // Create a longer hash by combining multiple hash operations
    std::stringstream ss;
    ss << std::hex << hash_value;
    
    // Add more entropy by hashing the hash
    std::string first_hash = ss.str();
    size_t second_hash = hasher(first_hash);
    ss << std::hex << second_hash;
    
    std::string result = ss.str();
    
    // Pad to 64 characters (SHA256 length)
    while (result.length() < 64) {
        result = "0" + result;
    }
    if (result.length() > 64) {
        result = result.substr(0, 64);
    }
    
    return result;
}

std::vector<std::string> split_string(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> result;
    if (delimiter.empty()) {
        result.push_back(str);
        return result;
    }
    
    size_t start = 0;
    size_t end = str.find(delimiter);
    
    while (end != std::string::npos) {
        result.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    
    result.push_back(str.substr(start));
    return result;
}

std::string join_strings(const std::vector<std::string>& strings, const std::string& delimiter) {
    if (strings.empty()) {
        return "";
    }
    
    std::string result = strings[0];
    for (size_t i = 1; i < strings.size(); ++i) {
        result += delimiter + strings[i];
    }
    
    return result;
}

std::string trim_string(const std::string& str) {
    auto start = str.begin();
    while (start != str.end() && std::isspace(*start)) {
        start++;
    }
    
    auto end = str.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));
    
    return std::string(start, end + 1);
}

} // namespace openai_harmony

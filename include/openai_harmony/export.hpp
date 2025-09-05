#pragma once

/**
 * @file export.hpp
 * @brief Export macros for OpenAI Harmony dynamic library
 */

// Define export/import macros for different platforms
#if defined(_WIN32) || defined(_WIN64)
    #ifdef OPENAI_HARMONY_EXPORTS
        #define OPENAI_HARMONY_API __declspec(dllexport)
    #elif defined(OPENAI_HARMONY_DLL)
        #define OPENAI_HARMONY_API __declspec(dllimport)
    #else
        #define OPENAI_HARMONY_API
    #endif
#elif defined(__GNUC__) && __GNUC__ >= 4
    #ifdef OPENAI_HARMONY_EXPORTS
        #define OPENAI_HARMONY_API __attribute__((visibility("default")))
    #else
        #define OPENAI_HARMONY_API
    #endif
#else
    #define OPENAI_HARMONY_API
#endif

// Convenience macro for class exports
#define OPENAI_HARMONY_CLASS OPENAI_HARMONY_API

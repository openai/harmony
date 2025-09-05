cmake_minimum_required(VERSION 3.20)
project(openai_harmony VERSION 0.0.4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find nlohmann/json - try different names
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    find_package(nlohmann_json3 QUIET)
    if(nlohmann_json3_FOUND)
        # Create alias for compatibility
        add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
    endif()
endif()

if(NOT nlohmann_json_FOUND AND NOT nlohmann_json3_FOUND)
    message(FATAL_ERROR "nlohmann/json not found. Please install nlohmann-json3-dev or nlohmann-json-devel")
endif()

# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Find CURL
find_package(CURL REQUIRED)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files (only include existing ones)
set(HARMONY_SOURCES
    src/chat.cpp
)

# Check for other source files and add them if they exist
foreach(src_file encoding registry tiktoken tiktoken_ext utils)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${src_file}.cpp")
        list(APPEND HARMONY_SOURCES "src/${src_file}.cpp")
    endif()
endforeach()

# Header files
set(HARMONY_HEADERS
    include/openai_harmony/chat.hpp
    include/openai_harmony/encoding.hpp
    include/openai_harmony/registry.hpp
    include/openai_harmony/tiktoken.hpp
    include/openai_harmony/tiktoken_ext.hpp
    include/openai_harmony/utils.hpp
    include/openai_harmony/harmony.hpp
)

# Create the main library
add_library(openai_harmony SHARED ${HARMONY_SOURCES} ${HARMONY_HEADERS})

# Set properties for shared library
set_target_properties(openai_harmony PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Define export macro for Windows DLL
target_compile_definitions(openai_harmony PRIVATE OPENAI_HARMONY_EXPORTS)
if(WIN32)
    target_compile_definitions(openai_harmony INTERFACE OPENAI_HARMONY_DLL)
endif()

# Link libraries - handle different nlohmann_json targets
if(TARGET nlohmann_json::nlohmann_json)
    set(JSON_TARGET nlohmann_json::nlohmann_json)
elseif(TARGET nlohmann_json)
    set(JSON_TARGET nlohmann_json)
else()
    message(FATAL_ERROR "nlohmann_json target not found")
endif()

target_link_libraries(openai_harmony 
    PRIVATE 
    ${JSON_TARGET}
    OpenSSL::SSL 
    OpenSSL::Crypto
    CURL::libcurl
    Threads::Threads
)

# Public headers
target_include_directories(openai_harmony PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Tests
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        enable_testing()
        
        # Check which test files exist
        set(TEST_SOURCES)
        foreach(test_file test_harmony test_chat test_encoding test_tiktoken)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/${test_file}.cpp")
                list(APPEND TEST_SOURCES "tests/${test_file}.cpp")
            endif()
        endforeach()
        
        if(TEST_SOURCES)
            add_executable(harmony_tests ${TEST_SOURCES})
            target_link_libraries(harmony_tests
                PRIVATE
                openai_harmony
                GTest::gtest_main
                GTest::gtest
            )
            add_test(NAME HarmonyTests COMMAND harmony_tests)
        endif()
    else()
        message(WARNING "GTest not found, tests disabled")
    endif()
endif()

# Set RPATH for dynamic library loading
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

# Examples
option(BUILD_EXAMPLES "Build examples" OFF)
if(BUILD_EXAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/basic_usage.cpp")
        add_executable(harmony_example examples/basic_usage.cpp)
        target_link_libraries(harmony_example PRIVATE openai_harmony)
        
        # Set RPATH for finding the shared library
        if(APPLE)
            set_target_properties(harmony_example PROPERTIES
                BUILD_RPATH "${CMAKE_BINARY_DIR}"
                INSTALL_RPATH "@loader_path/../lib"
            )
        elseif(UNIX)
            set_target_properties(harmony_example PROPERTIES
                BUILD_RPATH "${CMAKE_BINARY_DIR}"
                INSTALL_RPATH "$ORIGIN/../lib"
            )
        endif()
    else()
        message(WARNING "Basic example source file not found")
    endif()
    
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/interactive_example.cpp")
        add_executable(interactive_example examples/interactive_example.cpp)
        target_link_libraries(interactive_example PRIVATE openai_harmony)
        
        # Set RPATH for finding the shared library
        if(APPLE)
            set_target_properties(interactive_example PROPERTIES
                BUILD_RPATH "${CMAKE_BINARY_DIR}"
                INSTALL_RPATH "@loader_path/../lib"
            )
        elseif(UNIX)
            set_target_properties(interactive_example PROPERTIES
                BUILD_RPATH "${CMAKE_BINARY_DIR}"
                INSTALL_RPATH "$ORIGIN/../lib"
            )
        endif()
    else()
        message(WARNING "Interactive example source file not found")
    endif()
endif()

# Installation
install(TARGETS openai_harmony
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# Create config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/openai_harmony-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/openai_harmony-config.cmake
    INSTALL_DESTINATION lib/cmake/openai_harmony
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/openai_harmony-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/openai_harmony-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/openai_harmony-config-version.cmake
    DESTINATION lib/cmake/openai_harmony
)

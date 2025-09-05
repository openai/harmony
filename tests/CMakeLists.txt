# Google Test configuration
find_package(GTest REQUIRED)
include(GoogleTest)

# Test executable
add_executable(harmony_tests
    test_utils.cpp
    test_tiktoken.cpp
    test_encoding.cpp
    test_registry.cpp
    test_chat.cpp
    test_integration.cpp
)

target_link_libraries(harmony_tests
    PRIVATE
    openai_harmony
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)

target_include_directories(harmony_tests
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Discover tests
gtest_discover_tests(harmony_tests)

# Copy test data
file(COPY ${CMAKE_SOURCE_DIR}/test-data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

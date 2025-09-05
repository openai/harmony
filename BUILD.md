# Building OpenAI Harmony C++

This document provides instructions for building the OpenAI Harmony C++ library.

## Prerequisites

### System Requirements
- **C++ Compiler**: GCC 10+ or Clang 12+ with C++20 support
- **CMake**: Version 3.20 or later
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, or similar)

### Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev
```

#### CentOS/RHEL 8+
```bash
sudo dnf install -y \
    gcc-c++ \
    cmake \
    pkgconfig \
    openssl-devel \
    libcurl-devel \
    json-devel
```

#### Fedora
```bash
sudo dnf install -y \
    gcc-c++ \
    cmake \
    pkgconfig \
    openssl-devel \
    libcurl-devel \
    nlohmann-json-devel
```

## Building

### Basic Build

1. **Create build directory**:
   ```bash
   mkdir build
   cd build
   ```

2. **Configure with CMake**:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

3. **Build the library**:
   ```bash
   make -j$(nproc)
   ```

### Build Options

#### Debug Build
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

#### Enable Examples
```bash
cmake .. -DBUILD_EXAMPLES=ON
```

#### Enable Tests (requires Google Test)
```bash
cmake .. -DBUILD_TESTS=ON
```

#### Custom Install Prefix
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
```

#### All Options Combined
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_TESTS=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local
```

## Installation

### System-wide Installation
```bash
sudo make install
```

### Package Installation
```bash
make package
```

## Library Type: Dynamic/Shared Library

The OpenAI Harmony C++ library is built as a **dynamic/shared library** (.so on Linux, .dylib on macOS, .dll on Windows). This provides several benefits:

### Advantages
- **Smaller executable size**: Applications link to the shared library at runtime
- **Memory efficiency**: Multiple applications can share the same library in memory
- **Easy updates**: Library can be updated without recompiling applications
- **Plugin architecture**: Enables dynamic loading of functionality

### Runtime Requirements
- The shared library must be available at runtime
- Library path must be in system search paths or specified via RPATH/RUNPATH
- On Windows, the DLL must be in PATH or same directory as executable

### Library Loading
The build system automatically configures RPATH settings:
- **Linux**: Uses `$ORIGIN/../lib` for relative library paths
- **macOS**: Uses `@loader_path/../lib` for relative library paths
- **Windows**: Uses standard DLL search order

## Current Status

### Implemented Components
- ✅ **CMake Build System**: Complete with dependency management and dynamic library support
- ✅ **Header Files**: All interface definitions complete
- ✅ **Chat Module**: Complete implementation with JSON serialization
- ✅ **Basic Structure**: Project layout and build configuration
- ✅ **Dynamic Library**: Configured with proper symbol visibility and RPATH settings

### Pending Implementation
- ⏳ **Encoding Module**: Core harmony protocol implementation
- ⏳ **Tiktoken Module**: BPE tokenization engine
- ⏳ **Registry Module**: Encoding factory and management
- ⏳ **Utils Module**: Utility functions (base64, SHA, string ops)
- ⏳ **TiktokenExt Module**: Encoding loading and management

### Testing Status
- ⏳ **Unit Tests**: Test framework ready, tests need implementation
- ⏳ **Integration Tests**: End-to-end workflow testing
- ⏳ **Performance Tests**: Benchmarking against reference implementation

## Troubleshooting

### Common Issues

#### nlohmann/json Not Found
```
CMake Error: nlohmann/json not found
```

**Solution**: Install the JSON library:
```bash
# Ubuntu/Debian
sudo apt install nlohmann-json3-dev

# CentOS/RHEL
sudo dnf install json-devel

# Fedora
sudo dnf install nlohmann-json-devel
```

#### C++20 Not Supported
```
CMake Error: CMAKE_CXX_STANDARD is set to invalid value '20'
```

**Solution**: Update your compiler:
```bash
# Ubuntu
sudo apt install gcc-10 g++-10
export CC=gcc-10
export CXX=g++-10

# Or use Clang
sudo apt install clang-12
export CC=clang-12
export CXX=clang++-12
```

#### OpenSSL Not Found
```
CMake Error: Could NOT find OpenSSL
```

**Solution**: Install OpenSSL development packages:
```bash
# Ubuntu/Debian
sudo apt install libssl-dev

# CentOS/RHEL/Fedora
sudo dnf install openssl-devel
```

#### CURL Not Found
```
CMake Error: Could NOT find CURL
```

**Solution**: Install CURL development packages:
```bash
# Ubuntu/Debian
sudo apt install libcurl4-openssl-dev

# CentOS/RHEL/Fedora
sudo dnf install libcurl-devel
```

### Build Verification

#### Check Library Creation
```bash
# For dynamic library (Linux/macOS)
ls -la build/libopenai_harmony.so*

# For dynamic library (Windows)
ls -la build/openai_harmony.dll

# For dynamic library (macOS)
ls -la build/libopenai_harmony.dylib
```

#### Verify Headers
```bash
find build -name "*.hpp" | head -5
```

#### Test Basic Compilation
```bash
# If examples are enabled
./build/harmony_example

# If tests are enabled
./build/harmony_tests
```

## Development Workflow

### Adding New Source Files

1. **Create the source file** in `src/`:
   ```bash
   touch src/new_module.cpp
   ```

2. **CMake will automatically detect** the new file on next build:
   ```bash
   cd build
   make
   ```

### Adding New Headers

1. **Create the header file** in `include/openai_harmony/`:
   ```bash
   touch include/openai_harmony/new_module.hpp
   ```

2. **Update main header** `include/openai_harmony/harmony.hpp`:
   ```cpp
   #include "new_module.hpp"
   ```

### Testing Changes

1. **Incremental build**:
   ```bash
   cd build
   make -j$(nproc)
   ```

2. **Clean build** (if needed):
   ```bash
   cd build
   make clean
   make -j$(nproc)
   ```

## Next Steps

To complete the C++ implementation:

1. **Implement remaining source files**:
   - `src/encoding.cpp`
   - `src/tiktoken.cpp`
   - `src/registry.cpp`
   - `src/utils.cpp`
   - `src/tiktoken_ext.cpp`

2. **Add comprehensive tests**:
   - `tests/test_harmony.cpp`
   - `tests/test_chat.cpp`
   - `tests/test_encoding.cpp`
   - `tests/test_tiktoken.cpp`

3. **Performance optimization**:
   - Profile critical paths
   - Optimize memory allocations
   - Benchmark against reference implementation

4. **Documentation**:
   - API documentation with Doxygen
   - Usage examples and tutorials
   - Performance characteristics documentation

## Support

For build issues or questions:

1. **Check this document** for common solutions
2. **Verify dependencies** are correctly installed
3. **Check CMake version** meets minimum requirements
4. **Review compiler support** for C++20 features

The build system is designed to be flexible and handle missing components gracefully, so you can build and test incrementally as implementation progresses.

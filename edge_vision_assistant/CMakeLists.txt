cmake_minimum_required(VERSION 3.16)
project(edge_vision_assistant CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Optimization & Architecture Flags for Edge CPU Inference
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# ONNX Runtime dependency resolution:
# If ONNXRUNTIME_ROOTDIR is not specified, automatically download official release tarball
if(NOT ONNXRUNTIME_ROOTDIR)
    include(FetchContent)
    message(STATUS "Fetching pre-built ONNX Runtime C++ release for ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM_PROCESSOR})...")

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-aarch64-1.18.0.tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-osx-arm64-1.18.0.tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-osx-x86_64-1.18.0.tgz")
    elseif(WIN32)
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-win-x64-1.18.0.zip")
    else()
        message(FATAL_ERROR "Automatic ONNX Runtime download not configured for this platform. Please set -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime")
    endif()

    FetchContent_Declare(
        onnxruntime_bin
        URL ${ORT_URL}
    )
    FetchContent_MakeAvailable(onnxruntime_bin)
    set(ONNXRUNTIME_ROOTDIR ${onnxruntime_bin_SOURCE_DIR})
endif()

message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_ROOTDIR}")

# Locate headers and library
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${ONNXRUNTIME_ROOTDIR}/include
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_INCLUDE_DIR)
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h REQUIRED)
endif()

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOTDIR}/lib ${ONNXRUNTIME_ROOTDIR}/bin
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIB)
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime REQUIRED)
endif()

# Executable target
add_executable(edge_vision_assistant src/main.cpp)

target_include_directories(edge_vision_assistant PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(edge_vision_assistant PRIVATE
    ${ONNXRUNTIME_LIB}
)

# Runtime RPATH configuration so the binary locates libonnxruntime.so seamlessly
set_target_properties(edge_vision_assistant PROPERTIES
    BUILD_RPATH "\$ORIGIN;\$ORIGIN/../lib;${ONNXRUNTIME_ROOTDIR}/lib"
    INSTALL_RPATH "\$ORIGIN"
)

# Copy ONNX Runtime shared library directly to the build folder alongside the executable
if(UNIX)
    add_custom_command(TARGET edge_vision_assistant POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${ONNXRUNTIME_ROOTDIR}/lib/
            $<TARGET_FILE_DIR:edge_vision_assistant>
        COMMENT "Deploying ONNX Runtime shared library into build directory"
    )
elseif(WIN32)
    add_custom_command(TARGET edge_vision_assistant POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.dll
            $<TARGET_FILE_DIR:edge_vision_assistant>
        COMMENT "Deploying ONNX Runtime DLL into build directory"
    )
endif()

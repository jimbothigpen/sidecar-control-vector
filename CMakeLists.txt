cmake_minimum_required(VERSION 3.18)
project(sidecar_control_vector CXX)

#
# Standalone out-of-tree sidecar handler plugin: per-layer additive steering
# vector. Equivalent semantics to the legacy `--control-vector` primitive,
# exposed via the registry so users can do
#     --sidecar-vectors my_steering.cv.gguf
#
# Builds against an installed frankenturbo2 engine; produces
# libsidecar_control_vector.so.
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(LLAMA_INSTALL_PREFIX "/opt/llama-frankenturbo2-vulkan"
    CACHE PATH "frankenturbo2 install prefix (contains include/ and lib/)")

add_library(sidecar_control_vector SHARED control_vector.cpp)

target_include_directories(sidecar_control_vector PRIVATE
    "${LLAMA_INSTALL_PREFIX}/include")

find_library(LLAMA_LIB      llama      PATHS "${LLAMA_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB       ggml       PATHS "${LLAMA_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB  ggml-base  PATHS "${LLAMA_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

target_link_libraries(sidecar_control_vector PRIVATE
    ${LLAMA_LIB}
    ${GGML_LIB}
    ${GGML_BASE_LIB})

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(benchmark LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# Add current directory and csrc directory to include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}../../csrc)

# Define the .cu file separately
#set(BENCHMARK_FILES benchmark_kernel.cu)
set(BENCHMARK_FILES benchmark_kernel_cpu.cpp)

set(ABLATION_FILES ablation.cu)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# For CUDA add_executable should work just like for regular C++ code
add_executable(benchmark ${BENCHMARK_FILES})
add_executable(ablation ${ABLATION_FILES})

# Specify include directories for the target
target_include_directories(benchmark PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(ablation PRIVATE ${TORCH_INCLUDE_DIRS})

# CUDA specific properties
set_property(TARGET benchmark PROPERTY CUDA_STANDARD 14)
set_property(TARGET benchmark PROPERTY CUDA_ARCHITECTURES 80 86)
set_property(TARGET ablation PROPERTY CUDA_STANDARD 14)
set_property(TARGET ablation PROPERTY CUDA_ARCHITECTURES 80 86)

# Enable separation of compilation to speed up build for CUDA code
set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ablation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
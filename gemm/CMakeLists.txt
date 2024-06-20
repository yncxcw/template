cmake_minimum_required (VERSION 3.22.1)

project(CUDA_GEMM LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND) 
    message(STATUS "CUDA Toolkit found:")
    message(STATUS "  Version: ${CUDAToolkit_VERSION}")
    message(STATUS "  Libraries: ${CUDAToolkit_LIBRARIES}")
    message(STATUS "  Include directories: ${CUDAToolkit_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "CUDA Toolkit not found.")
endif()

add_library(gemm_lib gemm.cu)
include_directories(include)

add_executable(gemm_benchmark main.cc)
target_link_libraries(gemm_benchmark CUDA::cudart)
target_link_libraries(gemm_benchmark gemm_lib)
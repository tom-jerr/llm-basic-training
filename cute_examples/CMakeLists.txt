cmake_minimum_required(VERSION 3.18)
project(HelloCuTe LANGUAGES CXX CUDA)

# 1. 设置 C++ 标准为 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 2. 设置 GPU 架构 
# sm_80 对应 A100/RTX3090 (Ampere)
# sm_75 对应 T4/RTX2080 (Turing)
# sm_90 对应 H100 (Hopper)
set(CMAKE_CUDA_ARCHITECTURES 80) 

# 3. 包含 CUTLASS 头文件路径
# 假设 cutlass 文件夹在当前目录下
include_directories(${CMAKE_SOURCE_DIR}/cutlass/include ${CMAKE_SOURCE_DIR}/cutlass/tools/util/include)

# 4. 为了使用 CuTe 的打印功能 (print_layout 等)，通常需要包含这个
add_definitions(-DCUTE_USE_IOSTREAM)
add_executable(hello_cute test.cu)

# 分项目构建
add_subdirectory(vector_add)
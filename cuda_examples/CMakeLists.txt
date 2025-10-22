cmake_minimum_required(VERSION 3.18)
project(cuda_examples LANGUAGES CUDA CXX)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 如果你想让 host 和 device 代码都能优化
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(vector_add)

# 添加可执行文件
add_executable(vector_add_test
  test.cu
)

target_link_libraries(vector_add_test PUBLIC vector_add)

# 可选:如果 main.cu 里包含 C++ 代码
set_target_properties(vector_add_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 编译器不支持c++17 将会报错，而不是降级为 c++14 
set(CMAKE_CXX_EXTENSIONS OFF) # 是否开启编译器对 C++ 的语言特性扩展，默认为 ON
set(CMAKE_BUILD_TYPE Release) # 默认为 Debug

project(GEMM)
enable_language(C)
enable_language(CXX)

add_subdirectory(GEMM)
add_subdirectory(benchmark)
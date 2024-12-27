cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 编译器不支持c++17 将会报错，而不是降级为 c++14 
set(CMAKE_CXX_EXTENSIONS OFF) # 是否开启编译器对 C++ 的语言特性扩展，默认为 ON
# set(CMAKE_BUILD_TYPE Release) # 默认为 Debug
option(ENABLE_TESTS "Build the test" ON)

project(GEMM)
enable_language(C)
enable_language(CXX)

add_subdirectory(GEMM)

if (ENABLE_TESTS)
	# 启用测试
    enable_testing()
    add_subdirectory(test)  
endif ()
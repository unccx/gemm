file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS 
    src/*.cpp 
    include/*.hpp 
    include/*.h)

add_executable(benchmark ${SRC_FILES})
target_include_directories(benchmark PUBLIC include)

target_link_libraries(benchmark PUBLIC GEMM)

find_package(fmt REQUIRED)
target_link_libraries(benchmark PUBLIC fmt::fmt)

find_package(benchmark REQUIRED)
target_link_libraries(benchmark PUBLIC benchmark::benchmark)


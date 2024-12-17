file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS 
    src/*.cpp 
    include/*.hpp 
    include/*.h)

add_library(GEMM STATIC ${SRC_FILES})
target_include_directories(GEMM PUBLIC include)

find_package(OpenMP REQUIRED)
target_link_libraries(GEMM PUBLIC OpenMP::OpenMP_CXX)

target_compile_definitions(GEMM PUBLIC -DWITH_AVX2)
if (MSVC)
    target_compile_options(GEMM PUBLIC /fp:fast /arch:AVX)
else()
    target_compile_options(GEMM PUBLIC -ffast-math -march=native)
endif()
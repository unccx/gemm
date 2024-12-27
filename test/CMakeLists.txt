file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS 
    *.cpp 
    *.hpp 
    *.h)

add_executable(tests ${SRC_FILES})

target_link_libraries(tests PUBLIC GEMM)

find_package(Catch2 3 REQUIRED)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

find_package(fmt REQUIRED)
target_link_libraries(tests PUBLIC fmt::fmt)

include(CTest)
include(Catch)
catch_discover_tests(tests)
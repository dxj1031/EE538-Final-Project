cmake_minimum_required(VERSION 3.16)
project(int8_member1 LANGUAGES CXX)

add_library(int8_core src/quantization.cpp src/scalar_kernel.cpp)
target_include_directories(int8_core PUBLIC include)
target_compile_features(int8_core PUBLIC cxx_std_17)
set_target_properties(int8_core PROPERTIES CXX_EXTENSIONS OFF)

add_executable(int8_example examples/basic.cpp)
target_link_libraries(int8_example PRIVATE int8_core)

# INT4 proposal implementation reuses Matrix<T>; no duplicate storage code.
add_library(int4_core INTERFACE)
target_include_directories(int4_core INTERFACE include)
target_compile_features(int4_core INTERFACE cxx_std_17)
add_executable(int4_example examples/int4_basic.cpp)
target_link_libraries(int4_example PRIVATE int4_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(int8_tests tests/test_core.cpp)
    target_link_libraries(int8_tests PRIVATE int8_core)
    add_test(NAME int8_core_tests COMMAND int8_tests)
    add_executable(int4_tests tests/test_int4.cpp)
    target_link_libraries(int4_tests PRIVATE int4_core)
    add_test(NAME int4_core_tests COMMAND int4_tests)

    add_executable(int4_blocked_tests tests/test_blocked_kernel.cpp)
    target_link_libraries(int4_blocked_tests PRIVATE int4_core)
    add_test(NAME int4_blocked_kernel_tests COMMAND int4_blocked_tests)

endif()

foreach(target int8_core int8_example int8_tests int4_example int4_tests
               int4_blocked_tests)
    if(TARGET ${target})
        if(MSVC)
            target_compile_options(${target} PRIVATE /W4 /permissive-)
        else()
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        endif()
    endif()
endforeach()

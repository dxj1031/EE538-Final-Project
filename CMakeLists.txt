cmake_minimum_required(VERSION 3.16)
project(int8_member1 LANGUAGES CXX)

# Benchmarks must come from a release build; default single-config generators to Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(INT4_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer (GCC/Clang)" OFF)
if(INT4_SANITIZE AND NOT MSVC)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=address,undefined)
endif()

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

# Member 3: blocked kernel (include/int4/blocked_kernel.hpp) and benchmark driver.
add_executable(int4_bench bench/int4_bench.cpp)
target_link_libraries(int4_bench PRIVATE int4_core)
target_compile_definitions(int4_bench PRIVATE
    INT4_BENCH_BUILD_TYPE="$<CONFIG>"
    INT4_BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

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
    add_executable(int4_bench_tests tests/test_bench.cpp)
    target_link_libraries(int4_bench_tests PRIVATE int4_core)
    add_test(NAME int4_bench_support_tests COMMAND int4_bench_tests)

    # End-to-end smoke run of the driver (exit status 3 would mean a kernel mismatch).
    set(INT4_SMOKE_DIR ${CMAKE_CURRENT_BINARY_DIR}/bench_smoke)
    add_test(NAME int4_bench_smoke
             COMMAND int4_bench --quick --tuning-tiles 16,64 --csv ${INT4_SMOKE_DIR}/bench.csv
                     --dump-dir ${INT4_SMOKE_DIR}/dumps)
    set_tests_properties(int4_bench_smoke PROPERTIES FIXTURES_SETUP int4_bench_smoke_output)
    # Invalid input must give a diagnostic and a nonzero status.
    add_test(NAME int4_bench_rejects_bad_scale
             COMMAND int4_bench --quick --scales 300 --csv ${INT4_SMOKE_DIR}/rejected.csv)
    set_tests_properties(int4_bench_rejects_bad_scale PROPERTIES WILL_FAIL TRUE)

endif()

foreach(target int8_core int8_example int8_tests int4_example int4_tests
               int4_bench int4_blocked_tests int4_bench_tests)
    if(TARGET ${target})
        if(MSVC)
            target_compile_options(${target} PRIVATE /W4 /permissive-)
        else()
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        endif()
    endif()
endforeach()

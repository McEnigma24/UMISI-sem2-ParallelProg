cmake_minimum_required(VERSION 4.2)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# set(CMAKE_CXX_COMPILER g++)
# set(CMAKE_CXX_COMPILER clang++)

include_directories(_inc)
aux_source_directory(_src SOURCES)

aux_source_directory(_src TEST_SOURCES)
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*main\\.cpp")
aux_source_directory(_test TEST_SOURCES)

aux_source_directory(_src BENCH_SOURCES)
list(FILTER BENCH_SOURCES EXCLUDE REGEX ".*main\\.cpp")
aux_source_directory(_bench BENCH_SOURCES)


# MACRO FOR LOGS # - dont show the whole realpath to file

string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_SIZE)
add_definitions(-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE})

set(CONST_TARGET_NAME "TEMPLATE")

# COMPILER FLAGS #

# https://docs.oracle.com/cd/E19957-01/806-3572/Rev.html
# https://caiorss.github.io/C-Cpp-Notes/compiler-flags-options.html
# https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#index-Wfatal-errors

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra")                                                             # to see everything
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wfatal-errors -Werror=uninitialized -Werror=init-self -Werror=reorder")    # for normal coding
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -w")                                                         # for release

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

#                                                                  https://hjlebbink.github.io/x86doc/
#                                                                  - look for *.o file
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -S") # ASSEMBLY
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -S -fverbose-asm") # ASSEMBLY with comments


# set(COMPILE_FLAGS "${COMPILE_FLAGS} -m64")
# set(LINK_FLAGS "${LINK_FLAGS} -m64")


# 1 - core #
include(FetchContent)
FetchContent_Declare(
    core
    GIT_REPOSITORY https://github.com/McEnigma24/CORE_lib.git
    GIT_TAG        main # lub inny tag/commit
    GIT_SHALLOW    TRUE
    GIT_SUBMODULES_RECURSE TRUE
    UPDATE_DISCONNECTED ON
)
FetchContent_MakeAvailable(core)

set(ALL_LIBRARIES ${ALL_LIBRARIES} core)
include_directories(${core_SOURCE_DIR}/_inc)



# EXAMPLE - ssd1306 #
# include(FetchContent)
# FetchContent_Declare(
#     ssd1306
#     GIT_REPOSITORY https://github.com/armlabs/ssd1306_linux.git
#     GIT_TAG        master # lub inny tag/commit
# )
# FetchContent_MakeAvailable(ssd1306) # tutaj nie dzial, a normanie zbudowalby projekt, ale tutaj nie zrobi bo nie ma pliku CMakeLists.txt

# include_directories(${ssd1306_SOURCE_DIR})

# # Dodaj pliki źródłowe ręcznie
# file(GLOB SSD1306_SOURCES
#     ${ssd1306_SOURCE_DIR}/*.c
# )

# add_library(ssd1306 STATIC ${SSD1306_SOURCES})
# target_include_directories(ssd1306 PUBLIC ${ssd1306_SOURCE_DIR})

# set(ALL_LIBRARIES ${ALL_LIBRARIES} ssd1306) # dodajemy do glownego EXE



# Flagi buildu deklarujemy tylko wtedy, gdy to MY jesteśmy korzeniem drzewa CMake.
# option() zawsze ląduje w globalnym cache (block() tego nie zmienia - obejmuje wyłącznie
# zmienne zwykłe), więc podprojekt o tej samej nazwie dzieliłby wpis z projektem nadrzędnym.
# Ten guard zastępuje dawne prefiksowanie nazw per projekt (TEMPLATE___, CORE___).
if(PROJECT_IS_TOP_LEVEL)
    option(BUILD_LIBRARY "Build library instead of executable" OFF) # default value - could be overwritten by provided FLAG
    option(CTEST_ACTIVE "Testing is active" OFF)                    # default value - could be overwritten by provided FLAG
    option(BENCH_ACTIVE "Benchmarking is active" OFF)              # default value - could be overwritten by provided FLAG
else()
    set(BUILD_LIBRARY OFF)
    set(CTEST_ACTIVE OFF)
    set(BENCH_ACTIVE OFF)
endif()

if(BUILD_LIBRARY)
    message(WARNING "Building library")

    set(TARGET_NAME "${CONST_TARGET_NAME}")
    add_library(${TARGET_NAME} SHARED ${SOURCES})
    target_compile_definitions(${TARGET_NAME} PRIVATE LINUX)
    # set_target_properties(${TARGET_NAME} PROPERTIES ${COMPILE_FLAGS} ${LINK_FLAGS})
    target_link_libraries(${TARGET_NAME} PUBLIC ${ALL_LIBRARIES})
else()
    message(WARNING "Building exe")

    set(TARGET_NAME "${CONST_TARGET_NAME}.exe")
    add_executable(${TARGET_NAME} ${SOURCES})
    target_compile_definitions(${TARGET_NAME} PRIVATE LINUX BUILD_EXECUTABLE)
    # set_target_properties(${TARGET_NAME} PROPERTIES ${COMPILE_FLAGS} ${LINK_FLAGS})
    target_link_libraries(${TARGET_NAME} PUBLIC ${ALL_LIBRARIES})
endif()

if(CTEST_ACTIVE)
    message(WARNING "TESTING...")
    # Włączenie testów
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0  # możesz zmienić tag na nowszy gdy będzie dostępny
        GIT_SHALLOW    TRUE
        UPDATE_DISCONNECTED ON
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Dodanie testów
    set(TEST_EXE_NAME "test.texe")
    add_executable(${TEST_EXE_NAME} ${TEST_SOURCES})
    target_link_libraries(${TEST_EXE_NAME} PRIVATE gtest_main ${ALL_LIBRARIES})

    # Rejestracja testów
    include(GoogleTest)
    gtest_discover_tests(${TEST_EXE_NAME})
endif()

if(BENCH_ACTIVE)
    message(WARNING "BENCHMARKING...")

    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.5
        GIT_SHALLOW    TRUE
        UPDATE_DISCONNECTED ON
    )
    # bez tego benchmark buduje własne testy i zaciąga do nich googletesta
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR  OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)

    set(BENCH_EXE_NAME "bench.bexe")
    add_executable(${BENCH_EXE_NAME} ${BENCH_SOURCES})
    target_link_libraries(${BENCH_EXE_NAME} PRIVATE benchmark::benchmark ${ALL_LIBRARIES})
endif()
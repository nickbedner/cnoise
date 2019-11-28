cmake_minimum_required(VERSION 3.12)

set(ignoreMe "${CMAKE_CXX_COMPILER}")
set(CMAKE_C_STANDARD 11)

project(HelloNoise LANGUAGES C VERSION 0.1.0)
project(Benchmarks LANGUAGES C VERSION 0.1.0)

file(GLOB_RECURSE hello_noise_SRC
        include/*.h
        examples/hellonoise.c)
file(GLOB_RECURSE benchmarks_SRC
        include/*.h
        examples/benchmarks.c)
        
add_executable(HelloNoise ${hello_noise_SRC})
add_executable(Benchmarks ${benchmarks_SRC})

if (WIN32)
# Add OpenMP
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX /arch:AVX2")
        target_compile_options(Benchmarks PUBLIC "${CMAKE_C_FLAGS} /O2")
else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx2")
        find_package(OpenMP)
        if (OPENMP_FOUND)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        endif()
        target_compile_options(Benchmarks PUBLIC "${CMAKE_C_FLAGS} -O3")
endif()

if(APPLE)
        target_link_libraries(HelloNoise omp)
        target_link_libraries(Benchmarks omp)
endif()

if(UNIX AND NOT APPLE)
        target_link_libraries(HelloNoise m)
        target_link_libraries(Benchmarks m)
endif()

set(includeList ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(HelloNoise PUBLIC ${includeList})
target_include_directories(Benchmarks PUBLIC ${includeList})

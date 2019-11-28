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
else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx2")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
endif()

if(APPLE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor")
        target_link_libraries(HelloNoise m omp)
        target_link_libraries(Benchmarks m omp)
endif()

if(UNIX AND NOT APPLE)
        target_link_libraries(HelloNoise m)
        target_link_libraries(Benchmarks m)
endif()

set(includeList ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(HelloNoise PUBLIC ${includeList})
target_include_directories(Benchmarks PUBLIC ${includeList})

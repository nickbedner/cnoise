cmake_minimum_required(VERSION 3.12)

set(ignoreMe "${CMAKE_CXX_COMPILER}")
set(CMAKE_C_STANDARD 11)

if (WIN32)
# Add OpenMP
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX /arch:AVX2")
endif()

if(APPLE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx2 -fopenmp")
endif()

if(UNIX AND NOT APPLE)
        if(CMAKE_C_COMPILER MATCHES "clang")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx2 -fopenmp=libiomp5")
        else()
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mavx2 -fopenmp")
        endif() 
endif()

set(includeList ${CMAKE_CURRENT_SOURCE_DIR}/include)

project(HelloNoise LANGUAGES C VERSION 0.1.0)
file(GLOB_RECURSE hello_noise_SRC
        include/*.h
        examples/hellonoise.c)
add_executable(HelloNoise ${hello_noise_SRC})
target_include_directories(HelloNoise PUBLIC ${includeList})

if (WIN32)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2")
else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
endif()

project(Benchmarks LANGUAGES C VERSION 0.1.0)
file(GLOB_RECURSE benchmarks_SRC
        include/*.h
        examples/benchmarks.c)
add_executable(Benchmarks ${benchmarks_SRC})
target_include_directories(Benchmarks PUBLIC ${includeList})

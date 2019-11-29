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
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -mavx -mavx2")
        if (APPLE)
                set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include" CACHE INTERNAL "OpenMP flags for Xcode toolchain.")
                set(OpenMP_C_LIB_NAMES "omp" CACHE INTERNAL "OpenMP lib name for Xcode toolchain.")
                set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib" CACHE INTERNAL "OpenMP lib name for Xcode toolchain.")
        endif()

        find_package(OpenMP REQUIRED)
        #if (OPENMP_FOUND)
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        #endif()
endif()

if(APPLE)
        target_link_libraries(HelloNoise OpenMP::OpenMP_C)
        target_link_libraries(Benchmarks OpenMP::OpenMP_C)
endif()

if(UNIX AND NOT APPLE)
        target_link_libraries(HelloNoise m)
        target_link_libraries(Benchmarks m)
endif()

set(includeList ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(HelloNoise PUBLIC ${includeList})
target_include_directories(Benchmarks PUBLIC ${includeList})

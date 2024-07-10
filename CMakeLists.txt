cmake_minimum_required(VERSION 3.10)
project(SPM)

set(CMAKE_CXX_STANDARD 17)

# Add AVX2 flag
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-mavx2 -mfma)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

# Get the absolute path of the project directory
set(PROJECT_DIR ${CMAKE_SOURCE_DIR})

# Create the build directory if it doesn't exist
set(BUILD_DIR ${PROJECT_DIR}/build)
file(MAKE_DIRECTORY ${BUILD_DIR})

# Create the results directory if it doesn't exist
set(RESULTS_DIR ${PROJECT_DIR}/results)
file(MAKE_DIRECTORY ${RESULTS_DIR})

# Create the plots directory if it doesn't exist
set(RESULTS_DIR ${PROJECT_DIR}/plots)
file(MAKE_DIRECTORY ${RESULTS_DIR})

# Set the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR})

add_executable(SPM main.cpp
        src/utils/matrix.h
        src/sequential/sequential.cpp
        src/utils/timer.h
        src/sequential/sequential.h
        src/utils/csv.h)

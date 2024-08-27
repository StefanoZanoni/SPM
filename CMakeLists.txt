cmake_minimum_required(VERSION 3.14)
project(SPM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Python
find_package(Python3 REQUIRED)
# Use the specified Python executable
if(Python3_EXECUTABLE)
    set(Python3_EXECUTABLE ${Python3_EXECUTABLE})
else()
    message(FATAL_ERROR "Python3 not found")
endif()

find_package(MPI REQUIRED)
if(MPI_C_FOUND)
    include_directories(${MPI_C_INCLUDE_PATH})
    link_directories(${MPI_C_LIBRARIES})
else()
    message(FATAL_ERROR "MPI not found")
endif()

# Add AVX flag for GCC and Clang
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-mavx)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -mtune=native")

# Get the absolute path of the project directory
set(PROJECT_DIR ${CMAKE_SOURCE_DIR})

# Create the build directory if it doesn't exist
set(BUILD_DIR ${PROJECT_DIR}/build)
file(MAKE_DIRECTORY ${BUILD_DIR})

# Create the results directory if it doesn't exist
set(RESULTS_DIR ${PROJECT_DIR}/results)
file(MAKE_DIRECTORY ${RESULTS_DIR})

# Create the plots directory if it doesn't exist
set(PLOTS_DIR ${PROJECT_DIR}/plots)
file(MAKE_DIRECTORY ${PLOTS_DIR})

# Create the statistics directory if it doesn't exist
set(STATISTICS_DIR ${PROJECT_DIR}/statistics)
file(MAKE_DIRECTORY ${STATISTICS_DIR})

# Set the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIR})

set(FASTFLOW_DIR ${PROJECT_DIR}/fastflow-master)
include_directories(${FASTFLOW_DIR})

add_executable(sequential
        src/sequential/sequential.cpp
        test_sequential.cpp
        src/utils/csv.h
        src/utils/matrix.h
        src/utils/timer.h
        src/sequential/sequential.h
        src/sequential/seqmatrix.h)
add_executable(parallel
        src/fastflow/parallel.cpp
        test_parallel.cpp
        src/utils/csv.h
        src/utils/matrix.h
        src/utils/timer.h
        src/fastflow/parallel.h
        src/fastflow/ffmatrix.h)
add_executable(distributed
        src/mpi/distributed.cpp
        test_distributed.cpp
        src/utils/csv.h
        src/utils/matrix.h
        src/utils/timer.h
        src/mpi/mpimatrix.h
        src/mpi/distributed.h)

# Add the indicators library
include(FetchContent)
FetchContent_Declare(
        indicators
        GIT_REPOSITORY https://github.com/p-ranav/indicators.git
        GIT_TAG        v2.3
)
FetchContent_MakeAvailable(indicators)

target_link_libraries(sequential PRIVATE indicators)
target_link_libraries(parallel PRIVATE indicators)
target_link_libraries(distributed PRIVATE indicators MPI::MPI_C)

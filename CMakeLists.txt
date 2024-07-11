cmake_minimum_required(VERSION 3.14)
project(SPM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Python
find_package(Python3 REQUIRED)

find_path(NUMA_INCLUDE_DIR numa.h PATHS $ENV{HOME}/numa/include)
find_library(NUMA_LIBRARY NAMES numa PATHS $ENV{HOME}/numa/lib)

if(NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
    set(NUMA_FOUND TRUE)
    set(NUMA_LIBRARIES ${NUMA_LIBRARY})
    set(NUMA_INCLUDE_DIRS ${NUMA_INCLUDE_DIR})
else()
    set(NUMA_FOUND FALSE)
endif()

mark_as_advanced(NUMA_INCLUDE_DIR NUMA_LIBRARY)

# Use the specified Python executable
if(Python3_EXECUTABLE)
    set(Python3_EXECUTABLE ${Python3_EXECUTABLE})
endif()

# Add AVX2
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

set(FASTFLOW_DIR ${PROJECT_DIR}/fastflow-master)
include_directories(${FASTFLOW_DIR})

# Link the FastFlow library
link_directories(${FASTFLOW_DIR})

add_executable(SPM main.cpp
        src/utils/matrix.h
        src/sequential/sequential.cpp
        src/utils/timer.h
        src/sequential/sequential.h
        src/utils/csv.h
        src/fastflow/ffmatrix.h
        src/fastflow/parallel.cpp
        src/fastflow/parallel.h)

target_link_libraries(SPM ${NUMA_LIBRARIES})

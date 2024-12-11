cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(distanceQuery LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

file(GLOB GPU_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/APP/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/MyCuda/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/MyCuda/*.cpp)

# Add CUDA and C++ files to the executable
add_library(distanceQuery ${GPU_SOURCE})

# Specify the CUDA libraries to link against
target_link_libraries(distanceQuery ${CUDA_LIBRARIES})
target_include_directories(distanceQuery PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# CMakeLists.txt

cmake_minimum_required(VERSION 3.12)
project(testapp LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Find CUDA package
find_package(CUDA REQUIRED)

set(DistancePath ${CMAKE_CURRENT_SOURCE_DIR}/distanceQuery)

# Add CUDA include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(DistancePath)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/render)

add_subdirectory(distanceQuery)
add_subdirectory(render)

# Add CUDA libraries
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

# Add your source files
add_executable(testapp main.cu render.cu)

# Link CUDA libraries
target_link_libraries(testapp ${CUDA_LIBRARIES} distanceQuery render)

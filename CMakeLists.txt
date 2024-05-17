cmake_minimum_required(VERSION 3.10)
project(CUDArt CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda)

enable_language(CUDA)

file(GLOB_RECURSE SOURCES "*.cu" "*.cpp" "*.c" "*.h" "*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb)

# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

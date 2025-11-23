cmake_minimum_required(VERSION 3.20)
project(Engine CUDA CXX)

find_package(Vulkan REQUIRED)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -use_fast_math")
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glfw/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/stb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/json/include)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${Vulkan_INCLUDE_DIRS})

file(GLOB ENGINE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

file(GLOB ENGINE_HEADERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)

add_executable(${PROJECT_NAME}
    ${ENGINE_SOURCES}
    ${ENGINE_HEADERS}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    Core
    ${CUDA_LIBRARIES}
    ${Vulkan_LIBRARIES}
)

set(CMAKE_CONFIGURATION_TYPES "debug;release;relwithdebinfo" CACHE STRING "" FORCE)
if(CMAKE_BUILD_TYPE STREQUAL "debug")
    message(STATUS "debug build")
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_MODE)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
elseif(CMAKE_BUILD_TYPE STREQUAL "release")
    message(STATUS "release build")
    target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
elseif(CMAKE_BUILD_TYPE STREQUAL "relwithdebinfo")
    message(STATUS "release with debug info build")
    target_compile_definitions(${PROJECT_NAME} PRIVATE NDEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -g)
else()
    message(FATAL_ERROR "Invalid build type. Use Debug or Release.")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
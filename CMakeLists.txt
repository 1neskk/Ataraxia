cmake_minimum_required(VERSION 3.17)
project(PathTracingEngine CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

enable_language(CUDA)

find_package(Vulkan REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 --expt-relaxed-constexpr")

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
"${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Vulkan_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glfw/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glfw)
target_link_libraries(${PROJECT_NAME} glfw)

set(IMGUI_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui/backends/imgui_impl_vulkan.cpp
)

add_library(imgui STATIC ${IMGUI_SRC})
target_link_libraries(${PROJECT_NAME} imgui)

target_link_libraries(${PROJECT_NAME} ${Vulkan_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

if (UNIX AND NOT APPLE)
  add_definitions(-DPLATFORM_LINUX)
  set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda)
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
elseif(WIN32)
  add_definitions(-DPLATFORM_WINDOWS)
endif()

project(Core CUDA CXX)

enable_language(CUDA)

find_package(CUDA REQUIRED)
find_package(Vulkan REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 --expt-relaxed-constexpr")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glfw/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/stb)

set(IMGUI_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui/backends/imgui_impl_vulkan.cpp
)

add_library(ImGui STATIC
    ${IMGUI_SRC}
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glfw ${CMAKE_BINARY_DIR}/thirdparty/glfw)

include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

add_library(${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/input/Input.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Application.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Random.cu

    ${CMAKE_CURRENT_SOURCE_DIR}/include/Application.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Image.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/main.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Random.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Timer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/input/Input.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/input/KeyCodes.h

    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui/Roboto-Regular.embed
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/include/input
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

target_link_libraries(${PROJECT_NAME}
    ${Vulkan_LIBRARIES}
    ${CUDA_LIBRARIES}
    ImGui
    glfw
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

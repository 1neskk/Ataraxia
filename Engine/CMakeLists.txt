project(Engine CUDA CXX)

enable_language(CUDA)

find_package(CUDA REQUIRED)
find_package(Vulkan REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 --expt-relaxed-constexpr")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glfw/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/stb)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Vulkan_INCLUDE_DIRS})

add_executable(Engine
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Renderer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BRDF.cu

    ${CMAKE_CURRENT_SOURCE_DIR}/include/BRDF.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/DeviceCamera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Renderer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Scene.h
)

target_include_directories(Engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(Engine
    Core
    ${CUDA_LIBRARIES}
    ${Vulkan_LIBRARIES}
)

set_target_properties(Engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
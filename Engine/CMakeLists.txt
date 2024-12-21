project(Engine CUDA CXX)

enable_language(CUDA)

find_package(CUDA REQUIRED)
find_package(Vulkan REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75 --expt-relaxed-constexpr -use_fast_math")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/glfw/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/stb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/json/include)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Vulkan_INCLUDE_DIRS})

file(GLOB ENGINE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

file(GLOB ENGINE_HEADERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)

add_executable(Engine
    ${ENGINE_SOURCES}
    ${ENGINE_HEADERS}
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
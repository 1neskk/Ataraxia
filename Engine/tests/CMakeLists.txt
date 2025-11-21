file(GLOB TEST_SOURCES
    *.cpp
)

file(GLOB ENGINE_SOURCES_NO_MAIN
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu
)
list(FILTER ENGINE_SOURCES_NO_MAIN EXCLUDE REGEX ".*main\\.cpp$")

add_executable(EngineTests
    ${TEST_SOURCES}
    ${ENGINE_SOURCES_NO_MAIN}
)

target_link_libraries(EngineTests
    GTest::gtest_main
    Core
    ${CUDA_LIBRARIES}
    ${Vulkan_LIBRARIES}
)

target_include_directories(EngineTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/glm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/glfw/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/json/include
    ${CUDA_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
)

set_target_properties(EngineTests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

include(GoogleTest)
gtest_discover_tests(EngineTests)

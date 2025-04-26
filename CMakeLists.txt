cmake_minimum_required(VERSION 3.15)

project("Ataraxia")

set(CMAKE_CXX_STANDARD 20)

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DIST "${CMAKE_CXX_FLAGS_DIST} /MD")
endif()

set(ARCHITECTURE "x64")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory("Core")
add_subdirectory("Engine")

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT "Engine")

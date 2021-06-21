cmake_minimum_required(VERSION 3.12)

set(PACKAGE_NAME "PYQC")
set(PACKAGE_DESCRIPTION
    "quantum circuit simulator"
)

project(PYQC CXX)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP LIBS = ${OpenMP_CXX_LIBRARIES} FLAGS=${OpenMP_CXX_FLAGS}")
    add_compile_options("${OpenMP_CXX_FLAGS}")
    link_libraries("${OpenMP_CXX_LIBRARIES}")
else()
    message(STATUS "Unable to locate OpenMP in any standard locations.")
endif()

add_subdirectory(pyqc/backends/simulator/cpp_kernel/src)

add_subdirectory(pybind11)
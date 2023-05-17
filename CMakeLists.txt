cmake_minimum_required(VERSION 3.21)

project(
  lasso_solver
  VERSION 0.0.0
  DESCRIPTION "This is a lasso solver library"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
  message(FATAL_ERROR "Cannot make build in root directory.")
endif()

include(${PROJECT_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG d72ffb448c58b4ffb08b5ad629bc788646e2d59e
)
FetchContent_MakeAvailable(pybind11)

add_subdirectory(src)

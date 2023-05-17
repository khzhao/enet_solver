add_library(
  enet_solver
  STATIC
  enet_solver/enet_solver.cpp
)

target_include_directories(
  enet_solver
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/enet_solver
)

target_link_libraries(
  enet_solver
  PUBLIC
  CONAN_PKG::eigen
)

pybind11_add_module(bindings MODULE bindings.cpp)

target_link_libraries(
  bindings
  PUBLIC
  enet_solver
)

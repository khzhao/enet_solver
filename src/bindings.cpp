#include "enet_solver/enet_solver.h"

#include "pybind11/eigen.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

using namespace pybind11::literals;

namespace py = pybind11;

enet_solver::ENetResult solveENet(py::array_t<double> XTX,
                                  py::array_t<double> XTY, double l2,
                                  py::array_t<double> l1s, size_t max_iter,
                                  double tol) {
  py::buffer_info XTX_buf = XTX.request();
  py::buffer_info XTY_buf = XTY.request();
  py::buffer_info l1s_buf = l1s.request();

  if (XTX_buf.ndim != 2) {
    throw std::runtime_error("XTX must be 2 dimensional");
  }
  if (XTX_buf.shape[0] != XTX_buf.shape[1]) {
    throw std::runtime_error("XTX must be square");
  }
  if (XTY_buf.ndim != 1) {
    throw std::runtime_error("XTY must be 1 dimensional");
  }
  if (l1s_buf.ndim != 1) {
    throw std::runtime_error("L1 lambda array must be 1 dimensional");
  }

  size_t num_X = XTX_buf.shape[0];

  double* XTX_ptr = static_cast<double*>(XTX_buf.ptr);
  double* XTY_ptr = static_cast<double*>(XTY_buf.ptr);
  double* l1s_ptr = static_cast<double*>(l1s_buf.ptr);

  std::vector<double> l1s_vec;
  for (size_t i = 0; i < l1s_buf.shape[0]; ++i) {
    l1s_vec.push_back(l1s_ptr[i]);
  }

  auto solver =
      enet_solver::ENetSolver(XTX_ptr, XTY_ptr, XTX_buf.shape[0], l2, l1s_vec);
  auto result = solver.solve(max_iter, tol);
  return result;
}

PYBIND11_MODULE(enet_bindings, m) {
  py::class_<enet_solver::ENetResult>(m, "ENetResult")
      .def_readonly("converged", &enet_solver::ENetResult::converged)
      .def_readonly("beta_matrix", &enet_solver::ENetResult::beta_matrix);

  m.doc() = "ElasticNet Solver";
  m.def("solveENet", &solveENet, "XTX"_a, "XTY"_a, "l2"_a, "l1s"_a,
        "max_iter"_a = 1000, "tol"_a = 1e-9);
}

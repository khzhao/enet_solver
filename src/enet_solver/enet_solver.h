#pragma once

#include <Eigen/Core>

#include <vector>

namespace enet_solver {

struct ENetResult {
  using mat = Eigen::MatrixXd;

  ENetResult(size_t num_X, size_t num_l1s) {
    beta_matrix = mat::Zero(num_X, num_l1s);
  }

  mat beta_matrix;
  bool converged;
};

class ENetSolver {
public:
  using mat = Eigen::MatrixXd;
  using vec = Eigen::VectorXd;

  ENetSolver(double* XTX, double* XTY, size_t num_X, const double l2,
             std::vector<double> l1)
      : l1_(std::move(l1)) {
    std::sort(l1_.begin(), l1_.end());
    std::reverse(l1_.begin(), l1_.end());

    XTX_ = mat::Zero(num_X, num_X);
    XTY_ = vec::Zero(num_X);
    betas_ = vec::Zero(num_X);

    for (size_t i = 0; i < num_X; ++i) {
      for (size_t j = 0; j < num_X; ++j) {
        XTX_(i, j) = XTX[i * num_X + j];
      }
      XTY_(i) = XTY[i];
      // Elastic Net Criterion injected here
      XTX_(i, i) = XTX_(i, i) + l2;
    }
  }

  // (@kzhao): The Lasso Criterion
  // 0.5 * ||y - XB||_2 + lambda * ||B||_1
  // Removing constant factors
  double lossFn(vec& XTX_betas, vec& betas, const double l1) {
    return XTY_.dot(betas) + 0.5 * XTX_betas.dot(XTX_betas) +
           l1 * betas.lpNorm<1>();
  }

  double softThreshold(double rho, double lambda) {
    if (rho > lambda) {
      return rho - lambda;
    } else if (rho < -lambda) {
      return rho + lambda;
    }
    return 0;
  }

  ENetResult solve(int max_iter, double tol);

  bool coordinateDescent(const double l1, size_t max_iter, double tol);

private:
  std::vector<double> l1_;

  mat XTX_;
  vec XTY_;
  vec betas_;
};

} // namespace enet_solver

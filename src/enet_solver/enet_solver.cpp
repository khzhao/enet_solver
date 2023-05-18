#include "enet_solver.h"

namespace enet_solver {

ENetResult ENetSolver::solve(int max_iter, double tol) {
  auto result = ENetResult(XTX_.rows(), l1_.size());
  for (size_t i = 0; i < l1_.size(); ++i) {
    result.converged = coordinateDescent(l1_[i], max_iter, tol);
    result.beta_matrix.col(i) = betas_;
  }
  return result;
}

bool ENetSolver::coordinateDescent(const double l1, size_t max_iter,
                                   double tol) {
  size_t num_X = XTX_.rows();
  vec XTX_betas = XTX_ * betas_;
  double prev_loss = lossFn(XTX_betas, betas_, l1);

  for (size_t num_iters = 0; num_iters < max_iter; ++num_iters) {
    for (size_t i = 0; i < num_X; ++i) {
      double partial_residual = XTY_(i) - XTX_betas(i) + XTX_(i, i) * betas_(i);

      if (betas_(i) != 0) {
        XTX_betas -= XTX_.row(i) * betas_(i);
      }

      double rho = softThreshold(partial_residual, l1);
      betas_(i) = rho / XTX_(i, i);

      if (betas_(i) != 0) {
        XTX_betas += XTX_.row(i) * betas_(i);
      }
    }

    if (num_iters % 5 == 0) {
      double curr_loss = lossFn(XTX_betas, betas_, l1);
      if (std::abs(curr_loss - prev_loss) <= tol) {
        return true;
      }
      prev_loss = curr_loss;
    }
  }
  return false;
}

} // namespace enet_solver

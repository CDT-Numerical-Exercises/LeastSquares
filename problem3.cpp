#include <iostream>
#include <cmath>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include "csv.h"
#include "helpers.h"

const double get_y(const double x, const double w1, const double w2) {
  return std::exp(-w1*x) + std::exp(-w2*x);
}

constexpr double get_x(const int i, const double start, const double end,
               const double step) {
  return start + i*(end-start)/step;
}

void populate_y_vec(gsl_vector *y, const gsl_vector *x, const double w1,
                    const double w2) {
  for (int i = 0; i < y->size; ++i) {
    double x_val = gsl_vector_get(x, i);
    double y_val = get_y(x_val, w1, w2);
    gsl_vector_set(y, i, y_val);
  }
}

void calculate_residuals(gsl_vector *dest, const gsl_vector *y_obs,
                        const gsl_vector *y_fit) {
  // use axpy (y = ax + y) to compute the residuals
  // 1) copy y_obs to dest to avoid overwriting anything
  gsl_blas_dcopy(y_obs, dest);
  // 2) use axpy with a = -1 to get dest = dest - y_fit
  gsl_blas_daxpy(-1, y_fit, dest);
}

void calculate_jacobian(gsl_matrix *jacobian, const gsl_vector *x, const double w1,
                        const double w2) {
  for (int i = 0; i < x->size; ++i) {
    const double x_val = gsl_vector_get(x, i);
    gsl_matrix_set(jacobian, i, 0, -x_val*std::exp(-w1*x_val));
    gsl_matrix_set(jacobian, i, 1, -x_val*std::exp(-w2*x_val));
  }
}

// calculates j_dagger r without explicit calculation of an inverse
// using decomposition.
// j_dagger_r should have as many elements as the Jacobian has columns.
void calculate_j_dagger_r(gsl_vector *j_dagger_r, const gsl_matrix *jacobian, const gsl_vector *residuals) {
  // we want to obtain the vector J†r = (JT J)^-1 JTr
  // we can break this into the form Ax = b, then solve via decomposition
  //   Multiply both sides on the left by (JT J)
  //   (JT J) J†r = JTr
  //   let A = (JT J)
  //       b = JTr
  //       x = J†r
  //   Now we can solve via decomposition of A
  // Use LU decomposition for this

  // allocate work space
  gsl_vector *JTr = gsl_vector_alloc(jacobian->size2);
  gsl_matrix *JTJ = gsl_matrix_alloc(jacobian->size2, jacobian->size2);

  // populate these objects
  gsl_blas_dgemv(CblasTrans, 1, jacobian, residuals, 0, JTr);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, jacobian, jacobian, 0, JTJ);
  
  // generate the LU decomposition, clobbering JTJ in the process
  // JTJ is MxN
  // P is MxM
  int signum;
  gsl_permutation *P = gsl_permutation_alloc(JTJ->size1);
  gsl_linalg_LU_decomp(JTJ, P, &signum);

  // solve for J†r
  gsl_linalg_LU_solve(JTJ, P, JTr, j_dagger_r);

  // clean up
  gsl_vector_free(JTr);
  gsl_matrix_free(JTJ);
  gsl_permutation_free(P);
}

void gauss_newton_update(gsl_vector *weights, const gsl_vector *j_dagger_r,
                         const gsl_vector *residuals) {
  // gsl_blas_daxpy(1, j_dagger_r, weights);
  gsl_vector_add(weights, j_dagger_r);
}

int main() {
  // define initial weight vectors
  constexpr int n_weights = 2;
  const double weights_true[n_weights] = { 0.5, 0.25 };
  double weights[n_weights] = { 0.7, 0.4 };
  gsl_vector_view w_view = gsl_vector_view_array(weights, 2);
  gsl_vector *w_vec = &w_view.vector;
  
  // generate the data vectors
  constexpr double x_start = 0;
  constexpr double x_end = 10;
  constexpr double x_step = 0.1;
  constexpr int n_vals = (x_end-x_start)/x_step + 1; // +1 including the endpoint
  gsl_vector *x_vec = gsl_vector_alloc(n_vals);
  gsl_vector_linspace(x_vec, x_start, x_end);
  gsl_vector *y_vec = gsl_vector_alloc(n_vals);
  populate_y_vec(y_vec, x_vec, weights_true[0], weights_true[1]);

  // allocate working space
  gsl_vector *y_est = gsl_vector_alloc(y_vec -> size);
  gsl_vector *residuals = gsl_vector_alloc(y_vec -> size);
  // Jacobian will be NxM, where N is the length of y and M is the number of weights (2)
  gsl_matrix *jacobian = gsl_matrix_alloc(y_vec->size, n_weights);
  // j_dagger_r will have M values (number of weights)
  gsl_vector *j_dagger_r = gsl_vector_alloc(jacobian->size2);

  double current_norm = gsl_blas_dnrm2(w_vec);
  double last_norm; // we will initialise this in the loop
  double diff;
  do {
    // calculate the estimated data
    populate_y_vec(y_est, x_vec, weights[0], weights[1]);

    // calculate the residuals
    calculate_residuals(residuals, y_vec, y_est);

    // calculate the Jacobian
    calculate_jacobian(jacobian, x_vec, weights[0], weights[1]);

    // calculate J†r
    calculate_j_dagger_r(j_dagger_r, jacobian, residuals);

    // update the weights
    gauss_newton_update(w_vec, j_dagger_r, residuals);

    // update the norms
    last_norm = current_norm;
    current_norm = gsl_blas_dnrm2(w_vec);
    diff = abs(current_norm - last_norm);

    std::cout << "\rLast diff: " << std::scientific << std::setw(15) << diff;
    } while (diff > 1e-8);

  std::cout << std::endl;
  print_vector(w_vec);

  gsl_vector_free(x_vec);
  gsl_vector_free(y_vec);
  gsl_vector_free(y_est);
  gsl_vector_free(residuals);
  gsl_matrix_free(jacobian);
  gsl_vector_free(j_dagger_r);
}

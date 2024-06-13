#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gnuplot-iostream/gnuplot-iostream.h>
#include "csv.h"
#include "helpers.h"

constexpr int BasisModes = 3;
const double Frequencies[3] = {1.0, 2.0, 5.0};

constexpr double time(const int i) {
  return (double)i * 0.02;
}

double basis(const int base, const double t) {
  const double f = Frequencies[base];
  return std::sin(2.*M_PI*f*t);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Requires path to CSV file as arg." << std::endl;
    return 1;
  }

  // load the data
  // we know in advance that this will be a single column, so we can
  // make it into a vector
  gsl_matrix *Y = load_csv_to_dmatrix(argv[1]);
  gsl_vector_view y_view = gsl_matrix_column(Y, 0);
  gsl_vector *y = &y_view.vector;

  // create B
  gsl_matrix *B = gsl_matrix_calloc(y->size, BasisModes);
  //   add the basis functions
  for (int b = 0; b < BasisModes; ++b) {
    gsl_vector_view v = gsl_matrix_column(B, b);
    for (size_t i = 0; i < y->size; ++i) {
      gsl_vector_set(&v.vector, i, basis(b, time(i)));
    }
  }

  // determine the weights
  // we will solve the weights using SVD
  // gsl_linalg_SV_solve solves the equation Ax = b
  //   if we let A = (B^T B),
  //             b = B^T y, then
  //             x = w
  // so if we use this to solve (B^T B)w = (B^T y), we
  // can avoid explicitly calculating the inverse

  // first calculate A and b
  gsl_matrix *A = gsl_matrix_alloc(BasisModes, BasisModes);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, B, B, 0, A);

  gsl_vector *b = gsl_vector_alloc(BasisModes);
  gsl_blas_dgemv(CblasTrans, 1, B, y, 0, b);

  // now find the decomposition of A
  // if A is MxN, then
  // U is MxN (and replaces A)
  // S is NxN diagonal (so should be a vector of length N)
  // V is NxN
  // we also need a workspace vector with length N
  gsl_vector *S = gsl_vector_alloc(A->size2);
  gsl_matrix *V = gsl_matrix_alloc(A->size2, A->size2);
  double work_arr[A->size2];
  gsl_vector_view work_view = gsl_vector_view_array(work_arr, A->size2);
  gsl_linalg_SV_decomp(A, V, S, &work_view.vector);

  // now we can solve for the weights
  gsl_vector *w = gsl_vector_alloc(BasisModes);
  gsl_linalg_SV_solve(A, V, S, b, w);

  std::cout << "Weights: ";
  print_vector(w);
  const double w1 = gsl_vector_get(w, 0);
  const double w2 = gsl_vector_get(w, 1);
  const double w3 = gsl_vector_get(w, 2);

  // clean up what was used for calculating the weights
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(S);
  gsl_matrix_free(V);
  gsl_matrix_free(B);
  gsl_vector_free(w);

  // plot the data and the fit
  constexpr double t_min = time(0);
  const double t_max = time(y->size-1);
  
  Gnuplot gp;
  gp << "set xr [" << std::fixed << t_min << ":" << std::fixed << t_max << "]\n";
  gp << "plot '-' u 1:2 w linespoints title 'data', '' u 1:2 w lines title 'fit'\n";
  for (size_t i = 0; i < y->size; ++i) {
    double t = time(i);
    gp << t << " " << gsl_vector_get(y, i) << "\n";
  }
  gp << "e\n";

  constexpr double steps = 1000;
  const double stepwidth = (t_max-t_min)/(steps);
  for (size_t i = 0; i < steps; ++i) {
    double t = t_min + stepwidth*i;
    gp << t << " " << w1*basis(0, t) + w2*basis(1, t) + w3*basis(2, t) << "\n";
  }
  gp << "e\n";

  // clean up everything that's left
  gsl_matrix_free(Y);
  return 0;
}

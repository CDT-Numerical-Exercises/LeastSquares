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

constexpr size_t total_size(const size_t y_size) {
  return y_size*BasisModes // B
    + BasisModes*BasisModes // A
    + BasisModes // b
    + BasisModes // S
    + BasisModes*BasisModes // V
    + BasisModes // work
    + BasisModes // w
    ;
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

  // allocate a contiguous array to store the rest of our data
  std::cout << "Allocating array of " << sizeof(double)*total_size(y->size) << " bytes" << std::endl;
  // double *backing = new double[total_size(y->size)];
  double backing[total_size(y->size)];
  double *backing_head = backing;

  // create B
  gsl_matrix_view B_view = gsl_matrix_view_array(backing_head, y->size, BasisModes);
  backing_head += (y->size * BasisModes);
  gsl_matrix *B = &B_view.matrix;
  //   add the basis functions
  for (int b = 0; b < BasisModes; ++b) {
    gsl_vector_view v = gsl_matrix_column(B, b);
    for (size_t i = 0; i < y->size; ++i) {
      gsl_vector_set(&v.vector, i, basis(b, time(i)));
    }
  }

  // determine the weights
  // we will solve the weights using SVD
  // the GSL documentation is *very* explicit about not calculating
  // inverses unless you really have to, and suggests SVD as a more
  // efficient and more accurate way of solving linear equations.
  // gsl_linalg_SV_solve solves the equation Ax = b
  //   if we let A = (B^T B),
  //             b = B^T y, then
  //             x = w
  // so if we use this to solve (B^T B)w = (B^T y), we
  // can avoid explicitly calculating the inverse

  // first calculate A and b
  gsl_matrix_view A_view = gsl_matrix_view_array(backing_head, BasisModes, BasisModes);
  backing_head += (BasisModes*BasisModes);
  gsl_matrix *A = &A_view.matrix;
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, B, B, 0, A);

  gsl_vector_view b_view = gsl_vector_view_array(backing_head, BasisModes);
  backing_head += BasisModes;
  gsl_vector *b = &b_view.vector;
  gsl_blas_dgemv(CblasTrans, 1, B, y, 0, b);

  // now find the decomposition of A
  // if A is MxN, then
  // U is MxN (and replaces A)
  // S is NxN diagonal (so should be a vector of length N)
  // V is NxN
  // we also need a workspace vector with length N
  gsl_vector_view S_view = gsl_vector_view_array(backing_head, BasisModes);
  backing_head += BasisModes;
  gsl_vector *S = &S_view.vector;
  gsl_matrix_view V_view = gsl_matrix_view_array(backing_head, BasisModes, BasisModes);
  backing_head += (BasisModes*BasisModes);
  gsl_matrix *V = &V_view.matrix;
  gsl_vector_view work_view = gsl_vector_view_array(backing_head, BasisModes);
  backing_head += BasisModes;
  gsl_linalg_SV_decomp(A, V, S, &work_view.vector);

  // now we can solve for the weights
  gsl_vector_view w_view = gsl_vector_view_array(backing_head, BasisModes);
  backing_head += BasisModes;
  gsl_vector *w = &w_view.vector;
  gsl_linalg_SV_solve(A, V, S, b, w);

  std::cout << "Weights: ";
  print_vector(w);
  const double w1 = gsl_vector_get(w, 0);
  const double w2 = gsl_vector_get(w, 1);
  const double w3 = gsl_vector_get(w, 2);

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

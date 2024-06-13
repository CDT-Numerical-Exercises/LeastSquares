#include <iostream>
#include <iomanip>
#include <random>
#include "helpers.h"

namespace helpers_rng {
std::default_random_engine generator;
}

// use bounds checking to make sure this doesn't go badly.
// GSL's bounds checking should be good.
void gsl_vector_arange(gsl_vector *a, const double start, const double end, const double step) {
  double v = start;
  size_t i = 0;
  while (v <= end) {
    gsl_vector_set(a, i, v);
    v += step;
    ++i;
  }
}

void gsl_vector_linspace(gsl_vector *a, const double start, const double end,
                         const bool endpoint) {
  size_t n = a->size;
  double stepsize;
  if (endpoint) stepsize = (end-start)/(n-1); else stepsize = (end-start)/n;

  double v = start;
  for (size_t i = 0; i < n; i++) {
    gsl_vector_set(a, i, v);
    v += stepsize;
  }
}

void print_vector(const gsl_vector *a, const int width) {
  std::cout << "[ ";
  for (int i = 0; i < a->size; ++i) {
    std::cout << std::setw(width) << gsl_vector_get(a, i) << " ";
  }
  std::cout << "]" << std::endl;
}

void print_vector(const gsl_vector_view &a, const int width) {
  print_vector(&a.vector, width);
}

void print_matrix(const gsl_matrix *A, const int width) {
  for (int i = 0; i < A->size1; ++i) {
    for (int j = 0; j < A->size2; ++j) {
      std::cout << std::setw(width) << gsl_matrix_get(A, i, j) << " ";
    }
    std::cout << std::endl;
  }
}

void print_matrix(const gsl_matrix_view &A, const int width) {
  print_matrix(&A.matrix, width);
}

double any_random_real() {
  union { double f; uint64_t x; };
  do {
    x = randint<uint64_t>(0, UINT64_MAX);
  } while (!std::isfinite(f));
  return f;
}

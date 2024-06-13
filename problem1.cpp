#include <iostream>
#include <fstream>
#include <gsl/gsl_matrix.h>
#include "csv.h"
#include "helpers.h"

int main() {
  gsl_matrix *m = load_csv_to_dmatrix("test.csv");
  std::cout << m->size1 << "x" << m->size2 << std::endl;
  print_matrix(m);
  gsl_matrix_free(m);
  return 0;
}

#include <iostream>
#include <fstream>
#include <gsl/gsl_matrix.h>
#include "csv.h"
#include "helpers.h"

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

  print_vector(y);
  
  gsl_matrix_free(Y);
  return 0;
}

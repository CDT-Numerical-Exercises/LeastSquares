#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <gsl/gsl_matrix.h>

#include "csv.h"

void get_csv_dims(std::ifstream &csv_file, size_t &rows, size_t &cols, const char delimeter) {
  rows = 0;
  cols = 0;

  std::string line;
  while (std::getline(csv_file, line)) {
    // check that the line actually contains data
    std::string col;
    std::stringstream s(line);
    if (!(std::getline(s, col, delimeter))) {
      break;
    }
    ++rows;

    // if we haven't already measured the column length, keep reading
    // data as long as we can to figure out the length
    if (cols == 0) {
      cols = 1;
      while (std::getline(s, col, delimeter)) {
        ++cols;
      }
    }
  }

  // if either of these dimensions is zero, then we can assume the CSV
  // contains no data or is invalid
  if (rows == 0 || cols == 0) {
    rows = 0;
    cols = 0;
  }

  // clear any flags and seek back to the start of the file so that
  // other code doesn't see any difference in the file
  csv_file.clear();
  csv_file.seekg (0, std::ios::beg);
}

// for when the dimensions of the matrix are already known
gsl_matrix *load_csv_to_dmatrix_dims(std::ifstream &csv_file, const size_t rows,
                                     const size_t cols, const char delimeter) {
  gsl_matrix *m = gsl_matrix_calloc(rows, cols);

  int i = 0;
  int j = 0;
  std::string line;
  for (size_t i = 0; i < rows; ++i) {
    std::getline(csv_file, line);
    std::string col;
    std::stringstream s(line);

    // read all the data from the row
    for (size_t j = 0; j < cols; ++j) {
      std::getline(s, col, delimeter);
      gsl_matrix_set(m, i, j, stod(col));
    }
  }

  return m;
}

gsl_matrix *load_csv_to_dmatrix(std::filesystem::path csv_path, const char delimeter) {
  size_t rows;
  size_t cols;

  std::ifstream csv_file(csv_path);
  get_csv_dims(csv_file, rows, cols);

  gsl_matrix *m = load_csv_to_dmatrix_dims(csv_file, rows, cols, delimeter);

  csv_file.close();

  return m;
}

#ifndef CSV_H
#define CSV_H 1

#include <fstream>
#include <filesystem>
#include <gsl/gsl_matrix.h>

void get_csv_dims(std::ifstream &csv_file, size_t &rows, size_t &cols, const char delimeter = ',');
gsl_matrix *load_csv_to_dmatrix(std::filesystem::path csv_path, const char delimeter = ',');

#endif

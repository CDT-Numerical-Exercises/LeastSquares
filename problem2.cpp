#include <iostream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include <gnuplot-iostream/gnuplot-iostream.h>
#include "csv.h"
#include "helpers.h"

constexpr double get_x(int i) {
  return 0.1*i;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Requires path to CSV file as arg." << std::endl;
    return 1;
  }

  // load the data
  // we know in advance that the columns of this correspond to the
  // different basis spectra, and finally the data
  gsl_matrix *Y = load_csv_to_dmatrix(argv[1]);
  gsl_vector_view y_view = gsl_matrix_column(Y, Y->size2-1);
  gsl_vector *y = &y_view.vector;

  /*
  Let's use GSL's least squares fitting implementation this time.
  For a more manual implementation, see problem1.cpp
  
  We must define our problem in the form y = Xc
  Here, y are the observations
        c are the best fit parameters (our parameter weights)
        X is the matrix of predictor variables
  
  As an example, if we wanted to find the best fit parameters for
  
  y = a + bx + cx^2 + dx^3 + ...
  
  We would generate X as
  
  X_ij = x_i^j
  
  i.e. each column, j, of the matrix corresponds to the jth power
  of the independent variable.
  
  For some set of basis functions, b_j(x), we would generate X as
  
  X_ij = b_j(x_i)
  
  i.e. each basis function occupies a column of the matrix.

  We want both of these options for this problem. We first want to fit
  against just the basis functions, then we want to fit against the
  basis functions + two additional columns corresponding to the first
  order polynomial.
  */

  // problem (a) -- basis functions only

  // we already have the matrix of basis functions in the correct
  // format from the input data.
  // get a view into it
  gsl_matrix_view basis_view = gsl_matrix_submatrix(Y, 0, 0, Y->size1, Y->size2 - 1);

  // allocate a workspace for fitting
  // the number of observations, n, is the number of rows in Y
  // the number of parameters, p, is the number of columns - 1
  // we will allocate work to be big enough for the linear fit as well
  // this lets us avoid re-allocating it
  const int n_obs = Y->size1;
  const int n_basis = Y->size2 - 1;
  gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(n_obs, n_basis + 2);

  // allocate space for the results
  gsl_vector *c = gsl_vector_alloc(n_basis);
  gsl_matrix *cov = gsl_matrix_alloc(n_basis, n_basis);
  double chi2;

  // perform the least squares fit
  gsl_multifit_linear(&basis_view.matrix, y, c, cov, &chi2, work);

  // std::cout << "Fit parameters: [  basis1   basis2   basis3  ]" << std::endl;
  std::cout << "Fit parameters: ";
  if (n_basis < 10) {
    std::cout << "[ ";
    for (int i = 0; i < n_basis; ++i) {
      std::cout << " basis" << i + 1 << "  ";
    }
    std::cout << "]" << std::endl;
    std::cout << "                ";
  }
  print_vector(c, 8);

  // plot the fit
  Gnuplot gp;
  gp << "set multiplot layout 2, 1\n";
  gp << "plot '-' u 1:2 w points title 'data', '' u 1:2 w lines title 'fit'\n";
  for (int i = 0; i < y->size; ++i) {
    double x = get_x(i);
    gp << x << " " << gsl_vector_get(y, i) << "\n";
  }
  gp << "e\n";

  for (int i = 0; i < n_obs; ++i) {
    double x = get_x(i);
    double y = 0;
    for (int j = 0; j < n_basis; ++j) {
      y += gsl_matrix_get(&basis_view.matrix, i, j)*gsl_vector_get(c, j);
    }
    gp << x << " " << y << "\n";
  }
  gp << "e\n";

  // clean up
  gsl_vector_free(c);
  gsl_matrix_free(cov);

  // fit with the linear parameters as well
  // create a matrix for the parameters
  gsl_matrix *X = gsl_matrix_alloc(n_obs, n_basis + 2);

  // copy in the basis vectors
  gsl_matrix_view X_basis_section = gsl_matrix_submatrix(X, 0, 0, n_obs, n_basis);
  gsl_matrix_memcpy(&X_basis_section.matrix, &basis_view.matrix);

  // add the polynomial columns
  // first is the constant term
  gsl_vector_view X_const = gsl_matrix_column(X, n_basis);
  gsl_vector_set_all(&X_const.vector, 1);
  // then the linear term
  gsl_vector_view X_lin = gsl_matrix_column(X, n_basis + 1);
  for (int i = 0; i < n_obs; ++i) {
    gsl_vector_set(&X_lin.vector, i, get_x(i));
  }

  // allocate space for the results
  /* gsl_vector */ c = gsl_vector_alloc(n_basis+2);
  /* gsl_matrix */ cov = gsl_matrix_alloc(n_basis+2, n_basis+2);
  // double chi2;
  
  // perform the least squares fit
  gsl_multifit_linear(X, y, c, cov, &chi2, work);
  
  // std::cout << "Fit parameters: [  basis1   basis2   basis3  constant  linear  ]" << std::endl;
  std::cout << "Fit parameters: ";
  if (n_basis < 10) {
    std::cout << "[ ";
    for (int i = 0; i < n_basis; ++i) {
      std::cout << " basis" << i + 1 << "  ";
    }
    std::cout << "constant  linear  ]" << std::endl;
    std::cout << "                ";
  }
  print_vector(c, 8);

  // plot the fit
  gp << "plot '-' u 1:2 w points title 'data', '' u 1:2 w lines title 'fit'\n";
  for (int i = 0; i < y->size; ++i) {
    double x = get_x(i);
    gp << x << " " << gsl_vector_get(y, i) << "\n";
  }
  gp << "e\n";

  for (int i = 0; i < n_obs; ++i) {
    double x = get_x(i);
    double y = 0;
    for (int j = 0; j < n_basis; ++j) {
      y += gsl_matrix_get(&basis_view.matrix, i, j)*gsl_vector_get(c, j);
    }
    y += gsl_vector_get(c, n_basis) + gsl_vector_get(c, n_basis+1)*x;
    gp << x << " " << y << "\n";
  }
  gp << "e\n";
  gp << "unset multiplot\n";

  // clean up
  gsl_vector_free(c);
  gsl_matrix_free(cov);
  gsl_multifit_linear_free(work);
  gsl_matrix_free(X);
  gsl_matrix_free(Y);
  
  return 0;
}

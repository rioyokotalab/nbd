#pragma once

#include <complex>
#include <cstdint>

class Cell;
class CSR;

class Matrix {
public:
  double* A; int64_t M, N, LDA; };

class EvalDouble {
public:
  virtual double operator()(double d) const = 0;
};

class Laplace3D : public EvalDouble {
public:
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  double operator()(double d) const override {
    return d == 0. ? singularity : (1. / d);
  }
};

void gen_matrix(const EvalDouble& Eval, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda);

void mul_AS(const Matrix* RU, const Matrix* RV, Matrix* A);

void compute_basis(const EvalDouble& eval, int64_t rank_max, 
  int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t N, const double bodies[]);

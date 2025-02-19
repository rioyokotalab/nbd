#pragma once

#include <complex>
#include <cstdint>
#include <vector>

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

class MatrixAcc {
public:
  int64_t M, N;
  MatrixAcc(int64_t M, int64_t N) : M(M), N(N) {};
  virtual void op_Aij_mulB(char opA, int64_t mA, int64_t nA, int64_t nrhs, int64_t iA, int64_t jA, const std::complex<double>* B_in, int64_t strideB, std::complex<double>* C_out, int64_t strideC) const = 0;
};
  
class DenseZMat : public MatrixAcc {
public:
  std::complex<double>* A;
  DenseZMat(int64_t M, int64_t N);
  ~DenseZMat();
  void op_Aij_mulB(char opA, int64_t mA, int64_t nA, int64_t nrhs, int64_t iA, int64_t jA, const std::complex<double>* B_in, int64_t strideB, std::complex<double>* C_out, int64_t strideC) const override;
};
  
class LowRankMatrix {
public:
  int64_t M, N, rank;
  std::vector<std::complex<double>> U, V;
  std::vector<double> S;

  LowRankMatrix(int64_t m, int64_t n, const MatrixAcc& eval, int64_t iA, int64_t jA);
  LowRankMatrix(double epi, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, const MatrixAcc& eval, int64_t iA, int64_t jA);
  void matVecMul(int64_t mA, int64_t nA, int64_t nrhs, int64_t iA, int64_t jA, const std::complex<double>* B_in, int64_t strideB, std::complex<double>* C_out, int64_t strideC) const;
};

class Hmatrix {
public:
  std::vector<int64_t> yOffsets;
  std::vector<int64_t> xOffsets;
  std::vector<LowRankMatrix> L;
  
  Hmatrix(const MatrixAcc& eval, int64_t lbegin, int64_t len, const Cell cells[], const CSR& Near);
  Hmatrix(double epi, const MatrixAcc& eval, int64_t rank, int64_t p, int64_t niters, int64_t lbegin, int64_t len, const Cell cells[], const CSR& Far);
  void matVecMul(int64_t mA, int64_t nA, int64_t nrhs, int64_t iA, int64_t jA, const std::complex<double>* B_in, int64_t strideB, std::complex<double>* C_out, int64_t strideC) const;
};

void Zrsvd(double epi, int64_t m, int64_t n, int64_t* k, int64_t p, int64_t niters, const MatrixAcc& A, int64_t iA, int64_t jA, double* S, std::complex<double>* U, int64_t ldu, std::complex<double>* V, int64_t ldv);

void gen_matrix(const EvalDouble& Eval, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda);

void mmult(char ta, char tb, const Matrix* A, const Matrix* B, Matrix* C, double alpha, double beta);

void mul_AS(const Matrix* RU, const Matrix* RV, Matrix* A);

void compute_basis(const EvalDouble& eval, int64_t rank_max, 
  int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t Nclose, const double Cbodies[], int64_t Nfar, const double Fbodies[]);

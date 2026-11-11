#pragma once

#include <complex>
#include <vector>
#include <mpi.h>

class DenseMatrix {
public:
  long long N;
  std::vector<std::complex<double>> A;
  std::vector<long long> rowOffsets;
  int prime_rank; MPI_Comm prime, sub;
  DenseMatrix(long long localM, long long N, MPI_Comm prime, MPI_Comm sub);

  void matVecA(const DenseMatrix& X, std::complex<double>* B) const;
  void adjointA(DenseMatrix& B) const;
  void AijExtractBlock(long long mA, long long nA, long long iA, long long jA, std::complex<double>* A_out, long long stride);
  void compress(double epi, std::vector<long long>& recipe, std::vector<std::complex<double>>& basis);
  void copyFromA(const DenseMatrix& A);
};


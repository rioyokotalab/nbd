#pragma once

#include <complex>
#include <vector>
#include <mpi.h>

class DenseMatrix {
private:
  std::vector<std::complex<double>> A;

public:
  std::vector<long long> rowOffsets;
  long long N; int prime_rank; MPI_Comm prime, sub;
  DenseMatrix(long long localM, long long N, MPI_Comm prime, MPI_Comm sub);

  void matVecA(const DenseMatrix& matA, std::complex<double>* B) const;
  void adjointA(DenseMatrix& B);
  void compress(double epi, std::vector<long long>& recipe, std::vector<std::complex<double>>& basis);
  void copyFromA(const DenseMatrix& matA);

  void AijSetBlock(long long mA, long long nA, long long iA, long long jA, const std::complex<double>* A_in, long long stride);
  void AijExtractBlock(long long mA, long long nA, long long iA, long long jA, std::complex<double>* A_out, long long stride);

};


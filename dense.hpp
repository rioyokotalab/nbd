#pragma once

#include <complex>
#include <vector>
#include <mpi.h>

class DenseMatrix {
public:
  long long RowStart, RowEnd, N;
  std::vector<std::complex<double>> A;
  std::vector<long long> rowOffsets;
  MPI_Comm world;
  DenseMatrix(long long RowStart, long long RowEnd, long long N, MPI_Comm world);

  void Aij_block(long long mA, long long nA, long long iA, long long jA, std::complex<double>* A_out, long long stride) const;
  void matVecA(std::complex<double>* X) const;
  void adjointA(std::complex<double>* A) const;
};


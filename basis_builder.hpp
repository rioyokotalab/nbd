
#pragma once

#include <vector>
#include <complex>

class CSR;
class Cell;
class DenseMatrix;
class H2MatrixComm;

class BasisBuilder {
public:
  long long lbegin;
  long long lend;
  std::vector<long long> Ncur;
  std::vector<long long> N;
  std::vector<long long> M;
  std::vector<long long> Ranks;

  std::vector<std::vector<std::pair<long long, long long>>> fcells;
  std::vector<std::vector<std::complex<double>>> fmatrix;
  std::vector<std::vector<std::complex<double>>> basis;
  std::vector<std::vector<std::complex<double>>> basis_full;

  BasisBuilder(long long lbegin, long long len, const Cell cells[], const CSR& Far);
  BasisBuilder(long long lbegin, long long len, const Cell cells[], const CSR& Far, const BasisBuilder& upper);

  void evalLeafFMatrix(double epi, const DenseMatrix& matA, const Cell cells[]);
  void evalInnerFMatrix(double epi, const BasisBuilder& lower, const Cell cells[], const H2MatrixComm& comm);
};


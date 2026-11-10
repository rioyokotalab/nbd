#pragma once

#include <complex>
#include <vector>

class DenseMatrix;
class BasisBuilder;
class CSR;
class Cell;
class H2MatrixComm;

class H2Matrix {
public:
  std::vector<long long> URowPrefixes;
  std::vector<long long> UColPrefixes;
  std::vector<long long> VRowPrefixes;
  std::vector<long long> VColPrefixes;

  std::vector<long long> UPrefixes;
  std::vector<long long> VPrefixes;
  std::vector<long long> APrefixes;
  std::vector<long long> CPrefixes;

  std::vector<std::complex<double>> U;
  std::vector<std::complex<double>> V;
  std::vector<std::complex<double>> A;
  std::vector<std::complex<double>> C;
  
  void eval(Cell cells[], const BasisBuilder& row_side, const BasisBuilder& col_side, const CSR& Far_local, const CSR& Far, const H2MatrixComm& comm);
  void evalLeafA(double epi, DenseMatrix& matA, Cell cells[], const CSR& Far_local, const H2MatrixComm& comm);

  void matVecUpwardPass(const std::complex<double>* X_in, std::complex<double>* Z, const H2MatrixComm& comm) const;
  void matVecHorizontalandDownwardPass(std::complex<double>* Y_out, const std::complex<double>* Z, std::complex<double>* Y, const CSR& Far) const;
  void matVecLeafHorizontalPass(std::complex<double>* X_io, const std::complex<double>* Z, std::complex<double>* Y, const CSR& Far, const H2MatrixComm& comm) const;

};


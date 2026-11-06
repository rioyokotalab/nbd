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
  std::vector<long long> Dims;
  std::vector<long long> DimsLr;
  std::vector<long long> DimsOffsets;
  std::vector<long long> DimsLrOffsets;

  std::vector<long long> UOffsets;
  std::vector<long long> AOffsets;
  std::vector<long long> COffsets;

  std::vector<std::complex<double>> U;
  std::vector<std::complex<double>> V;
  std::vector<std::complex<double>> A;
  std::vector<std::complex<double>> C;
  
  void eval(const Cell cells[], const CSR& Far, const BasisBuilder& row_side, const BasisBuilder& col_side, const CSR& Far_local, const H2MatrixComm& comm);
  void evalLeafA(const DenseMatrix& matA, const Cell cells[], const H2MatrixComm& comm);

  void matVecUpwardPass(const std::complex<double>* X_in, std::complex<double>* Z, const H2MatrixComm& comm) const;
  void matVecHorizontalandDownwardPass(std::complex<double>* Y_out, const std::complex<double>* Z, std::complex<double>* Y, const CSR& Far) const;
  void matVecLeafHorizontalPass(std::complex<double>* X_io, const std::complex<double>* Z, std::complex<double>* Y, const CSR& Far, const H2MatrixComm& comm) const;

};


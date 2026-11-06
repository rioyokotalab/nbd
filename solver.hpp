#pragma once

#include <complex>
#include <vector>
#include <array>
#include <mpi.h>

class DenseMatrix;
class BasisBuilder;
class CSR;
class Cell;
class H2MatrixComm;
class H2Matrix;

class H2MatrixSolver {
public:
  long long levels;
  std::vector<H2MatrixComm> comm;
  std::vector<CSR> Far_local;
  MPI_Comm world;

  std::vector<std::vector<long long>> childOffsets;
  std::vector<std::vector<std::array<long long, 4>>> lowerIndA;
  std::vector<std::vector<std::array<long long, 4>>> lowerIndC;
  std::vector<H2Matrix> A;

  std::pair<long long, long long> local_bodies;

  H2MatrixSolver(const std::vector<Cell>& cells, const CSR& Near, const CSR& Far, long long levels, MPI_Comm world);
  void evalMatVecA(const DenseMatrix& matA, double epi, const Cell cells[], const CSR& Far);
  void allocRHSwork(std::vector<long long>& iwork, std::vector<std::complex<double>>& cwork) const;

  void matVecMul(std::complex<double> X[]) const;
  void matVecMul(std::complex<double> X[], const long long iwork[], std::complex<double> cwork[]) const;
  double solveRelErr(const std::complex<double> X[], const std::complex<double> ref[]);

  void free_all_comms();
};

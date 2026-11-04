
#pragma once

#include <vector>
#include <cstdint>

class Matrix;
class Cell;

class Base {
public:
  int64_t dimR, dimS, dimN;
  std::vector<int64_t> Dims, DimsLr;
  Matrix *Uo, *R;
  double *M, *U, *R_cpu; 
};

class EvalDouble;
class CSR;
class CellComm;

void buildBasis(const EvalDouble& eval, Base basis[], Cell* cells, int64_t levels, const CellComm* comm, const double* bodies, int64_t nbodies, int64_t mrank);

void basis_free(Base* basis);


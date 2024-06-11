
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

class Matrix;
class Cell;

class Base {
public:
  int64_t dimR, dimS, dimN;
  std::vector<int64_t> Dims, DimsLr;
  Matrix *Uo, *R;
  double *M, *U, *R_cpu;
  std::vector<std::pair<int64_t, int64_t>> LocalChild, LocalParent;
};

class EvalDouble;
class CSR;
class ColCommMPI;

void buildBasis(const EvalDouble& eval, Base basis[], Cell* cells, const CSR* rel_near, int64_t mrank, int64_t leaf, int64_t branches, int64_t levels,
  const ColCommMPI* comm, const double* bodies, int64_t nbodies, int64_t sp_pts);

void basis_free(Base* basis);


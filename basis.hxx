
#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

struct Base { 
  int64_t dimR, dimS, dimN;
  std::vector<int64_t> Dims, DimsLr;
  struct Matrix *Uo, *R;
  double *M_gpu, *M_cpu, *U_gpu, *U_cpu, *R_gpu, *R_cpu; 
};

struct EvalDouble;

void buildBasis(const EvalDouble& eval, struct Base basis[], struct Cell* cells, const CSR* rel_near, int64_t levels,
  const struct CellComm* comm, const double* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts, int64_t alignment);

void basis_free(struct Base* basis);


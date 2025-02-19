
#include <umv.hpp>
#include <basis.hpp>
#include <build_tree.hpp>
#include <gpu_linalg.hpp>
#include <linalg.hpp>
#include <comm.hpp>

#include <cstdio>
#include <cstring>
#include <cmath>

#include <cstdlib>
#include <algorithm>
#include <Eigen/Dense>

class RightHandSides { public: Matrix *X, *Xc, *Xo, *B; };

void allocRightHandSidesMV(RightHandSides rhs[], const Base base[], const CellComm comm[], int64_t levels) {
  for (int64_t l = 0; l <= levels; l++) {
    int64_t len;
    content_length(NULL, &len, NULL, &comm[l]);
    int64_t len_arr = len * 4;
    Matrix* arr_m = (Matrix*)calloc(len_arr, sizeof(Matrix));
    rhs[l].X = arr_m;
    rhs[l].B = &arr_m[len];
    rhs[l].Xo = &arr_m[len * 2];
    rhs[l].Xc = &arr_m[len * 3];

    int64_t len_data = len * base[l].dimN * 2;
    double* data = (double*)calloc(len_data, sizeof(double));
    for (int64_t i = 0; i < len; i++) {
      std::pair<int64_t, int64_t> p = comm[l].LocalParent[i];
      arr_m[i] = (Matrix) { &data[i * base[l].dimN], base[l].dimN, 1, base[l].dimN }; // X
      arr_m[i + len] = (Matrix) { &data[len * base[l].dimN + i * base[l].dimN], base[l].dimN, 1, base[l].dimN }; // B

      double* x_next = (l == 0) ? NULL : &rhs[l - 1].X[0].A[std::get<1>(p) * base[l].dimS + std::get<0>(p) * base[l - 1].dimN];
      arr_m[i + len * 2] = (Matrix) { x_next, base[l].dimS, 1, base[l].dimS }; // Xo

      double* b_next = (l == 0) ? NULL : &rhs[l - 1].B[0].A[std::get<1>(p) * base[l].dimS + std::get<0>(p) * base[l - 1].dimN];
      arr_m[i + len * 3] = (Matrix) { b_next, base[l].dimS, 1, base[l].dimS }; // Xc
    }
  }
}

void rightHandSides_free(RightHandSides* rhs) {
  double* data = rhs->X[0].A;
  if (data)
    free(data);
  free(rhs->X);
}

void matVecA(const Node A[], const Base basis[], const CSR rels_near[], double* X, const CellComm comm[], int64_t levels) {
  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &comm[levels]);

  std::vector<RightHandSides> rhs(levels + 1);
  allocRightHandSidesMV(&rhs[0], basis, comm, levels);
  memcpy(rhs[levels].X[lbegin].A, X, llen * basis[levels].dimN * sizeof(double));

  for (int64_t i = levels; i > 0; i--) {
    int64_t ibegin = 0, iboxes = 0, xlen = 0;
    content_length(&iboxes, &xlen, &ibegin, &comm[i]);

    level_merge_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);
    neighbor_bcast_cpu(rhs[i].X[0].A, basis[i].dimN, &comm[i]);
    dup_bcast_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);

    for (int64_t j = 0; j < iboxes; j++)
      mmult('T', 'N', &basis[i].Uo[j + ibegin], &rhs[i].X[j + ibegin], &rhs[i].Xo[j + ibegin], 1., 0.);
  }

  level_merge_cpu(rhs[0].X[0].A, basis[0].dimN, &comm[0]);
  dup_bcast_cpu(rhs[0].X[0].A, basis[0].dimN, &comm[0]);
  mmult('N', 'N', &A[0].A[0], &rhs[0].X[0], &rhs[0].B[0], 1., 0.);

  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = 0, iboxes = 0;
    content_length(&iboxes, NULL, &ibegin, &comm[i]);
    for (int64_t j = 0; j < iboxes; j++)
      mmult('N', 'N', &basis[i].Uo[j + ibegin], &rhs[i].Xc[j + ibegin], &rhs[i].B[j + ibegin], 1., 0.);
    for (int64_t y = 0; y < iboxes; y++)
      for (int64_t xy = rels_near[i].RowIndex[y]; xy < rels_near[i].RowIndex[y + 1]; xy++) {
        int64_t x = rels_near[i].ColIndex[xy];
        mmult('N', 'N', &A[i].A[xy], &rhs[i].X[x], &rhs[i].B[y + ibegin], 1., 1.);
      }
  }
  memcpy(X, rhs[levels].B[lbegin].A, llen * basis[levels].dimN * sizeof(double));
  for (int64_t i = 0; i <= levels; i++)
    rightHandSides_free(&rhs[i]);
}


void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX) {
  double err[2] = { 0., 0. };
  Eigen::Map<const Eigen::VectorXd> vecX(X, lenX);
  Eigen::Map<const Eigen::VectorXd> vecR(ref, lenX);
  err[0] = (vecX - vecR).dot(vecX - vecR);
  err[1] = vecR.dot(vecR);

  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = sqrt(err[0] / err[1]);
}


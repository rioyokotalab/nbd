
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <array>
#include <vector>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include "cblas.h"
#include "lapacke.h"
#include "mpi.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "nccl.h"

extern "C" void dgemv_(const char*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*) noexcept;

class Matrix {
public:
  double* A; int64_t M, N, LDA; };

class Base {
public:
  int64_t dimR, dimS, dimN;
  std::vector<int64_t> Dims, DimsLr;
  Matrix *Uo, *R;
  double *M, *U, *R_cpu; 
};

class Cell {
public:
  int64_t Child[2], Body[2], Level;
  double R[3], C[3];
};

class CellComm {
public:
  int64_t Proc;
  std::vector<std::pair<int64_t, int64_t>> ProcBoxes;
  std::vector<std::pair<int64_t, int64_t>> LocalChild, LocalParent;
  
  std::vector<std::pair<int, MPI_Comm>> Comm_box;
  MPI_Comm Comm_share, Comm_merge;

  cudaStream_t stream;
  std::vector<std::pair<int, ncclComm_t>> NCCL_box;
  ncclComm_t NCCL_merge, NCCL_share;
  
  int64_t iLocal(int64_t iglobal) const;
  int64_t iGlobal(int64_t ilocal) const;
};


class CSR {
public:
  int64_t M, N;
  std::vector<int64_t> RowIndex;
  std::vector<int64_t> ColIndex;

  int64_t lookupIJ(int64_t i, int64_t j) const {
    if (j < 0 || j >= N)
    { return -1; }
    const int64_t* row = &ColIndex[0];
    int64_t jbegin = RowIndex[j];
    int64_t jend = RowIndex[j + 1];
    const int64_t* row_iter = &row[jbegin];
    while (row_iter != &row[jend] && *row_iter != i)
      row_iter = row_iter + 1;
    int64_t k = row_iter - row;
    return (k < jend) ? k : -1;
  }
};

class EvalDouble {
public:
  virtual double operator()(double d) const = 0;
};

class Laplace3D : public EvalDouble {
public:
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  double operator()(double d) const override {
    return d == 0. ? singularity : (1. / d);
  }
};

class BatchedFactorParams {
public:
  int64_t N_r, N_s, N_upper, L_diag, L_nnz, L_lower, L_rows, L_tmp;
  const double** U_r, **U_s, **V_x, **A_sx, **U_i, *U_d0;
  double** A_x, **A_s, **A_l, **B_x, **A_upper, *V_data, *A_data;
  int* ipiv, *info;
  double** X_d, *X_data, *Xc_d0, *X_d0;
  int64_t Kfwd, Kback;
  double** Xo_Y, **Xc_Y, **Xc_X, **Xo_I;
  double** ACC_Y, **ACC_X, **ACC_I, *ACC_data;
  double** ONE_LIST, *ONE_DATA;
};

class Node {
public:
  int64_t lenA, lenS;
  Matrix *A, *S;
  double* A_ptr, *A_buf, *X_ptr, *X_buf;
  double* U_ptr, *U_buf;
  int64_t sizeA, sizeU;
  BatchedFactorParams params; 
};

template <typename T>
void memcpy2d(T* dst, const T* src, int64_t rows, int64_t cols, int64_t ld_dst, int64_t ld_src) {
  if (rows == ld_dst && rows == ld_src)
    std::copy(src, src + rows * cols, dst);
  else 
    for (int64_t i = 0; i < cols; i++)
      std::copy(&src[i * ld_src], &src[i * ld_src + rows], &dst[i * ld_dst]);
}

void mul_AS(const Matrix* RU, const Matrix* RV, Matrix* A) {
  if (A->M > 0 && A->N > 0) {
    std::vector<double> tmp(A->M * A->N);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->M, A->N, A->M, 1., RU->A, RU->LDA, A->A, A->LDA, 0., &tmp[0], A->M);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->M, A->N, A->N, 1., &tmp[0], A->M, RV->A, RV->LDA, 0., A->A, A->LDA);
  }
}

void gen_matrix(const EvalDouble& Eval, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * m]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * n]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    int64_t ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      int64_t iy = std::distance(bi3, &i);
      double x = i[0] - j[0];
      double y = i[1] - j[1];
      double z = i[2] - j[2];
      double d = std::sqrt(x * x + y * y + z * z);
      Aij[iy + ix * lda] = Eval(d);
    });
  });
}

void compute_basis(const EvalDouble& eval, int64_t rank_max, int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t N, const double bodies[]) {

  if (M > 0 && N > 0) {
    int64_t ldm = std::max(M, N);
    std::vector<double> Aall(M * ldm, 0.), U(M * M), S(M * 2);
    std::vector<int32_t> ipiv(M, 0);
    gen_matrix(eval, N, M, bodies, Xbodies, &Aall[0], ldm);

    LAPACKE_dgeqp3(LAPACK_COL_MAJOR, N, M, &Aall[0], ldm, &ipiv[0], &S[0]);
    LAPACKE_dlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, 0., 0., &Aall[1], ldm);

    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'A', M, M, &Aall[0], ldm, &S[0], NULL, M, &U[0], M, &S[M]);
    int64_t rank = rank_max <= 0 ? M : std::min(rank_max, M);    
    if (rank > 0) {
      if (rank < M)
        LAPACKE_dgesv(LAPACK_COL_MAJOR, rank, M - rank, &U[0], M, (int32_t*)&S[0], &U[rank * M], M);
      LAPACKE_dlaset(LAPACK_COL_MAJOR, 'F', rank, rank, 0., 1., &U[0], M);
    }

    std::vector<double> Xpiv(M * 3);
    for (int64_t i = 0; i < M; i++) {
      int64_t piv = (int64_t)ipiv[i] - 1;
      if (rank > 0)
      std::copy(&U[i * M], &U[i * M + rank], &Aall[piv * M]);
      std::copy(&Xbodies[piv * 3], &Xbodies[piv * 3 + 3], &Xpiv[i * 3]);
    }
    std::copy(Xpiv.begin(), Xpiv.end(), Xbodies);

    if (rank > 0) {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, rank, M, 1., A, LDA, &Aall[0], M, 0., &U[0], M);
      LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'O', M, rank, &U[0], M, &S[0], A, LDA, &U[0], M, &S[M]);

      for (int64_t i = 0; i < rank; i++)
        for (int64_t j = 0; j < rank; j++)
          A[(M + i) * LDA + j] = S[j] * U[i * M + j];
    }
  }
}

void content_length(int64_t* local, int64_t* neighbors, int64_t* local_off, const CellComm* comm) {
  int64_t slen = 0, offset = -1, len_self = -1;
  for (int64_t i = 0; i < (int64_t)comm->ProcBoxes.size(); i++) {
    if (i == comm->Proc)
    { offset = slen; len_self = std::get<1>(comm->ProcBoxes[i]); }
    slen = slen + std::get<1>(comm->ProcBoxes[i]);
  }
  if (local)
    *local = len_self;
  if (neighbors)
    *neighbors = slen;
  if (local_off)
    *local_off = offset;
}

int64_t neighbor_bcast_sizes_cpu(int64_t* data, const CellComm* comm) {
  int64_t max = 0;
  if (comm->Comm_box.size() > 0 || comm->Comm_share != MPI_COMM_NULL) {
    int64_t y = 0;
    for (int64_t p = 0; p < (int64_t)comm->Comm_box.size(); p++) {
      int64_t llen = std::get<1>(comm->ProcBoxes[p]);
      int64_t* loc = &data[y];
      MPI_Bcast(loc, llen, MPI_INT64_T, std::get<0>(comm->Comm_box[p]), std::get<1>(comm->Comm_box[p]));
      y = y + llen;
    }
    content_length(NULL, &y, NULL, comm);
    if (comm->Comm_share != MPI_COMM_NULL)
      MPI_Bcast(data, y, MPI_DOUBLE, 0, comm->Comm_share);

    for (int64_t i = 0; i < y; i++)
      max = std::max(max, data[i]);
    MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  }
  return max;
}

void neighbor_bcast_cpu(double* data, int64_t seg, const CellComm* comm) {
  if (comm->Comm_box.size() > 0) {
    int64_t y = 0;
    for (int64_t p = 0; p < (int64_t)comm->Comm_box.size(); p++) {
      int64_t llen = std::get<1>(comm->ProcBoxes[p]) * seg;
      double* loc = &data[y];
      MPI_Bcast(loc, llen, MPI_DOUBLE, std::get<0>(comm->Comm_box[p]), std::get<1>(comm->Comm_box[p]));
      y = y + llen;
    }
  }
}

void neighbor_reduce_cpu(double* data, int64_t seg, const CellComm* comm) {
  if (comm->Comm_box.size() > 0) {
    int64_t y = 0;
    for (int64_t p = 0; p < (int64_t)comm->Comm_box.size(); p++) {
      int64_t llen = std::get<1>(comm->ProcBoxes[p]) * seg;
      double* loc = &data[y];
      if (p == comm->Proc)
        MPI_Reduce(MPI_IN_PLACE, loc, llen, MPI_DOUBLE, MPI_SUM, std::get<0>(comm->Comm_box[p]), std::get<1>(comm->Comm_box[p]));
      else
        MPI_Reduce(loc, loc, llen, MPI_DOUBLE, MPI_SUM, std::get<0>(comm->Comm_box[p]), std::get<1>(comm->Comm_box[p]));
      y = y + llen;
    }
  }
}

void level_merge_cpu(double* data, int64_t len, const CellComm* comm) {
  if (comm->Comm_merge != MPI_COMM_NULL) {
    MPI_Allreduce(MPI_IN_PLACE, data, len, MPI_DOUBLE, MPI_SUM, comm->Comm_merge);
  }
}

void dup_bcast_cpu(double* data, int64_t len, const CellComm* comm) {
  if (comm->Comm_share != MPI_COMM_NULL) {
    MPI_Bcast(data, len, MPI_DOUBLE, 0, comm->Comm_share);
  }
}

void buildBasis(const EvalDouble& eval, Base basis[], Cell* cells, int64_t levels, const CellComm* comm, const double* bodies, int64_t nbodies, int64_t mrank) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0, ibegin = 0, nodes = 0;
    content_length(&nodes, &xlen, &ibegin, &comm[l]);
    int64_t iend = ibegin + nodes;
    basis[l].Dims = std::vector<int64_t>(xlen, 0);
    basis[l].DimsLr = std::vector<int64_t>(xlen, 0);

    Matrix* arr_m = (Matrix*)calloc(xlen * 2, sizeof(Matrix));
    basis[l].Uo = arr_m;
    basis[l].R = &arr_m[xlen];
    std::vector<int64_t> celli(xlen, 0);

    for (int64_t i = 0; i < xlen; i++) {
      int64_t childi = std::get<0>(comm[l].LocalChild[i]);
      int64_t clen = std::get<1>(comm[l].LocalChild[i]);
      int64_t gi = comm[l].iGlobal(i);
      celli[i] = gi;

      if (childi >= 0 && l < levels)
        for (int64_t j = 0; j < clen; j++)
          basis[l].Dims[i] = basis[l].Dims[i] + basis[l + 1].DimsLr[childi + j];
      else
        basis[l].Dims[i] = cells[celli[i]].Body[1] - cells[celli[i]].Body[0];
    }

    int64_t seg_dim = neighbor_bcast_sizes_cpu(&basis[l].Dims[0], &comm[l]);
    int64_t seg_skeletons = 3 * seg_dim;
    int64_t seg_matrix = seg_dim * seg_dim * 2;
    std::vector<double> Skeletons(xlen * seg_skeletons, 0.);
    std::vector<double> matrix_data(nodes * seg_matrix, 0.);
    
    if (l < levels) {
      int64_t seg = basis[l + 1].dimS;
      for (int64_t i = 0; i < nodes; i++) {
        int64_t childi = std::get<0>(comm[l].LocalChild[i + ibegin]);
        int64_t clen = std::get<1>(comm[l].LocalChild[i + ibegin]);
        int64_t y = 0;
        for (int64_t j = 0; j < clen; j++) {
          int64_t len = basis[l + 1].DimsLr[childi + j];
          memcpy(&Skeletons[(i + ibegin) * seg_skeletons + y * 3], &basis[l + 1].M[(childi + j) * seg * 3], len * 3 * sizeof(double));
          memcpy2d(&matrix_data[i * seg_matrix + y * (seg_dim + 1)], 
            &basis[l + 1].R_cpu[(childi + j) * seg * seg], len, len, seg_dim, seg);
          y = y + len;
        }
      }
      neighbor_bcast_cpu(&Skeletons[0], seg_skeletons, &comm[l]);
      dup_bcast_cpu(&Skeletons[0], seg_skeletons * xlen, &comm[l]);
    }
    else
      for (int64_t i = 0; i < xlen; i++) {
        int64_t len = cells[celli[i]].Body[1] - cells[celli[i]].Body[0];
        int64_t offset = 3 * cells[celli[i]].Body[0];
        memcpy(&Skeletons[i * seg_skeletons], &bodies[offset], len * 3 * sizeof(double));
        if (ibegin <= i && i < iend)
          for (int64_t j = 0; j < len; j++)
            matrix_data[seg_matrix * (i - ibegin) + j * (seg_dim + 1)] = 1.;
      }

    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = basis[l].Dims[i + ibegin];
      double* mat = &matrix_data[seg_matrix * i];
      double* Xbodies = &Skeletons[(i + ibegin) * seg_skeletons];
      int64_t ci = celli[i + ibegin];

      std::vector<double> Fbodies;
      for (int64_t j = 0; j < nbodies; j++)
        for (int64_t k = 0; k < 3; k++)
          if (j < cells[ci].Body[0] || cells[ci].Body[1] <= j)
            Fbodies.emplace_back(bodies[j * 3 + k]);
      
      compute_basis(eval, mrank, ske_len, mat, seg_dim, &Xbodies[0], Fbodies.size() / 3, &Fbodies[0]);
      basis[l].DimsLr[i + ibegin] = std::min(mrank, ske_len);
    }
    neighbor_bcast_sizes_cpu(basis[l].DimsLr.data(), &comm[l]);

    int64_t max[3] = { 0, 0, 0 };
    for (int64_t i = 0; i < xlen; i++) {
      int64_t i1 = basis[l].DimsLr[i];
      int64_t i2 = basis[l].Dims[i] - basis[l].DimsLr[i];
      max[0] = std::max(max[0], i1);
      max[1] = std::max(max[1], i2);
      max[2] = std::max(max[2], std::get<1>(comm[l].LocalChild[i]));
    }

    basis[l].dimN = std::max(max[0] + max[1], basis[l + 1].dimS * max[2]);
    basis[l].dimS = max[0];
    basis[l].dimR = basis[l].dimN - basis[l].dimS;
    int64_t stride = basis[l].dimN * basis[l].dimN;
    int64_t stride_r = basis[l].dimS * basis[l].dimS;
    int64_t LD = basis[l].dimN;

    basis[l].M = (double*)calloc(basis[l].dimS * xlen * 3, sizeof(double));
    basis[l].U = (double*)calloc(stride * xlen + nodes * basis[l].dimR, sizeof(double));
    basis[l].R_cpu = (double*)calloc(stride_r * xlen, sizeof(double));

    for (int64_t i = 0; i < xlen; i++) {
      double* M_ptr = basis[l].M + i * basis[l].dimS * 3;
      double* Uc_ptr = basis[l].U + i * stride;
      double* Uo_ptr = Uc_ptr + basis[l].dimR * basis[l].dimN;
      double* R_ptr = basis[l].R_cpu + i * stride_r;

      int64_t Nc = basis[l].Dims[i] - basis[l].DimsLr[i];
      int64_t No = basis[l].DimsLr[i];
      int64_t M = basis[l].Dims[i];

      if (ibegin <= i && i < iend) {
        int64_t child = std::get<0>(comm[l].LocalChild[i]);
        int64_t clen = std::get<1>(comm[l].LocalChild[i]);
        if (child >= 0 && l < levels) {
          int64_t row = 0;
          for (int64_t j = 0; j < clen; j++) {
            int64_t N = basis[l + 1].DimsLr[child + j];
            int64_t Urow = j * basis[l + 1].dimS;
            memcpy2d(&Uc_ptr[Urow], &matrix_data[(i - ibegin) * seg_matrix + No * seg_dim + row], N, Nc, LD, seg_dim);
            memcpy2d(&Uo_ptr[Urow], &matrix_data[(i - ibegin) * seg_matrix + row], N, No, LD, seg_dim);
            row = row + N;
          }
        }
        else {
          memcpy2d(Uc_ptr, &matrix_data[(i - ibegin) * seg_matrix + No * seg_dim], M, Nc, LD, seg_dim);
          memcpy2d(Uo_ptr, &matrix_data[(i - ibegin) * seg_matrix], M, No, LD, seg_dim);
        }
        memcpy2d(R_ptr, &matrix_data[(i - ibegin) * seg_matrix + M * seg_dim], No, No, basis[l].dimS, seg_dim);
        memcpy(M_ptr, &Skeletons[i * seg_skeletons], 3 * No * sizeof(double));

        double* Ui_ptr = basis[l].U + xlen * stride + (i - ibegin) * basis[l].dimR; 
        for (int64_t j = Nc; j < basis[l].dimR; j++)
          Ui_ptr[j] = 1.;
      }

      basis[l].Uo[i] = (Matrix) { Uo_ptr, basis[l].dimN, basis[l].dimS, basis[l].dimN };
      basis[l].R[i] = (Matrix) { R_ptr, basis[l].dimS, basis[l].dimS, basis[l].dimS };
    }
    neighbor_bcast_cpu(basis[l].M, 3 * basis[l].dimS, &comm[l]);
    dup_bcast_cpu(basis[l].M, 3 * basis[l].dimS * xlen, &comm[l]);
    neighbor_bcast_cpu(basis[l].U, stride, &comm[l]);
    dup_bcast_cpu(basis[l].U, stride * xlen, &comm[l]);
    neighbor_bcast_cpu(basis[l].R_cpu, stride_r, &comm[l]);
    dup_bcast_cpu(basis[l].R_cpu, stride_r * xlen, &comm[l]);
  }
}


void basis_free(Base* basis) {
  free(basis->Uo);
  if (basis->M)
    free(basis->M);
  if (basis->U)
    free(basis->U);
  if (basis->R_cpu)
    free(basis->R_cpu);
}


void get_bounds(const double* bodies, int64_t nbodies, double R[], double C[]) {
  double Xmin[3];
  double Xmax[3];
  Xmin[0] = Xmax[0] = bodies[0];
  Xmin[1] = Xmax[1] = bodies[1];
  Xmin[2] = Xmax[2] = bodies[2];

  for (int64_t i = 1; i < nbodies; i++) {
    const double* x_bi = &bodies[i * 3];
    Xmin[0] = fmin(x_bi[0], Xmin[0]);
    Xmin[1] = fmin(x_bi[1], Xmin[1]);
    Xmin[2] = fmin(x_bi[2], Xmin[2]);

    Xmax[0] = fmax(x_bi[0], Xmax[0]);
    Xmax[1] = fmax(x_bi[1], Xmax[1]);
    Xmax[2] = fmax(x_bi[2], Xmax[2]);
  }

  C[0] = (Xmin[0] + Xmax[0]) / 2.;
  C[1] = (Xmin[1] + Xmax[1]) / 2.;
  C[2] = (Xmin[2] + Xmax[2]) / 2.;

  double d0 = Xmax[0] - Xmin[0];
  double d1 = Xmax[1] - Xmin[1];
  double d2 = Xmax[2] - Xmin[2];

  R[0] = (d0 == 0. && Xmin[0] == 0.) ? 0. : (1.e-8 + d0 / 2.);
  R[1] = (d1 == 0. && Xmin[1] == 0.) ? 0. : (1.e-8 + d1 / 2.);
  R[2] = (d2 == 0. && Xmin[2] == 0.) ? 0. : (1.e-8 + d2 / 2.);
}

void sort_bodies(double* bodies, int64_t nbodies, int64_t sdim) {
  std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(bodies);
  std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[3 * nbodies]);
  std::sort(bodies3, bodies3_end, [=](std::array<double, 3>& i, std::array<double, 3>& j)->bool {
    double x = i[sdim];
    double y = j[sdim];
    return x < y;
  });
}

int admis_check(double theta, const double C1[], const double C2[], const double R1[], const double R2[]) {
  double dCi[3];
  dCi[0] = C1[0] - C2[0];
  dCi[1] = C1[1] - C2[1];
  dCi[2] = C1[2] - C2[2];

  dCi[0] = dCi[0] * dCi[0];
  dCi[1] = dCi[1] * dCi[1];
  dCi[2] = dCi[2] * dCi[2];

  double dRi[3];
  dRi[0] = R1[0] * R1[0];
  dRi[1] = R1[1] * R1[1];
  dRi[2] = R1[2] * R1[2];

  double dRj[3];
  dRj[0] = R2[0] * R2[0];
  dRj[1] = R2[1] * R2[1];
  dRj[2] = R2[2] * R2[2];

  double dC = dCi[0] + dCi[1] + dCi[2];
  double dR = (dRi[0] + dRi[1] + dRi[2] + dRj[0] + dRj[1] + dRj[2]) * theta;
  return (int)(dC > dR);
}

void buildTree(int64_t* ncells, Cell* cells, double* bodies, int64_t nbodies, int64_t levels) {
  Cell* root = &cells[0];
  root->Body[0] = 0;
  root->Body[1] = nbodies;
  root->Level = 0;
  get_bounds(bodies, nbodies, root->R, root->C);

  int64_t len = 1;
  int64_t i = 0;
  while (i < len) {
    Cell* ci = &cells[i];
    ci->Child[0] = -1;
    ci->Child[1] = -1;

    if (ci->Level < levels) {
      int64_t sdim = 0;
      double maxR = ci->R[0];
      if (ci->R[1] > maxR)
      { sdim = 1; maxR = ci->R[1]; }
      if (ci->R[2] > maxR)
      { sdim = 2; maxR = ci->R[2]; }

      int64_t i_begin = ci->Body[0];
      int64_t i_end = ci->Body[1];
      int64_t nbody_i = i_end - i_begin;
      sort_bodies(&bodies[i_begin * 3], nbody_i, sdim);
      int64_t loc = i_begin + nbody_i / 2;

      Cell* c0 = &cells[len];
      Cell* c1 = &cells[len + 1];
      ci->Child[0] = len;
      ci->Child[1] = len + 2;
      len = len + 2;

      c0->Body[0] = i_begin;
      c0->Body[1] = loc;
      c1->Body[0] = loc;
      c1->Body[1] = i_end;
      
      c0->Level = ci->Level + 1;
      c1->Level = ci->Level + 1;

      get_bounds(&bodies[i_begin * 3], loc - i_begin, c0->R, c0->C);
      get_bounds(&bodies[loc * 3], i_end - loc, c1->R, c1->C);
    }
    i++;
  }
  *ncells = len;
}

void getList(char NoF, int64_t* len, int64_t rels[], int64_t ncells, const Cell cells[], int64_t i, int64_t j, double theta) {
  const Cell* Ci = &cells[i];
  const Cell* Cj = &cells[j];
  int64_t ilevel = Ci->Level;
  int64_t jlevel = Cj->Level; 
  if (ilevel == jlevel) {
    int admis = admis_check(theta, Ci->C, Cj->C, Ci->R, Cj->R);
    int write_far = NoF == 'F' || NoF == 'f';
    int write_near = NoF == 'N' || NoF == 'n';
    if (admis ? write_far : write_near) {
      int64_t n = *len;
      rels[n] = i + j * ncells;
      *len = n + 1;
    }
    if (admis)
      return;
  }
  if (ilevel <= jlevel && Ci->Child[0] >= 0)
    for (int64_t k = Ci->Child[0]; k < Ci->Child[1]; k++)
      getList(NoF, len, rels, ncells, cells, k, j, theta);
  else if (jlevel <= ilevel && Cj->Child[0] >= 0)
    for (int64_t k = Cj->Child[0]; k < Cj->Child[1]; k++)
      getList(NoF, len, rels, ncells, cells, i, k, theta);
}

void traverse(char NoF, CSR* rels, int64_t ncells, const Cell* cells, double theta) {
  rels->M = ncells;
  rels->N = ncells;
  std::vector<int64_t> rel_arr(ncells * ncells);
  int64_t len = 0;
  getList(NoF, &len, &rel_arr[0], ncells, cells, 0, 0, theta);
  std::sort(rel_arr.begin(), rel_arr.begin() + len);

  rels->RowIndex.resize(ncells + 1);
  rels->ColIndex.resize(len);

  int64_t loc = -1;
  for (int64_t i = 0; i < len; i++) {
    int64_t r = rel_arr[i];
    int64_t x = r / ncells;
    int64_t y = r - x * ncells;
    rels->ColIndex[i] = y;
    while (x > loc)
      rels->RowIndex[++loc] = i;
  }
  for (int64_t i = loc + 1; i <= ncells; i++)
    rels->RowIndex[i] = len;
}

void countMaxIJ(int64_t* max_i, int64_t* max_j, const CSR* rels) {
  std::vector<int64_t> countx(rels->N, 0), county(rels->M, 0);
  for (int64_t x = 0; x < rels->N; x++)
    for (int64_t yx = rels->RowIndex[x]; yx < rels->RowIndex[x + 1]; yx++) {
      int64_t y = rels->ColIndex[yx];
      countx[x] = countx[x] + 1;
      county[y] = county[y] + 1;
    }
  if (max_i)
    *max_i = *std::max_element(county.begin(), county.end());
  if (max_j)
    *max_j = *std::max_element(countx.begin(), countx.end());
}

void loadX(double* X, int64_t seg, const double Xbodies[], int64_t Xbegin, int64_t ncells, const Cell cells[]) {
  for (int64_t i = 0; i < ncells; i++) {
    int64_t b0 = cells[i].Body[0] - Xbegin;
    int64_t lenB = cells[i].Body[1] - cells[i].Body[0];
    for (int64_t j = 0; j < lenB; j++)
      X[i * seg + j] = Xbodies[j + b0];
  }
}

void evalD(const EvalDouble& eval, Matrix* D, const CSR* rels, const Cell* cells, const double* bodies, const CellComm* comm) {
  int64_t ibegin = 0, nodes = 0;
  content_length(&nodes, NULL, &ibegin, comm);
  ibegin = comm->iGlobal(ibegin);

  for (int64_t i = 0; i < nodes; i++) {
    int64_t lc = ibegin + i;
    const Cell* ci = &cells[lc];
    int64_t nbegin = rels->RowIndex[lc];
    int64_t nlen = rels->RowIndex[lc + 1] - nbegin;
    const int64_t* ngbs = &rels->ColIndex[nbegin];
    int64_t x_begin = ci->Body[0];
    int64_t n = ci->Body[1] - x_begin;
    int64_t offsetD = nbegin - rels->RowIndex[ibegin];

    for (int64_t j = 0; j < nlen; j++) {
      int64_t lj = ngbs[j];
      const Cell* cj = &cells[lj];
      int64_t y_begin = cj->Body[0];
      int64_t m = cj->Body[1] - y_begin;
      gen_matrix(eval, n, m, &bodies[x_begin * 3], &bodies[y_begin * 3], D[offsetD + j].A, D[offsetD + j].LDA);
    }
  }
}

void evalS(const EvalDouble& eval, Matrix* S, const Base* basis, const CSR* rels, const CellComm* comm) {
  int64_t ibegin = 0;
  content_length(NULL, NULL, &ibegin, comm);
  int64_t seg = basis->dimS * 3;

  for (int64_t x = 0; x < rels->N; x++) {
    int64_t n = basis->DimsLr[x + ibegin];

    for (int64_t yx = rels->RowIndex[x]; yx < rels->RowIndex[x + 1]; yx++) {
      int64_t y = rels->ColIndex[yx];
      int64_t m = basis->DimsLr[y];
      gen_matrix(eval, n, m, &basis->M[(x + ibegin) * seg], &basis->M[y * seg], S[yx].A, S[yx].LDA);
      mul_AS(&basis->R[x + ibegin], &basis->R[y], &S[yx]);
    }
  }
}

MPI_Comm MPI_Comm_split_unique(std::vector<MPI_Comm>& unique_comms, int color, int mpi_rank) {
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, color, mpi_rank, &comm);

  if (comm != MPI_COMM_NULL) {
    auto iter = std::find_if(unique_comms.begin(), unique_comms.end(), [comm](MPI_Comm c) -> bool { 
      int result; MPI_Comm_compare(comm, c, &result); return result == MPI_IDENT || result == MPI_CONGRUENT; });
    if (iter == unique_comms.end())
      unique_comms.emplace_back(comm);
    else {
      MPI_Comm_free(&comm);
      comm = *iter;
    }
  }
  return comm;
}

std::pair<int64_t, int64_t> local_to_pnx(int64_t ilocal, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  int64_t iter = 0;
  while (iter < (int64_t)ProcBoxes.size() && std::get<1>(ProcBoxes[iter]) <= ilocal) {
    ilocal = ilocal - std::get<1>(ProcBoxes[iter]);
    iter = iter + 1;
  }
  if (0 <= ilocal && iter < (int64_t)ProcBoxes.size())
    return std::make_pair(iter, ilocal);
  else
    return std::make_pair(-1, -1);
}

std::pair<int64_t, int64_t> global_to_pnx(int64_t iglobal, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  int64_t iter = 0;
  while (iter < (int64_t)ProcBoxes.size() && (std::get<0>(ProcBoxes[iter]) + std::get<1>(ProcBoxes[iter])) <= iglobal)
    iter = iter + 1;
  if (iter < (int64_t)ProcBoxes.size() && std::get<0>(ProcBoxes[iter]) <= iglobal)
    return std::make_pair(iter, iglobal - std::get<0>(ProcBoxes[iter]));
  else
    return std::make_pair(-1, -1);
}

int64_t pnx_to_local(std::pair<int64_t, int64_t> pnx, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  if (std::get<0>(pnx) >= 0 && std::get<0>(pnx) < (int64_t)ProcBoxes.size() && std::get<1>(pnx) >= 0) {
    int64_t iter = 0, slen = 0;
    while (iter < std::get<0>(pnx)) {
      slen = slen + std::get<1>(ProcBoxes[iter]);
      iter = iter + 1;
    }
    return std::get<1>(pnx) + slen;
  }
  else
    return -1;
}

int64_t pnx_to_global(std::pair<int64_t, int64_t> pnx, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  if (std::get<0>(pnx) >= 0 && std::get<0>(pnx) < (int64_t)ProcBoxes.size() && std::get<1>(pnx) >= 0)
    return std::get<1>(pnx) + std::get<0>(ProcBoxes[std::get<0>(pnx)]);
  else
    return -1;
}

void get_level_procs(std::vector<std::pair<int64_t, int64_t>>& Procs, std::vector<std::pair<int64_t, int64_t>>& Levels, 
  int64_t mpi_rank, int64_t mpi_size, const std::vector<std::pair<int64_t, int64_t>>& Child, int64_t levels) {
  int64_t ncells = (int64_t)Child.size();
  std::vector<int64_t> levels_cell(ncells);
  Procs[0] = std::make_pair(0, mpi_size);
  levels_cell[0] = 0;

  for (int64_t i = 0; i < ncells; i++) {
    int64_t child = std::get<0>(Child[i]);
    int64_t lenC = std::get<1>(Child[i]);
    int64_t lenP = std::get<1>(Procs[i]) - std::get<0>(Procs[i]);
    int64_t p = std::get<0>(Procs[i]);
    
    if (child >= 0 && lenC > 0) {
      double divP = (double)lenP / (double)lenC;
      for (int64_t j = 0; j < lenC; j++) {
        int64_t p0 = j == 0 ? 0 : (int64_t)std::floor(j * divP);
        int64_t p1 = j == (lenC - 1) ? lenP : (int64_t)std::floor((j + 1) * divP);
        p1 = std::max(p1, p0 + 1);
        Procs[child + j] = std::make_pair(p + p0, p + p1);
        levels_cell[child + j] = levels_cell[i] + 1;
      }
    }
  }
  
  int64_t begin = 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t ibegin = std::distance(levels_cell.begin(), 
      std::find(levels_cell.begin() + begin, levels_cell.end(), i));
    int64_t iend = std::distance(levels_cell.begin(), 
      std::find(levels_cell.begin() + begin, levels_cell.end(), i + 1));
    int64_t pbegin = std::distance(Procs.begin(), 
      std::find_if(Procs.begin() + ibegin, Procs.begin() + iend, [=](std::pair<int64_t, int64_t>& p) -> bool {
        return std::get<0>(p) <= mpi_rank && mpi_rank < std::get<1>(p);
      }));
    int64_t pend = std::distance(Procs.begin(), 
      std::find_if_not(Procs.begin() + pbegin, Procs.begin() + iend, [=](std::pair<int64_t, int64_t>& p) -> bool {
        return std::get<0>(p) <= mpi_rank && mpi_rank < std::get<1>(p);
      }));
    Levels[i] = std::make_pair(pbegin, pend);
    begin = iend;
  }
}

void buildComm(CellComm* comms, int64_t ncells, const Cell* cells, const CSR* cellFar, const CSR* cellNear, int64_t levels) {
  int __mpi_rank = 0, __mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  int64_t mpi_rank = __mpi_rank;
  int64_t mpi_size = __mpi_size;

  std::vector<MPI_Comm> unique_comms;
  std::vector<std::pair<int64_t, int64_t>> Child(ncells), Procs(ncells), Levels(levels + 1);
  std::transform(cells, &cells[ncells], Child.begin(), [](const Cell& c) {
    return std::make_pair(c.Child[0], c.Child[1] - c.Child[0]);
  });
  get_level_procs(Procs, Levels, mpi_rank, mpi_size, Child, levels);

  for (int64_t i = levels; i >= 0; i--) {
    int64_t mbegin = std::get<0>(Levels[i]);
    int64_t mend = std::get<1>(Levels[i]);
    int64_t p = std::get<0>(Procs[mbegin]);
    int64_t lenp = std::get<1>(Procs[mbegin]) - p;

    std::vector<int64_t> ProcTargets;
    for (int64_t j = 0; j < mpi_size; j++) {
      int is_ngb = 0;
      for (int64_t k = cellNear->RowIndex[mbegin]; k < cellNear->RowIndex[mend]; k++)
        if (std::get<0>(Procs[cellNear->ColIndex[k]]) == j)
          is_ngb = 1;
      for (int64_t k = cellFar->RowIndex[mbegin]; k < cellFar->RowIndex[mend]; k++)
        if (std::get<0>(Procs[cellFar->ColIndex[k]]) == j)
          is_ngb = 1;
      
      int color = (is_ngb && p == mpi_rank) ? 1 : MPI_UNDEFINED;
      MPI_Comm comm = MPI_Comm_split_unique(unique_comms, color, mpi_rank);

      if (comm != MPI_COMM_NULL) {
        int root = 0;
        if (j == p)
          MPI_Comm_rank(comm, &root);
        MPI_Allreduce(MPI_IN_PLACE, &root, 1, MPI_INT, MPI_SUM, comm);
        comms[i].Comm_box.emplace_back(root, comm);
      }
      if (is_ngb)
        ProcTargets.emplace_back(j);
    }
    comms[i].Proc = std::distance(ProcTargets.begin(), std::find(ProcTargets.begin(), ProcTargets.end(), p));

    int color = MPI_UNDEFINED;
    int64_t cc = std::get<0>(Child[mbegin]);
    int64_t clen = std::get<1>(Child[mbegin]);
    if (lenp > 1 && cc >= 0)
      for (int64_t j = 0; j < clen; j++)
        if (std::get<0>(Procs[cc + j]) == mpi_rank)
          color = p;
    comms[i].Comm_merge = MPI_Comm_split_unique(unique_comms, color, mpi_rank);
  
    color = lenp > 1 ? p : MPI_UNDEFINED;
    comms[i].Comm_share = MPI_Comm_split_unique(unique_comms, color, mpi_rank);

    std::pair<int64_t, int64_t> local = std::make_pair(mbegin, mend - mbegin);
    comms[i].ProcBoxes = std::vector<std::pair<int64_t, int64_t>>(ProcTargets.size(), local);

    for (int64_t j = 0; j < (int64_t)comms[i].Comm_box.size(); j++)
      MPI_Bcast(&comms[i].ProcBoxes[j], sizeof(std::pair<int64_t, int64_t>), MPI_BYTE, std::get<0>(comms[i].Comm_box[j]), std::get<1>(comms[i].Comm_box[j]));
    if (comms[i].Comm_share != MPI_COMM_NULL)
      MPI_Bcast(&comms[i].ProcBoxes[0], sizeof(std::pair<int64_t, int64_t>) * comms[i].ProcBoxes.size(), MPI_BYTE, 0, comms[i].Comm_share);

    for (int64_t j = 0; j < (int64_t)comms[i].ProcBoxes.size(); j++)
      for (int64_t k = 0; k < std::get<1>(comms[i].ProcBoxes[j]); k++) {
        int64_t ki = k + std::get<0>(comms[i].ProcBoxes[j]);
        int64_t li = pnx_to_local(std::make_pair(j, k), comms[i].ProcBoxes);
        int64_t lc = std::get<0>(Child[ki]);
        int64_t lclen = std::get<1>(Child[ki]);
        if (i < levels) {
          std::pair<int64_t, int64_t> pnx = global_to_pnx(lc, comms[i + 1].ProcBoxes);
          lc = pnx_to_local(pnx, comms[i + 1].ProcBoxes);
          if (lc >= 0)
            std::for_each(comms[i + 1].LocalParent.begin() + lc, comms[i + 1].LocalParent.begin() + (lc + lclen), 
              [&](std::pair<int64_t, int64_t>& x) { std::get<0>(x) = li; std::get<1>(x) = std::distance(&comms[i + 1].LocalParent[lc], &x); });
          else
            lclen = 0;
        }
        comms[i].LocalChild.emplace_back(lc, lclen);
      }
    comms[i].LocalParent = std::vector<std::pair<int64_t, int64_t>>(comms[i].LocalChild.size(), std::make_pair(-1, -1));
  }

  std::vector<ncclUniqueId> nccl_ids(unique_comms.size());
  std::vector<ncclComm_t> nccl_comms(unique_comms.size());
  ncclGroupStart();
  for (int64_t i = 0; i < (int64_t)unique_comms.size(); i++) {
    int rank, size;
    MPI_Comm_rank(unique_comms[i], &rank);
    MPI_Comm_size(unique_comms[i], &size);
    if (rank == 0)
      ncclGetUniqueId(&nccl_ids[i]);
    MPI_Bcast((void*)&nccl_ids[i], sizeof(ncclUniqueId), MPI_BYTE, 0, unique_comms[i]);
    ncclCommInitRank(&nccl_comms[i], size, nccl_ids[i], rank);
  }
  ncclGroupEnd();

  for (int64_t i = levels; i >= 0; i--) {
    comms[i].NCCL_box = std::vector<std::pair<int, ncclComm_t>>(comms[i].Comm_box.size(), std::make_pair(0, (ncclComm_t)NULL));
    comms[i].NCCL_merge = NULL;
    comms[i].NCCL_share = NULL; 

    for (int64_t j = 0; j < (int64_t)comms[i].Comm_box.size(); j++) {
      int64_t k = std::distance(unique_comms.begin(), std::find(unique_comms.begin(), unique_comms.end(), std::get<1>(comms[i].Comm_box[j])));
      comms[i].NCCL_box[j] = std::make_pair(std::get<0>(comms[i].Comm_box[j]), nccl_comms[k]);
    }
    if (comms[i].Comm_merge != MPI_COMM_NULL) {
      int64_t k = std::distance(unique_comms.begin(), std::find(unique_comms.begin(), unique_comms.end(), comms[i].Comm_merge));
      comms[i].NCCL_merge = nccl_comms[k];
    }
    if (comms[i].Comm_share != MPI_COMM_NULL) {
      int64_t k = std::distance(unique_comms.begin(), std::find(unique_comms.begin(), unique_comms.end(), comms[i].Comm_share));
      comms[i].NCCL_share = nccl_comms[k];
    }
  }
}

void cellComm_free(CellComm* comms, int64_t levels) {
  std::vector<MPI_Comm> mpi_comms;
  std::vector<ncclComm_t> nccl_comms;

  for (int64_t i = 0; i <= levels; i++) {
    for (int64_t j = 0; j < (int64_t)comms[i].Comm_box.size(); j++)
      mpi_comms.emplace_back(std::get<1>(comms[i].Comm_box[j]));
    if (comms[i].Comm_merge != MPI_COMM_NULL)
      mpi_comms.emplace_back(comms[i].Comm_merge);
    if (comms[i].Comm_share != MPI_COMM_NULL)
      mpi_comms.emplace_back(comms[i].Comm_share);

    for (int64_t j = 0; j < (int64_t)comms[i].NCCL_box.size(); j++)
      nccl_comms.emplace_back(std::get<1>(comms[i].NCCL_box[j]));
    if (comms[i].NCCL_merge != NULL)
      nccl_comms.emplace_back(comms[i].NCCL_merge);
    if (comms[i].NCCL_share != NULL)
      nccl_comms.emplace_back(comms[i].NCCL_share);
    
    comms[i].Comm_box.clear();
    comms[i].Comm_merge = MPI_COMM_NULL;
    comms[i].Comm_share = MPI_COMM_NULL;
    comms[i].NCCL_box.clear();
    comms[i].NCCL_merge = NULL;
    comms[i].NCCL_share = NULL;
  }

  std::sort(mpi_comms.begin(), mpi_comms.end());
  std::sort(nccl_comms.begin(), nccl_comms.end());
  mpi_comms.erase(std::unique(mpi_comms.begin(), mpi_comms.end()), mpi_comms.end());
  nccl_comms.erase(std::unique(nccl_comms.begin(), nccl_comms.end()), nccl_comms.end());

  for (int64_t i = 0; i < (int64_t)mpi_comms.size(); i++)
    MPI_Comm_free(&mpi_comms[i]);
  for (int64_t i = 0; i < (int64_t)nccl_comms.size(); i++)
    ncclCommDestroy(nccl_comms[i]);
}

void relations(CSR rels[], const CSR* cellRel, int64_t levels, const CellComm* comm) {
 
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes, neighbors, ibegin;
    content_length(&nodes, &neighbors, &ibegin, &comm[i]);
    ibegin = comm[i].iGlobal(ibegin);
    CSR* csc = &rels[i];

    csc->M = neighbors;
    csc->N = nodes;
    int64_t ent_max = nodes * csc->M;
    csc->RowIndex.resize(nodes + 1);
    csc->ColIndex.resize(ent_max);

    int64_t count = 0;
    for (int64_t j = 0; j < nodes; j++) {
      int64_t lc = ibegin + j;
      csc->RowIndex[j] = count;
      int64_t cbegin = cellRel->RowIndex[lc];
      int64_t ent = cellRel->RowIndex[lc + 1] - cbegin;
      for (int64_t k = 0; k < ent; k++) {
        csc->ColIndex[count + k] = comm[i].iLocal(cellRel->ColIndex[cbegin + k]);
      }
      count = count + ent;
    }

    if (count < ent_max)
      csc->ColIndex.resize(count);
    csc->RowIndex[nodes] = count;
  }
}

int64_t CellComm::iLocal(int64_t iglobal) const {
  return pnx_to_local(global_to_pnx(iglobal, ProcBoxes), ProcBoxes);
}

int64_t CellComm::iGlobal(int64_t ilocal) const {
  return pnx_to_global(local_to_pnx(ilocal, ProcBoxes), ProcBoxes);
}

void neighbor_bcast_gpu(double* data, int64_t seg, const CellComm* comm) {
  if (comm->NCCL_box.size() > 0) {
    ncclGroupStart();
    int64_t y = 0;
    for (int64_t p = 0; p < (int64_t)comm->NCCL_box.size(); p++) {
      int64_t llen = std::get<1>(comm->ProcBoxes[p]) * seg;
      double* loc = &data[y];
      ncclBroadcast((const void*)loc, loc, llen, ncclDouble, std::get<0>(comm->NCCL_box[p]), std::get<1>(comm->NCCL_box[p]), comm->stream);
      y = y + llen;
    }
    ncclGroupEnd();
  }
}

void neighbor_reduce_gpu(double* data, int64_t seg, const CellComm* comm) {
  if (comm->NCCL_box.size() > 0) {
    int64_t y = 0;
    ncclGroupStart();
    for (int64_t p = 0; p < (int64_t)comm->NCCL_box.size(); p++) {
      int64_t llen = std::get<1>(comm->ProcBoxes[p]) * seg;
      double* loc = &data[y];
      ncclReduce((const void*)loc, loc, llen, ncclDouble, ncclSum, std::get<0>(comm->NCCL_box[p]), std::get<1>(comm->NCCL_box[p]), comm->stream);
      y = y + llen;
    }
    ncclGroupEnd();
  }
}

void level_merge_gpu(double* data, int64_t len, const CellComm* comm) {
  if (comm->NCCL_merge != NULL) {
    ncclAllReduce((const void*)data, data, len, ncclDouble, ncclSum, comm->NCCL_merge, comm->stream);
  }
}

void dup_bcast_gpu(double* data, int64_t len, const CellComm* comm) {
  if (comm->NCCL_share != NULL) {
    ncclBroadcast((const void*)data, data, len, ncclDouble, 0, comm->NCCL_share, comm->stream);
  }
}

cudaStream_t stream = NULL;
cublasHandle_t cublasH = NULL;
cusolverDnHandle_t cusolverH = NULL;

void* init_libs(int* argc, char*** argv) {
  if (MPI_Init(argc, argv) != MPI_SUCCESS)
    fprintf(stderr, "MPI Init Error\n");
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int num_device;
  int gpu_avail = (cudaGetDeviceCount(&num_device) == cudaSuccess);

  if (gpu_avail) {
    int device = mpi_rank % num_device;
    cudaSetDevice(device);
    
    cudaStreamCreate(&stream);
    cublasCreate(&cublasH);
    cublasSetStream(cublasH, stream);

    cusolverDnCreate(&cusolverH);
    cusolverDnSetStream(cusolverH, stream);
  }
  return stream;
}

void fin_libs() {
  if (stream)
    cudaStreamDestroy(stream);
  if (cublasH)
    cublasDestroy(cublasH);
  if (cusolverH)
    cusolverDnDestroy(cusolverH);
  MPI_Finalize();
}

void set_work_size(int64_t Lwork, double** D_DATA, int64_t* D_DATA_SIZE) {
  if (Lwork > *D_DATA_SIZE) {
    *D_DATA_SIZE = Lwork;
    if (*D_DATA)
      cudaFree(*D_DATA);
    cudaMalloc((void**)D_DATA, sizeof(double) * Lwork);
  }
  else if (Lwork <= 0) {
    *D_DATA_SIZE = 0;
    if (*D_DATA)
      cudaFree(*D_DATA);
  }
}

void allocBufferedList(void** A_ptr, void** A_buffer, int64_t element_size, int64_t count) {
  int64_t bytes = element_size * count;
  cudaMalloc((void**)A_ptr, bytes);
  *A_buffer = malloc(bytes);
  memset((void*)*A_buffer, 0, bytes);
}

void flushBuffer(char dir, void* A_ptr, void* A_buffer, int64_t element_size, int64_t count) {
  int64_t bytes = element_size * count;
  if (dir == 'G' || dir == 'g')
    cudaMemcpy(A_buffer, A_ptr, bytes, cudaMemcpyDeviceToHost);
  else if (dir == 'S' || dir == 's')
    cudaMemcpy(A_ptr, A_buffer, bytes, cudaMemcpyHostToDevice);
}

void freeBufferedList(void* A_ptr, void* A_buffer) {
  cudaFree(A_ptr);
  free(A_buffer);
}

int64_t partition_DLU(int64_t row_coords[], int64_t col_coords[], int64_t orders[], int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]) {
  int64_t NNZ = col_A[N_cols] - col_A[0];
  std::vector<std::tuple<int64_t, int64_t, int64_t>> coo_list(NNZ);
  std::iota(orders, &orders[NNZ], 0);
  for (int64_t x = 0; x < N_cols; x++) {
    int64_t begin = col_A[x] - col_A[0];
    int64_t end = col_A[x + 1] - col_A[0];
    std::transform(row_A + begin, row_A + end, orders + begin, coo_list.begin() + begin, 
      [=](int64_t y, int64_t yx) { return std::make_tuple(y, x + col_offset, yx); });
  }

  auto iter = std::stable_partition(coo_list.begin(), coo_list.end(), 
    [](std::tuple<int64_t, int64_t, int64_t> i) { return std::get<0>(i) == std::get<1>(i); });
  auto iterL = std::stable_partition(iter, coo_list.end(),
    [](std::tuple<int64_t, int64_t, int64_t> i) { return std::get<0>(i) > std::get<1>(i); });

  std::transform(coo_list.begin(), coo_list.end(), row_coords,
    [](std::tuple<int64_t, int64_t, int64_t> i) { return std::get<0>(i); });
  std::transform(coo_list.begin(), coo_list.end(), col_coords, 
    [](std::tuple<int64_t, int64_t, int64_t> i) { return std::get<1>(i); });
  std::transform(coo_list.begin(), coo_list.end(), orders, 
    [](std::tuple<int64_t, int64_t, int64_t> i) { return std::get<2>(i); });
  return std::distance(iter, iterL);
}

int64_t count_apperance_x(const int64_t X[], int64_t AX[], int64_t lenX) {
  std::pair<const int64_t*, const int64_t*> minmax_e = std::minmax_element(X, &X[lenX]);
  int64_t min_e = *std::get<0>(minmax_e);
  int64_t max_e = *std::get<1>(minmax_e);
  std::vector<int64_t> count(max_e - min_e + 1, 0);
  for (int64_t i = 0; i < lenX; i++) {
    int64_t x = X[i] - min_e;
    int64_t c = count[x];
    AX[i] = c;
    count[x] = c + 1;
  }
  return *std::max_element(count.begin(), count.end());
}

void batchParamsCreate(BatchedFactorParams* params, int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, double* X_ptr, int64_t N_up, double** A_up, double** X_up,
  double* Workspace, int64_t Lwork, int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]) {
  
  int64_t N_dim = R_dim + S_dim;
  int64_t NNZ = col_A[N_cols] - col_A[0];
  int64_t stride = N_dim * N_dim;
  int64_t lenB = Lwork / stride;
  lenB = lenB > NNZ ? NNZ : lenB;
  int64_t N_rows_aligned = ((N_rows >> 4) + ((N_rows & 15) > 0)) * 16;
  int64_t NNZ_aligned = ((NNZ >> 4) + ((NNZ & 15) > 0)) * 16;

  std::vector<int64_t> rows(NNZ), cols(NNZ), orders(NNZ);
  int64_t lenL = partition_DLU(&rows[0], &cols[0], &orders[0], N_cols, col_offset, row_A, col_A);
  std::vector<int64_t> urows(NNZ), ucols(NNZ);
  int64_t K1 = count_apperance_x(&rows[0], &urows[0], NNZ);
  int64_t K2 = count_apperance_x(&cols[0], &ucols[0], NNZ);

  std::vector<double> one_data(N_rows, 1.);
  double* one_data_dev;
  cudaMalloc(&one_data_dev, sizeof(double) * N_rows);
  cudaMemcpy(one_data_dev, &one_data[0], sizeof(double) * N_rows, cudaMemcpyHostToDevice);

  const int64_t NZ = 13, ND = 6;
  std::vector<double*> ptrs_nnz_cpu(NZ * NNZ_aligned);
  std::vector<double*> ptrs_diag_cpu(ND * N_rows_aligned);

  const double** _U_r = (const double**)&ptrs_nnz_cpu[0 * NNZ_aligned];
  const double** _U_s = (const double**)&ptrs_nnz_cpu[1 * NNZ_aligned];
  const double** _V_x = (const double**)&ptrs_nnz_cpu[2 * NNZ_aligned];
  const double** _A_sx = (const double**)&ptrs_nnz_cpu[3 * NNZ_aligned];
  double** _A_x = (double**)&ptrs_nnz_cpu[4 * NNZ_aligned];
  double** _B_x = (double**)&ptrs_nnz_cpu[5 * NNZ_aligned];
  double** _A_upper = (double**)&ptrs_nnz_cpu[6 * NNZ_aligned];
  double** _A_s = (double**)&ptrs_nnz_cpu[7 * NNZ_aligned];
  double** _Xo_Y = (double**)&ptrs_nnz_cpu[8 * NNZ_aligned];
  double** _Xc_Y = (double**)&ptrs_nnz_cpu[9 * NNZ_aligned];
  double** _Xc_X = (double**)&ptrs_nnz_cpu[10 * NNZ_aligned];
  double** _ACC_Y = (double**)&ptrs_nnz_cpu[11 * NNZ_aligned];
  double** _ACC_X = (double**)&ptrs_nnz_cpu[12 * NNZ_aligned];
  
  double** _X_d = (double**)&ptrs_diag_cpu[0 * N_rows_aligned];
  double** _A_l = (double**)&ptrs_diag_cpu[1 * N_rows_aligned];
  const double** _U_i = (const double**)&ptrs_diag_cpu[2 * N_rows_aligned];
  double** _ACC_I = (double**)&ptrs_diag_cpu[3 * N_rows_aligned];
  double** _Xo_I = (double**)&ptrs_diag_cpu[4 * N_rows_aligned];
  double** _ONE_LIST = (double**)&ptrs_diag_cpu[5 * N_rows_aligned];

  double* _V_data = Workspace;
  double* _ACC_data = &Workspace[N_cols * R_dim];

  std::vector<int64_t> ind(std::max(N_rows, NNZ) + 1);
  std::iota(ind.begin(), ind.end(), 0);

  std::transform(rows.begin(), rows.end(), _U_r, [=](int64_t y) { return &U_ptr[stride * y]; });
  std::transform(rows.begin(), rows.end(), _U_s, [=](int64_t y) { return &U_ptr[stride * y + R_dim * N_dim]; });
  std::transform(cols.begin(), cols.end(), _V_x, [=](int64_t x) { return &U_ptr[stride * x]; });
  std::transform(orders.begin(), orders.end(), _A_x, [=](int64_t yx) { return &A_ptr[stride * yx]; });
  std::transform(orders.begin(), orders.end(), _A_s, [=](int64_t yx) { return &A_ptr[stride * yx + R_dim * R_dim]; });
  std::transform(orders.begin(), orders.begin() + N_cols, _A_l, [=](int64_t yx) { return &A_ptr[stride * yx + R_dim * N_dim]; });
  std::transform(orders.begin(), orders.end(), _A_upper, [=](int64_t yx) { return A_up[yx]; });

  std::transform(rows.begin(), rows.end(), _Xo_Y, [=](int64_t y) { return X_up[y]; });
  std::transform(rows.begin(), rows.end(), _Xc_Y, [=](int64_t y) { return &X_ptr[y * R_dim]; });
  std::transform(cols.begin(), cols.end(), _Xc_X, [=](int64_t x) { return &_V_data[(x - col_offset) * R_dim]; });
  std::transform(ind.begin(), ind.begin() + N_rows, _Xo_I, [=](int64_t i) { return X_up[i]; });

  std::transform(rows.begin(), rows.end(), urows.begin(), _ACC_Y, 
    [=](int64_t y, int64_t uy) { return &_ACC_data[(y * K1 + uy) * N_dim]; });
  std::transform(cols.begin(), cols.end(), ucols.begin(), _ACC_X, 
    [=](int64_t x, int64_t ux) { return &_ACC_data[((x - col_offset) * K2 + ux) * N_dim]; });
  std::transform(ind.begin(), ind.begin() + N_rows, _ACC_I, [=](int64_t i) { return &_ACC_data[i * N_dim * K1]; });
  std::fill(_ONE_LIST, _ONE_LIST + N_rows, one_data_dev);

  std::transform(ind.begin(), ind.begin() + lenB, _B_x, [=](int64_t i) { return &_V_data[i * stride]; });
  std::transform(ind.begin(), ind.begin() + lenB, _A_sx, [=](int64_t i) { return &_V_data[i * stride + R_dim]; });
  std::transform(ind.begin(), ind.begin() + N_cols, _X_d, [=](int64_t i) { return &X_ptr[N_dim * (i + col_offset)]; });
  std::transform(ind.begin(), ind.begin() + N_cols, _U_i, [=](int64_t i) { return &U_ptr[stride * N_rows + R_dim * i]; });
  
  memset((void*)params, 0, sizeof(BatchedFactorParams));

  params->N_r = R_dim;
  params->N_s = S_dim;
  params->N_upper = N_up;
  params->L_diag = N_cols;
  params->L_nnz = NNZ;
  params->L_lower = lenL;
  params->L_rows = N_rows;
  params->L_tmp = lenB;
  params->Kfwd = K1;
  params->Kback = K2;

  void** ptrs_nnz, **ptrs_diag;
  cudaMalloc((void**)&ptrs_nnz, sizeof(double*) * NNZ_aligned * NZ);
  cudaMalloc((void**)&ptrs_diag, sizeof(double*) * N_rows_aligned * ND);
  cudaMalloc((void**)&params->info, sizeof(int) * N_cols);
  cudaMalloc((void**)&params->ipiv, sizeof(int) * R_dim * N_cols);

  params->U_r = (const double**)&ptrs_nnz[0 * NNZ_aligned];
  params->U_s = (const double**)&ptrs_nnz[1 * NNZ_aligned];
  params->V_x = (const double**)&ptrs_nnz[2 * NNZ_aligned];
  params->A_sx = (const double**)&ptrs_nnz[3 * NNZ_aligned];
  params->A_x = (double**)&ptrs_nnz[4 * NNZ_aligned];
  params->B_x = (double**)&ptrs_nnz[5 * NNZ_aligned];
  params->A_upper = (double**)&ptrs_nnz[6 * NNZ_aligned];
  params->A_s = (double**)&ptrs_nnz[7 * NNZ_aligned];
  params->Xo_Y = (double**)&ptrs_nnz[8 * NNZ_aligned];
  params->Xc_Y = (double**)&ptrs_nnz[9 * NNZ_aligned];
  params->Xc_X = (double**)&ptrs_nnz[10 * NNZ_aligned];
  params->ACC_Y = (double**)&ptrs_nnz[11 * NNZ_aligned];
  params->ACC_X = (double**)&ptrs_nnz[12 * NNZ_aligned];

  params->X_d = (double**)&ptrs_diag[0 * N_rows_aligned];
  params->A_l = (double**)&ptrs_diag[1 * N_rows_aligned];
  params->U_i = (const double**)&ptrs_diag[2 * N_rows_aligned];
  params->ACC_I = (double**)&ptrs_diag[3 * N_rows_aligned];
  params->Xo_I = (double**)&ptrs_diag[4 * N_rows_aligned];
  params->ONE_LIST = (double**)&ptrs_diag[5 * N_rows_aligned];

  params->U_d0 = U_ptr + stride * col_offset;
  params->Xc_d0 = X_ptr + R_dim * col_offset;
  params->X_d0 = X_ptr + N_dim * col_offset;
  params->V_data = _V_data;
  params->A_data = A_ptr;
  params->X_data = X_ptr;
  params->ACC_data = _ACC_data;
  params->ONE_DATA = one_data_dev;

  cudaMemcpy(ptrs_nnz, ptrs_nnz_cpu.data(), sizeof(double*) * NNZ_aligned * NZ, cudaMemcpyHostToDevice);
  cudaMemcpy(ptrs_diag, ptrs_diag_cpu.data(), sizeof(double*) * N_rows_aligned * ND, cudaMemcpyHostToDevice);
}

void batchParamsDestory(BatchedFactorParams* params) {
  if (params->X_d)
    cudaFree(params->X_d);
  if (params->U_r)
    cudaFree(params->U_r);
  if (params->ONE_DATA)
    cudaFree(params->ONE_DATA);
  if (params->info)
    cudaFree(params->info);
  if (params->ipiv)
    cudaFree(params->ipiv);  
}

void batchCholeskyFactor(BatchedFactorParams* params, const CellComm* comm) {
  int64_t U = params->N_upper, R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag;
  double one = 1., zero = 0., minus_one = -1.;
  int info_host = 0;

  level_merge_gpu(params->A_data, N * N * params->L_nnz, comm);

  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &one, 
    params->U_r, N, params->A_x, N, &zero, params->B_x, N, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, N, N, &one, 
    params->B_x, N, params->U_r, N, &zero, params->A_x, R, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, S, N, &one, 
    params->B_x, N, params->U_s, N, &zero, params->A_l, R, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, N, &one, 
    params->A_sx, N, params->U_s, N, &zero, params->A_upper, U, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 1, R, 1, &one, 
    params->ONE_LIST, 1, params->U_i, 1, &one, params->A_x, R + 1, D);

  cublasDgetrfBatched(cublasH, R, params->A_x, R, params->ipiv, params->info, D);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_N, R, S, params->A_x, R, params->ipiv, params->A_l, R, &info_host, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, S, S, R, &minus_one, 
    params->A_s, R, params->A_l, R, &one, params->A_upper, U, D);

  for (int64_t i = 0; i < params->L_lower; i += params->L_tmp) {
    int64_t len = std::min(params->L_lower - i, params->L_tmp);
    cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &one, 
      &params->V_x[i + D], N, &params->A_x[i + D], N, &zero, params->B_x, N, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, N, N, &one, 
      params->B_x, N, &params->U_r[i + D], N, &zero, &params->A_x[i + D], R, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, N, &one,
      params->A_sx, N, &params->U_s[i + D], N, &zero, &params->A_upper[i + D], U, len);
  }

  int64_t offsetU = D + params->L_lower;
  int64_t lenU = params->L_nnz - offsetU;
  for (int64_t i = 0; i < lenU; i += params->L_tmp) {
    int64_t len = std::min(lenU - i, params->L_tmp);
    cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &one, 
      &params->V_x[i + offsetU], N, &params->A_x[i + offsetU], N, &zero, params->B_x, N, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, S, N, &one, 
      params->B_x, N, &params->U_s[i + offsetU], N, &zero, &params->A_s[i + offsetU], R, len);
    cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, S, N, &one,
      params->A_sx, N, &params->U_s[i + offsetU], N, &zero, &params->A_upper[i + offsetU], U, len);
  }
}

void batchForwardULV(BatchedFactorParams* params, const CellComm* comm) {
  int64_t R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag, ONE = 1;
  int64_t K = params->Kfwd;
  double one = 1., zero = 0., minus_one = -1.;
  int info_host = 0;

  level_merge_gpu(params->X_data, params->L_rows * N, comm);
  neighbor_reduce_gpu(params->X_data, N, comm);

  cublasDgemmStridedBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, ONE, N, &one,
    params->U_d0, N, N * N, params->X_d0, N, N * ONE, &zero, params->V_data, R, R * ONE, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, S, ONE, N, &one,
    params->U_s, N, params->X_d, N, &zero, params->Xo_Y, S, D);
  cudaMemsetAsync(params->X_data, 0, sizeof(double) * params->L_rows * R, stream);
  cublasDcopy(cublasH, R * D, params->V_data, 1, params->Xc_d0, 1);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_T, R, ONE, params->A_x, R, params->ipiv, params->Xc_X, R, &info_host, D);

  cudaMemsetAsync(params->ACC_data, 0, sizeof(double) * params->L_rows * N * K, stream);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, S, ONE, R, &one, 
    params->A_s, R, params->Xc_X, R, &zero, params->ACC_Y, N, params->L_nnz);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, S, ONE, K, &minus_one,
    params->ACC_I, N, params->ONE_LIST, K, &one, params->Xo_I, S, params->L_rows);

  cudaMemsetAsync(params->ACC_data, 0, sizeof(double) * params->L_rows * N * K, stream);
  cublasDgemmBatched(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, R, ONE, R, &one, 
    &params->A_x[D], R, &params->Xc_X[D], R, &zero, &params->ACC_Y[D], N, params->L_lower);
  cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, ONE, K, &minus_one,
    params->ACC_data, N, N * K, params->ONE_DATA, K, 0, &one, params->X_data, R, R, params->L_rows);
}

void batchBackwardULV(BatchedFactorParams* params, const CellComm* comm) {
  int64_t R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag, ONE = 1;
  int64_t K = params->Kback;
  double one = 1., zero = 0., minus_one = -1.;
  int info_host;

  neighbor_reduce_gpu(params->X_data, R, comm);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_N, R, ONE, params->A_x, R, params->ipiv, params->Xc_Y, R, &info_host, D);
  neighbor_bcast_gpu(params->X_data, R, comm);

  cudaMemsetAsync(params->ACC_data, 0, sizeof(double) * D * N * K, stream);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, ONE, S, &one, 
    params->A_s, R, params->Xo_Y, S, &zero, params->ACC_X, N, params->L_nnz);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, ONE, R, &one, 
    &params->A_x[D], R, &params->Xc_Y[D], R, &one, &params->ACC_X[D], N, params->L_lower);
  cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, R, ONE, K, &minus_one,
    params->ACC_data, N, N * K, params->ONE_DATA, K, 0, &zero, params->V_data, R, R, D);
  cublasDgetrsBatched(cublasH, CUBLAS_OP_N, R, ONE, params->A_x, R, params->ipiv, params->Xc_X, R, &info_host, D);
  
  cublasDaxpy(cublasH, R * D, &one, params->Xc_d0, 1, params->V_data, 1);
  cublasDgemmStridedBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, ONE, R, &one,
    params->U_d0, N, N * N, params->V_data, R, R * ONE, &zero, params->X_d0, N, N * ONE, D);
  cublasDgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, ONE, S, &one,
    params->U_s, N, params->Xo_Y, S, &one, params->X_d, N, D);
  
  neighbor_bcast_gpu(params->X_data, N, comm);
  dup_bcast_gpu(params->X_data, params->L_rows * N, comm);
}

void lastParamsCreate(BatchedFactorParams* params, double* A, double* X, int64_t N, int64_t S, int64_t clen, const int64_t cdims[]) {
  memset((void*)params, 0, sizeof(BatchedFactorParams));

  params->A_data = A;
  params->X_data = X;
  params->N_r = N;

  int Lwork;
  cusolverDnDgetrf_bufferSize(cusolverH, N, N, A, N, &Lwork);
  Lwork = std::max((int64_t)Lwork, N);
  cudaMalloc((void**)&params->ONE_DATA, sizeof(double) * Lwork);
  params->L_tmp = Lwork;

  std::vector<double> I(N, 1.);
  for (int64_t i = 0; i < clen; i++)
    std::fill(I.begin() + i * S, I.begin() + i * S + cdims[i], 0.);
  cudaMemcpy(params->ONE_DATA, &I[0], sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&params->ipiv, sizeof(int) * N);
  cudaMalloc((void**)&params->info, sizeof(int));
}

void chol_decomp(BatchedFactorParams* params, const CellComm* comm) {
  double* A = params->A_data;
  int64_t N = params->N_r;
  double one = 1.;

  level_merge_gpu(params->A_data, N * N, comm);
  cublasDaxpy(cublasH, N, &one, params->ONE_DATA, 1, A, N + 1);
  cusolverDnDgetrf(cusolverH, N, N, A, N, params->ONE_DATA, params->ipiv, params->info);
}

void chol_solve(BatchedFactorParams* params, const CellComm* comm) {
  const double* A = params->A_data;
  double* X = params->X_data;
  int64_t N = params->N_r;

  level_merge_gpu(X, N, comm);
  cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, A, N, params->ipiv, X, N, params->info);
}

void allocNodes(Node A[], double** Workspace, int64_t* Lwork, const Base basis[], const CSR rels_near[], const CSR rels_far[], const CellComm comm[], int64_t levels) {
  int64_t work_size = 0;

  for (int64_t i = levels; i >= 0; i--) {
    int64_t n_i = 0, ulen = 0, nloc = 0;
    content_length(&n_i, &ulen, &nloc, &comm[i]);
    int64_t nnz = rels_near[i].RowIndex[n_i];
    int64_t nnz_f = rels_far[i].RowIndex[n_i];

    Matrix* arr_m = (Matrix*)malloc(sizeof(Matrix) * (nnz + nnz_f));
    A[i].A = arr_m;
    A[i].S = &arr_m[nnz];
    A[i].lenA = nnz;
    A[i].lenS = nnz_f;

    int64_t dimn = basis[i].dimR + basis[i].dimS;
    int64_t dimn_up = i > 0 ? basis[i - 1].dimN : 0;

    int64_t stride = dimn * dimn;
    A[i].sizeA = stride * nnz;
    A[i].sizeU = stride * ulen + n_i * basis[i].dimR;
    allocBufferedList((void**)&A[i].A_ptr, (void**)&A[i].A_buf, sizeof(double), A[i].sizeA);
    allocBufferedList((void**)&A[i].X_ptr, (void**)&A[i].X_buf, sizeof(double), dimn * ulen);
    allocBufferedList((void**)&A[i].U_ptr, (void**)&A[i].U_buf, sizeof(double), A[i].sizeU);

    std::copy(basis[i].U, &basis[i].U[A[i].sizeU], A[i].U_buf);

    int64_t k1, k2;
    countMaxIJ(&k1, &k2, &rels_near[i]);
    int64_t acc_required = std::max(k1 * ulen, k2 * n_i);
    int64_t work_required = std::max(n_i * stride, (acc_required + n_i) * dimn);
    work_size = std::max(work_size, work_required);

    for (int64_t x = 0; x < n_i; x++) {
      for (int64_t yx = rels_near[i].RowIndex[x]; yx < rels_near[i].RowIndex[x + 1]; yx++)
        arr_m[yx] = (Matrix) { &A[i].A_buf[yx * stride], dimn, dimn, dimn }; // A

      for (int64_t yx = rels_far[i].RowIndex[x]; yx < rels_far[i].RowIndex[x + 1]; yx++)
        arr_m[yx + nnz] = (Matrix) { NULL, basis[i].dimS, basis[i].dimS, dimn_up }; // S
    }

    if (i < levels) {
      int64_t ploc = 0;
      content_length(NULL, NULL, &ploc, &comm[i + 1]);
      int64_t seg = basis[i + 1].dimS;

      for (int64_t j = 0; j < rels_near[i].N; j++) {
        int64_t x0 = std::get<0>(comm[i].LocalChild[j + nloc]) - ploc;
        int64_t lenx = std::get<1>(comm[i].LocalChild[j + nloc]);

        for (int64_t ij = rels_near[i].RowIndex[j]; ij < rels_near[i].RowIndex[j + 1]; ij++) {
          int64_t li = rels_near[i].ColIndex[ij];
          int64_t y0 = std::get<0>(comm[i].LocalChild[li]);
          int64_t leny = std::get<1>(comm[i].LocalChild[li]);
          
          for (int64_t x = 0; x < lenx; x++)
            if ((x + x0) >= 0 && (x + x0) < rels_far[i + 1].N)
              for (int64_t yx = rels_far[i + 1].RowIndex[x + x0]; yx < rels_far[i + 1].RowIndex[x + x0 + 1]; yx++)
                for (int64_t y = 0; y < leny; y++)
                  if (rels_far[i + 1].ColIndex[yx] == (y + y0))
                    A[i + 1].S[yx].A = &A[i].A[ij].A[(y * dimn + x) * seg];
        }
      }
    }
  }
  
  set_work_size(work_size, Workspace, Lwork);
  for (int64_t i = levels; i > 0; i--) {
    int64_t ibegin = 0, N_rows = 0, N_cols = 0;
    content_length(&N_cols, &N_rows, &ibegin, &comm[i]);
    int64_t nnz = A[i].lenA;
    int64_t dimc = basis[i].dimR;
    int64_t dimr = basis[i].dimS;

    int64_t n_next = basis[i - 1].dimR + basis[i - 1].dimS;
    int64_t ibegin_next = 0;
    content_length(NULL, NULL, &ibegin_next, &comm[i - 1]);

    std::vector<double*> A_next(nnz);
    for (int64_t x = 0; x < N_cols; x++)
      for (int64_t yx = rels_near[i].RowIndex[x]; yx < rels_near[i].RowIndex[x + 1]; yx++) {
        int64_t y = rels_near[i].ColIndex[yx];
        std::pair<int64_t, int64_t> px = comm[i].LocalParent[x + ibegin];
        std::pair<int64_t, int64_t> py = comm[i].LocalParent[y];
        int64_t ij = rels_near[i - 1].lookupIJ(std::get<0>(py), std::get<0>(px) - ibegin_next);
        A_next[yx] = &A[i - 1].A_ptr[(std::get<1>(py) * n_next + std::get<1>(px)) * basis[i].dimS + ij * n_next * n_next];
      }

    std::vector<double*> X_next(N_rows);
    for (int64_t x = 0; x < N_rows; x++) { 
      std::pair<int64_t, int64_t> p = comm[i].LocalParent[x];
      X_next[x] = &A[i - 1].X_ptr[std::get<1>(p) * basis[i].dimS + std::get<0>(p) * n_next];
    }

    batchParamsCreate(&A[i].params, dimc, dimr, A[i].U_ptr, A[i].A_ptr, A[i].X_ptr, n_next, &A_next[0], &X_next[0],
      *Workspace, work_size, N_rows, N_cols, ibegin, &rels_near[i].ColIndex[0], &rels_near[i].RowIndex[0]);
  }

  int64_t child = std::get<0>(comm[0].LocalChild[0]);
  int64_t clen = std::get<1>(comm[0].LocalChild[0]);
  std::vector<int64_t> cdims(clen);
  if (child >= 0)
    for (int64_t i = 0; i < clen; i++)
      cdims[i] = basis[1].DimsLr[child + i];
  else
    cdims.emplace_back(basis[0].Dims[0]);
  int64_t low_s = clen > 0 ? basis[1].dimS : 0;
  lastParamsCreate(&A[0].params, A[0].A_ptr, A[0].X_ptr, basis[0].dimN, low_s, cdims.size(), &cdims[0]);
}

void node_free(Node* node) {
  freeBufferedList(node->A_ptr, node->A_buf);
  freeBufferedList(node->X_ptr, node->X_buf);
  freeBufferedList(node->U_ptr, node->U_buf);
  free(node->A);
  batchParamsDestory(&node->params);
}

void factorA_mov_mem(Node A[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    flushBuffer('S', A[i].A_ptr, A[i].A_buf, sizeof(double), A[i].sizeA);
    flushBuffer('S', A[i].U_ptr, A[i].U_buf, sizeof(double), A[i].sizeU);
  }
}

double solveRelErr(const double* X, const double* ref, int64_t lenX) {
  double err[2] = { 0., 0. };
  for (int64_t i = 0; i < lenX; ++i) {
    std::complex<double> diff = X[i] - ref[i];
    err[0] += std::norm(diff);
    err[1] += std::norm(ref[i]);
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return std::sqrt(err[0] / err[1]);
}

void mesh_unit_sphere(double* bodies, int64_t nbodies) {
  const double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians
  for (int64_t i = 0; i < nbodies; i++) {
    const double y = 1. - ((double)i / ((double)nbodies - 1)) * 2.;  // y goes from 1 to -1

    // Note: setting constant radius = 1 will produce a cylindrical shape
    const double radius = std::sqrt(std::max(0., 1. - y * y));  // radius at y
    const double theta = (double)i * phi;

    const double x = radius * std::cos(theta);
    const double z = radius * std::sin(theta);
    bodies[i * 3] = x;
    bodies[i * 3 + 1] = y;
    bodies[i * 3 + 2] = z;
  }
}

void body_neutral_charge(double X[], int64_t nbodies, double cmax, unsigned int seed) {
  if (seed > 0)
    srand(seed);

  double cmax2 = cmax * 2;
  for (int64_t i = 0; i < nbodies; i++)
    X[i] = ((double)rand() / RAND_MAX) * cmax2 - cmax;
}

class DenseMatrix {
public:
  int64_t N;
  std::vector<double> A;
  std::vector<int64_t> rowOffsets;
  int32_t mpi_rank; MPI_Comm world;
  DenseMatrix(int64_t localM, int64_t N, MPI_Comm world) : N(N), world(world) {
    if (0 < N) {
      A.resize(localM * N, 0.); int mpi_size = 1;
      MPI_Comm_rank(world, &mpi_rank); MPI_Comm_size(world, &mpi_size);
      rowOffsets.resize(mpi_size + 1); rowOffsets[0] = 0;
      MPI_Allgather(&localM, 1, MPI_INT64_T, &rowOffsets[1], 1, MPI_INT64_T, world);
      std::inclusive_scan(rowOffsets.begin() + 1, rowOffsets.end(), rowOffsets.begin() + 1);
    }
  }

  void matVecA(const DenseMatrix& X, double* B) const {
    assert(X.rowOffsets.back() == N && X.N == int64_t(1));
    std::vector<double> X_in(N);
    std::vector<int> recvcounts(rowOffsets.size() - 1), displs(rowOffsets.size() - 1);
    std::transform(X.rowOffsets.begin() + 1, X.rowOffsets.end(), X.rowOffsets.begin(), recvcounts.begin(), [](int64_t end, int64_t begin) { return (int)(end - begin); });
    std::transform(X.rowOffsets.begin(), std::prev(X.rowOffsets.end()), displs.begin(), [](int64_t begin) { return (int)begin; });

    MPI_Allgatherv(&(X.A)[0], int(X.rowOffsets[mpi_rank + 1] - X.rowOffsets[mpi_rank]), MPI_DOUBLE, &X_in[0], &recvcounts[0], &displs[0], MPI_DOUBLE, world);
    int M = int(rowOffsets[mpi_rank + 1] - rowOffsets[mpi_rank]);
    auto gemv = [](int m, int n, const double* a, const double* x, double* b) {
      char transA = 'T'; int incx = 1; double one = 1., zero = 0.; dgemv_(&transA, &n, &m, &one, a, &n, x, &incx, &zero, b, &incx);
    };
    gemv(M, N, &A[0], &X_in[0], B);
  }

  void AijBlock(int64_t mA, int64_t nA, int64_t iA, int64_t jA, double* A_out, int64_t stride) const {
    int64_t row_begin = std::max(rowOffsets[mpi_rank], iA), row_end = std::min(rowOffsets[mpi_rank + 1], iA + mA);
    int64_t col_begin = std::max(int64_t(0), jA), col_end = std::min(N, jA + nA);
    if (row_begin < row_end && col_begin < col_end) {
      int64_t rows = row_end - row_begin, cols = col_end - col_begin;
      const double* src = &A[(row_begin - rowOffsets[mpi_rank]) * N + col_begin];
      double* dst = &A_out[(row_begin - iA) + (col_begin - jA) * stride];
      for (int64_t x = 0; x < cols; ++x) for (int64_t y = 0; y < rows; ++y)
      { dst[y + stride * x] = src[y * N + x]; }
    }
  }

  void transposeA(DenseMatrix& B) const {
    assert(B.rowOffsets.back() == N && rowOffsets.back() == B.N);
    int64_t MA = rowOffsets[mpi_rank + 1] - rowOffsets[mpi_rank], MB = B.rowOffsets[mpi_rank + 1] - B.rowOffsets[mpi_rank];
    for (int i = 0; i < int(rowOffsets.size()) - 1; ++i) {
      int64_t NA = B.rowOffsets[i + 1] - B.rowOffsets[i], NB = rowOffsets[i + 1] - rowOffsets[i];
      std::vector<double> AH(MA * NA), BH(MB * NB); AijBlock(MA, NA, rowOffsets[mpi_rank], B.rowOffsets[i], &AH[0], MA);
      MPI_Sendrecv(&AH[0], AH.size(), MPI_DOUBLE, i, 0, &BH[0], BH.size(), MPI_DOUBLE, i, 0, world, MPI_STATUS_IGNORE);
      memcpy2d(&(B.A)[rowOffsets[i]], &BH[0], NB, MB, B.N, NB);
    }
  }

  void AijSetZero(int64_t mA, int64_t nA, int64_t iA, int64_t jA) {
    int64_t row_begin = std::max(rowOffsets[mpi_rank], iA), row_end = std::min(rowOffsets[mpi_rank + 1], iA + mA);
    int64_t col_begin = std::max(int64_t(0), jA), col_end = std::min(N, jA + nA);
    if (row_begin < row_end && col_begin < col_end) {
      int64_t rows = row_end - row_begin, cols = col_end - col_begin;
      double* src = &A[(row_begin - rowOffsets[mpi_rank]) * N + col_begin];
      for (int64_t y = 0; y < rows; ++y) { std::fill_n(&src[y * N], cols, 0.); }
    }
  }
};

int main(int argc, char* argv[]) {
  cudaStream_t stream = (cudaStream_t)init_libs(&argc, &argv);
  double prog_time = MPI_Wtime();

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 2048;
  int64_t leaf_size = argc > 2 ? atol(argv[2]) : 512;
  int64_t rank_max = argc > 3 ? atol(argv[3]) : 400;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  int64_t levels = (int64_t)log2((double)Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  int64_t ncells = Nleaf + Nleaf - 1;
  
  double* body = (double*)malloc(sizeof(double) * Nbody * 3);
  Cell* cell = (Cell*)calloc(ncells, sizeof(Cell));
  CSR cellNear, cellFar;
  CSR* rels_far = (CSR*)calloc(levels + 1, sizeof(CSR));
  CSR* rels_near = (CSR*)calloc(levels + 1, sizeof(CSR));
  CellComm* cell_comm = (CellComm*)calloc(levels + 1, sizeof(CellComm));
  Base* basis = (Base*)calloc(levels + 1, sizeof(Base));
  Node* nodes = (Node*)malloc(sizeof(Node) * (levels + 1));

  mesh_unit_sphere(body, Nbody);
  buildTree(&ncells, cell, body, Nbody, levels);

  traverse('N', &cellNear, ncells, cell, 0.);
  traverse('F', &cellFar, ncells, cell, 0.);

  Laplace3D eval(1.e-1);

  buildComm(cell_comm, ncells, cell, &cellFar, &cellNear, levels);
  for (int64_t i = 0; i <= levels; i++)
    cell_comm[i].stream = stream;
  relations(rels_near, &cellNear, levels, cell_comm);
  relations(rels_far, &cellFar, levels, cell_comm);

  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &cell_comm[levels]);
  int64_t gbegin = cell_comm[levels].iGlobal(lbegin);
  int64_t body_local[2] = { cell[gbegin].Body[0], cell[gbegin + llen - 1].Body[1] };

  MPI_Barrier(MPI_COMM_WORLD);
  double construct_time = MPI_Wtime();
  buildBasis(eval, basis, cell, levels, cell_comm, body, Nbody, rank_max);

  MPI_Barrier(MPI_COMM_WORLD);
  construct_time = MPI_Wtime() - construct_time;

  double* Workspace = NULL;
  int64_t Lwork = 0;
  allocNodes(nodes, &Workspace, &Lwork, basis, rels_near, rels_far, cell_comm, levels);

  evalD(eval, nodes[levels].A, &cellNear, cell, body, &cell_comm[levels]);
  for (int64_t i = 0; i <= levels; i++)
    evalS(eval, nodes[i].S, &basis[i], &rels_far[i], &cell_comm[i]);

  int64_t lenX = rels_near[levels].N * basis[levels].dimN;
  std::vector<double> X1(lenX), X2(lenX);
  std::fill(X2.begin(), X2.end(), 0.);

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  DenseMatrix dataA(lenX, Nbody, MPI_COMM_WORLD), Xbody(lenX, 1, MPI_COMM_WORLD); // DenseMatrix is row-major
  gen_matrix(eval, Nbody, lenX, body, &body[body_local[0] * 3], dataA.A.data(), Nbody);
  body_neutral_charge(Xbody.A.data(), lenX, 1., 999 ^ mpi_rank);
  dataA.matVecA(Xbody, X1.data());
  
  factorA_mov_mem(nodes, levels);
  MPI_Barrier(MPI_COMM_WORLD);
  double factor_time = MPI_Wtime();

  for (int64_t i = levels; i > 0; i--)
    batchCholeskyFactor(&nodes[i].params, &cell_comm[i]);
  chol_decomp(&nodes[0].params, &cell_comm[0]);

  cudaStreamSynchronize(stream);
  MPI_Barrier(MPI_COMM_WORLD);

  factor_time = MPI_Wtime() - factor_time;

  cudaMemcpy(&nodes[levels].X_ptr[lbegin * basis[levels].dimN], X1.data(), lenX * sizeof(double), cudaMemcpyHostToDevice);

  MPI_Barrier(MPI_COMM_WORLD);
  double solve_time = MPI_Wtime();

  for (int64_t i = levels; i > 0; i--)
    batchForwardULV(&nodes[i].params, &cell_comm[i]);
  chol_solve(&nodes[0].params, &cell_comm[0]);
  for (int64_t i = 1; i <= levels; i++)
    batchBackwardULV(&nodes[i].params, &cell_comm[i]);

  cudaStreamSynchronize(stream);
  MPI_Barrier(MPI_COMM_WORLD);

  solve_time = MPI_Wtime() - solve_time;

  cudaMemcpy(X1.data(), &nodes[levels].X_ptr[lbegin * basis[levels].dimN], lenX * sizeof(double), cudaMemcpyDeviceToHost);

  loadX(X2.data(), basis[levels].dimN, Xbody.A.data(), 0, llen, &cell[gbegin]);
  double err = solveRelErr(X1.data(), X2.data(), lenX);

  prog_time = MPI_Wtime() - prog_time;

  if (mpi_rank == 0)
    printf("LORASP: %d,%d,%d\n"
      "Construct: %lf s.\n"
      "Factorize: %lf s.\n"
      "Solution: %lf s.\n"
      "Err: %le\n"
      "Program: %lf s.\n",
      (int)Nbody, (int)(Nbody / Nleaf), (int)mpi_size,
      construct_time, factor_time, solve_time, err, prog_time);

  for (int64_t i = 0; i <= levels; i++) {
    basis_free(&basis[i]);
    node_free(&nodes[i]);
  }
  cellComm_free(cell_comm, levels);
  
  free(body);
  free(cell);
  free(rels_far);
  free(rels_near);
  free(cell_comm);
  free(basis);
  free(nodes);
  set_work_size(0, &Workspace, &Lwork);

  fin_libs();
  return 0;
}

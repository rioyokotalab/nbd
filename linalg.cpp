
#include <linalg.hpp>
#include <kernel.hpp>
#include <build_tree.hpp>
#include <basis.hpp>
#include <comm-mpi.hpp>

#include "mkl.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <tuple>
#include <iostream>

void mmult(char ta, char tb, const Matrix* A, const Matrix* B, Matrix* C, double alpha, double beta) {
  int64_t k = ta == 'N' ? A->N : A->M;
  CBLAS_TRANSPOSE tac = ta == 'N' ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE tbc = tb == 'N' ? CblasNoTrans : CblasTrans;
  int64_t lda = 1 < A->LDA ? A->LDA : 1;
  int64_t ldb = 1 < B->LDA ? B->LDA : 1;
  int64_t ldc = 1 < C->LDA ? C->LDA : 1;
  cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, lda, B->A, ldb, beta, C->A, ldc);
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

void compute_basis(const EvalDouble& eval, int64_t rank_max, 
  int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t Nclose, const double Cbodies[], int64_t Nfar, const double Fbodies[]) {

  if (M > 0 && (Nclose > 0 || Nfar > 0)) {
    int64_t ldm = std::max(M, Nclose + Nfar);
    std::vector<double> Aall(M * ldm, 0.), U(M * M), S(M * 2);
    std::vector<long long> ipiv(M);
    gen_matrix(eval, Nclose, M, Cbodies, Xbodies, &Aall[0], ldm);
    gen_matrix(eval, Nfar, M, Fbodies, Xbodies, &Aall[Nclose], ldm);

    std::fill(ipiv.begin(), ipiv.end(), 0);
    LAPACKE_dgeqp3(LAPACK_COL_MAJOR, Nclose + Nfar, M, &Aall[0], ldm, &ipiv[0], &S[0]);
    LAPACKE_dlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, 0., 0., &Aall[1], ldm);

    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'A', M, M, &Aall[0], ldm, &S[0], NULL, M, &U[0], M, &S[M]);
    int64_t rank = rank_max <= 0 ? M : std::min(rank_max, M);    
    if (rank > 0) {
      if (rank < M)
        LAPACKE_dgesv(LAPACK_COL_MAJOR, rank, M - rank, &U[0], M, (long long*)&S[0], &U[rank * M], M);
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
        vdMul(rank, &S[0], &U[i * M], &A[(M + i) * LDA]);
    }
  }
}


void mat_vec_reference(const EvalDouble& eval, int64_t begin, int64_t end, double B[], int64_t nbodies, const double* bodies, const double Xbodies[]) {
  int64_t M = end - begin;
  int64_t N = nbodies;
  int64_t size = 1024;
  std::vector<double> A(size * size);
  
  for (int64_t i = 0; i < M; i += size) {
    int64_t y = begin + i;
    int64_t m = std::min(M - i, size);
    const double* bi = &bodies[y * 3];
    for (int64_t j = 0; j < N; j += size) {
      const double* bj = &bodies[j * 3];
      int64_t n = std::min(N - j, size);
      gen_matrix(eval, m, n, bi, bj, &A[0], size);
      cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, 1., &A[0], size, &Xbodies[j], 1, 1., &B[i], 1);
    }
  }
}

void set_work_size(int64_t Lwork, double** D_DATA, int64_t* D_DATA_SIZE) {
  if (Lwork > *D_DATA_SIZE) {
    *D_DATA_SIZE = Lwork;
    if (*D_DATA)
      free(*D_DATA);
    *D_DATA = (double*)malloc(sizeof(double) * Lwork);
  }
  else if (Lwork <= 0) {
    *D_DATA_SIZE = 0;
    if (*D_DATA)
      free(*D_DATA);
  }
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

void batchParamsCreate(struct BatchedFactorParams* params, int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, double* X_ptr, int64_t N_up, double** A_up, double** X_up,
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
  one_data_dev = (double*)malloc(sizeof(double) * N_rows);
  memcpy(one_data_dev, &one_data[0], sizeof(double) * N_rows);

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
  
  memset((void*)params, 0, sizeof(struct BatchedFactorParams));

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
  ptrs_nnz = (void**)malloc(sizeof(double*) * NNZ_aligned * NZ);
  ptrs_diag = (void**)malloc(sizeof(double*) * N_rows_aligned * ND);
  params->info = (int*)malloc(sizeof(int) * N_cols);
  params->ipiv = (int64_t*)malloc(sizeof(int64_t) * R_dim * N_cols);

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

  memcpy(ptrs_nnz, ptrs_nnz_cpu.data(), sizeof(double*) * NNZ_aligned * NZ);
  memcpy(ptrs_diag, ptrs_diag_cpu.data(), sizeof(double*) * N_rows_aligned * ND);
}

void batchParamsDestory(struct BatchedFactorParams* params) {
  if (params->X_d)
    free(params->X_d);
  if (params->U_r)
    free(params->U_r);
  if (params->ONE_DATA)
    free(params->ONE_DATA);
  if (params->info)
    free(params->info);
  if (params->ipiv)
    free(params->ipiv);  
}

void batchCholeskyFactor(struct BatchedFactorParams* params, const struct ColCommMPI* comm) {
  int64_t U = params->N_upper, R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag;
  double one = 1., zero = 0., minus_one = -1.;
  int info_host = 0;

  level_merge_cpu(params->A_data, N * N * params->L_nnz, comm);

  for (int64_t i = 0; i < D; i++) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, N, N, one,
      params->U_r[i], N, params->A_x[i], N, zero, params->B_x[i], N);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, N, N, one, 
      params->B_x[i], N, params->U_r[i], N, zero, params->A_x[i], R);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, S, N, one, 
      params->B_x[i], N, params->U_s[i], N, zero, params->A_l[i], R);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, S, S, N, one, 
      params->A_sx[i], N, params->U_s[i], N, zero, params->A_upper[i], U);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, R, 1, one, 
      params->ONE_LIST[i], 1, params->U_i[i], 1, one, params->A_x[i], R + 1);

    LAPACKE_dgetrf(LAPACK_COL_MAJOR, R, R, params->A_x[i], R, (long long*)&params->ipiv[i * R]);
    LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', R, S, params->A_x[i], R, (long long*)&params->ipiv[i * R], params->A_l[i], R);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, S, S, R, minus_one,
      params->A_s[i], R, params->A_l[i], R, one, params->A_upper[i], U);
  }

  for (int64_t i = 0; i < params->L_lower; i += params->L_tmp) {
    int64_t len = std::min(params->L_lower - i, params->L_tmp);
    for (int64_t j = 0; j < len; j++) {
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, N, N, one,
        params->V_x[i + D + j], N, params->A_x[i + D + j], N, zero, params->B_x[j], N);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, N, N, one,
        params->B_x[j], N, params->U_r[i + D + j], N, zero, params->A_x[i + D + j], R);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, S, S, N, one,
        params->A_sx[j], N, params->U_s[i + D + j], N, zero, params->A_upper[i + D + j], U);
    }
  }

  int64_t offsetU = D + params->L_lower;
  int64_t lenU = params->L_nnz - offsetU;
  for (int64_t i = 0; i < lenU; i += params->L_tmp) {
    int64_t len = std::min(lenU - i, params->L_tmp);
    for (int64_t j = 0; j < len; j++) {
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, N, N, one,
        params->V_x[i + offsetU + j], N, params->A_x[i + offsetU + j], N, zero, params->B_x[j], N);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, S, N, one,
        params->B_x[j], N, params->U_s[i + offsetU + j], N, zero, params->A_s[i + offsetU + j], R);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, S, S, N, one,
        params->A_sx[j], N, params->U_s[i + offsetU + j], N, zero, params->A_upper[i + offsetU + j], U);
    }
  }
}

void batchForwardULV(struct BatchedFactorParams* params, const struct ColCommMPI* comm) {
  int64_t R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag, ONE = 1;
  int64_t K = params->Kfwd;
  double one = 1., zero = 0., minus_one = -1.;
  int info_host = 0;

  level_merge_cpu(params->X_data, params->L_rows * N, comm);
  neighbor_reduce_cpu(params->X_data, N, comm);

  for (int64_t i = 0; i < D; i++) {
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, R, ONE, N, one,
      &params->U_d0[i * N * N], N, &params->X_d0[i * N], N, zero, &params->V_data[i * R], R);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, S, ONE, N, one,
      params->U_s[i], N, params->X_d[i], N, zero, params->Xo_Y[i], S);
  }
  std::fill(params->X_data, &params->X_data[params->L_rows * R], 0.);
  cblas_dcopy(R * D, params->V_data, 1, params->Xc_d0, 1);

  for (int64_t i = 0; i < D; i++)
    LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'T', R, ONE, params->A_x[i], R, (long long*)&params->ipiv[i * R], params->Xc_X[i], R);

  std::fill(params->ACC_data, &params->ACC_data[params->L_rows * N * K], 0.);
  for (int64_t i = 0; i < params->L_nnz; i++)
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, S, ONE, R, one,
      params->A_s[i], R, params->Xc_X[i], R, zero, params->ACC_Y[i], N);
  for (int64_t i = 0; i < params->L_rows; i++)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, S, ONE, K, minus_one,
      params->ACC_I[i], N, params->ONE_LIST[i], K, one, params->Xo_I[i], S);

  std::fill(params->ACC_data, &params->ACC_data[params->L_rows * N * K], 0.);
  for (int64_t i = 0; i < params->L_lower; i++)
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, R, ONE, R, one,
      params->A_x[i + D], R, params->Xc_X[i + D], R, zero, params->ACC_Y[i + D], N);
  for (int64_t i = 0; i < params->L_rows; i++)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, ONE, K, minus_one,
      &params->ACC_data[i * N * K], N, params->ONE_DATA, K, one, &params->X_data[i * R], R);
}

void batchBackwardULV(struct BatchedFactorParams* params, const struct ColCommMPI* comm) {
  int64_t R = params->N_r, S = params->N_s, N = R + S, D = params->L_diag, ONE = 1;
  int64_t K = params->Kback;
  double one = 1., zero = 0., minus_one = -1.;
  int info_host;

  neighbor_reduce_cpu(params->X_data, R, comm);
  for (int64_t i = 0; i < D; i++)
    LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', R, ONE, params->A_x[i], R, (long long*)&params->ipiv[i * R], params->Xc_Y[i], R);
  neighbor_bcast_cpu(params->X_data, R, comm);

  std::fill(params->ACC_data, &params->ACC_data[D * N * K], 0.);
  for (int64_t i = 0; i < params->L_nnz; i++)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, ONE, S, one,
      params->A_s[i], R, params->Xo_Y[i], S, zero, params->ACC_X[i], N);
  for (int64_t i = 0; i < params->L_lower; i++)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, ONE, R, one,
      params->A_x[i + D], R, params->Xc_Y[i + D], R, one, params->ACC_X[i + D], N);
  for (int64_t i = 0; i < D; i++) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R, ONE, K, minus_one,
      &params->ACC_data[i * N * K], N, params->ONE_DATA, K, zero, &params->V_data[i * R], R);
    LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', R, ONE, params->A_x[i], R, (long long*)&params->ipiv[i * R], params->Xc_X[i], R);
  }

  cblas_daxpy(R * D, one, params->Xc_d0, 1, params->V_data, 1);
  for (int64_t i = 0; i < D; i++) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, ONE, R, one,
      &params->U_d0[i * N * N], N, &params->V_data[i * R], R, zero, &params->X_d0[i * N], N);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, ONE, S, one,
      params->U_s[i], N, params->Xo_Y[i], S, one, params->X_d[i], N);
  }
  
  neighbor_bcast_cpu(params->X_data, N, comm);
  dup_bcast_cpu(params->X_data, params->L_rows * N, comm);
}

void lastParamsCreate(struct BatchedFactorParams* params, double* A, double* X, int64_t N, int64_t S, int64_t clen, const int64_t cdims[]) {
  memset((void*)params, 0, sizeof(struct BatchedFactorParams));

  params->A_data = A;
  params->X_data = X;
  params->N_r = N;

  int Lwork;
  Lwork = N;
  params->ONE_DATA = (double*)malloc(sizeof(double) * Lwork);
  params->L_tmp = Lwork;

  std::vector<double> I(N, 1.);
  for (int64_t i = 0; i < clen; i++)
    std::fill(I.begin() + i * S, I.begin() + i * S + cdims[i], 0.);
  memcpy(params->ONE_DATA, &I[0], sizeof(double) * N);
  params->ipiv = (int64_t*)malloc(sizeof(int64_t) * N);
  params->info = (int*)malloc(sizeof(int));
}

void chol_decomp(struct BatchedFactorParams* params, const struct ColCommMPI* comm) {
  double* A = params->A_data;
  int64_t N = params->N_r;
  double one = 1.;

  level_merge_cpu(params->A_data, N * N, comm);
  cblas_daxpy(N, one, params->ONE_DATA, 1, A, N + 1);
  *params->info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, N, N, A, N, (long long*)params->ipiv);
}

void chol_solve(struct BatchedFactorParams* params, const struct ColCommMPI* comm) {
  const double* A = params->A_data;
  double* X = params->X_data;
  int64_t N = params->N_r;

  level_merge_cpu(X, N, comm);
  LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', N, 1, A, N, (long long*)params->ipiv, X, N);
}

template <typename T>
void memcpy2d(T* dst, const T* src, int64_t rows, int64_t cols, int64_t ld_dst, int64_t ld_src) {
  if (rows == ld_dst && rows == ld_src)
    std::copy(src, src + rows * cols, dst);
  else 
    for (int64_t i = 0; i < cols; i++)
      std::copy(&src[i * ld_src], &src[i * ld_src + rows], &dst[i * ld_dst]);
}

void allocNodes(Node A[], double** Workspace, int64_t* Lwork, int64_t rank, int64_t leaf, int64_t branches, const Cell cells[], const Base basis[], const CSR rels_near[], const CSR rels_far[], const ColCommMPI comm[], int64_t levels) {
  int64_t work_size = 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t n_i = 0, ulen = 0, nloc = 0;
    content_length(&n_i, &ulen, &nloc, &comm[i]);
    int64_t nnz = rels_near[i].RowIndex[n_i];
    int64_t nnz_f = rels_far[i].RowIndex[n_i];

    Matrix* arr_m = (Matrix*)malloc(sizeof(Matrix) * (nnz + nnz_f));
    A[i].A = arr_m;
    A[i].S = &arr_m[nnz];
    A[i].lenA = nnz;
    A[i].lenS = nnz_f;

    A[i].X = std::vector<Matrix>(ulen);
    A[i].Xc = std::vector<Matrix>(ulen);
    A[i].Xo = std::vector<Matrix>(ulen);
    A[i].B = std::vector<Matrix>(ulen);

    int64_t dimn = i == levels ? leaf : (rank * branches);
    int64_t dims = std::min(rank, dimn);
    int64_t dimr = dimn - dims;
    int64_t dimn_up = rank * branches;

    int64_t stride = dimn * dimn;
    A[i].sizeA = stride * nnz;
    A[i].sizeU = stride * ulen + n_i * dimr;
    A[i].A_ptr = (double*)calloc(A[i].sizeA, sizeof(double));
    A[i].X_ptr = (double*)calloc(dimn * ulen, sizeof(double));
    A[i].B_ptr = (double*)calloc(dimn * ulen, sizeof(double));
    A[i].U_ptr = (double*)calloc(A[i].sizeU, sizeof(double));

    for (int64_t j = nloc; j < (nloc + n_i); j++) {
      double* Uc_ptr = A[i].U_ptr + j * stride;
      double* Uo_ptr = Uc_ptr + dimr * dimn;
      const double* U_src = basis[i].Uo[j].A;

      int64_t Nc = basis[i].Dims[j] - basis[i].DimsLr[j];
      int64_t No = basis[i].DimsLr[j];
      int64_t M = basis[i].Dims[j];
      int64_t LD = basis[i].Uo[j].LDA;

      memcpy2d(Uc_ptr, &U_src[No * LD], M, Nc, LD, dimn);
      memcpy2d(Uo_ptr, U_src, M, No, LD, dimn);
      double *Ui_ptr = A[i].U_ptr + ulen * stride + (j - nloc) * dimr; 
      for (int64_t k = Nc; k < dimr; k++)
        Ui_ptr[k] = 1.;
    }
    neighbor_bcast_cpu(A[i].U_ptr, stride, &comm[i]);

    for (int64_t x = 0; x < n_i; x++) {
      for (int64_t yx = rels_near[i].RowIndex[x]; yx < rels_near[i].RowIndex[x + 1]; yx++)
        arr_m[yx] = (Matrix) { &A[i].A_ptr[yx * stride], dimn, dimn, dimn }; // A
    }

    for (int64_t y = 0; y < ulen; y++) {
      A[i].X[y] = (Matrix) { &(A[i].X_ptr)[y * dimn], dimn, 1, dimn }; // X
      A[i].B[y] = (Matrix) { &(A[i].B_ptr)[y * dimn], dimn, 1, dimn }; // B
    }

    if (0 < i) {
      int64_t ploc = 0;
      content_length(NULL, NULL, &ploc, &comm[i - 1]);

      for (int64_t x = 0; x < n_i; x++) {
        for (int64_t yx = rels_far[i].RowIndex[x]; yx < rels_far[i].RowIndex[x + 1]; yx++) {
          int64_t y = rels_far[i].ColIndex[yx];
          std::pair<int64_t, int64_t> px = basis[i].LocalParent[x + nloc];
          std::pair<int64_t, int64_t> py = basis[i].LocalParent[y];
          int64_t ij = rels_near[i - 1].lookupIJ(std::get<0>(py), std::get<0>(px) - ploc);
          double* s_ptr = &A[i - 1].A_ptr[(std::get<1>(py) * dimn_up + std::get<1>(px)) * dims + ij * dimn_up * dimn_up];

          arr_m[yx + nnz] = (Matrix) { s_ptr, dims, dims, dimn_up }; // S
        }
      }

      for (int64_t y = 0; y < ulen; y++) {
        std::pair<int64_t, int64_t> p = basis[i].LocalParent[y];
        double* x_next = &(A[i - 1].X_ptr)[std::get<1>(p) * dims + std::get<0>(p) * dimn_up];
        A[i].Xo[y] = (Matrix) { x_next, dims, 1, dims }; // Xo

        double* b_next = &(A[i - 1].B_ptr)[std::get<1>(p) * dims + std::get<0>(p) * dimn_up];
        A[i].Xc[y] = (Matrix) { b_next, dims, 1, dims }; // Xc
      }
    }

    int64_t k1, k2;
    countMaxIJ(&k1, &k2, &rels_near[i]);
    int64_t acc_required = std::max(k1 * ulen, k2 * n_i);
    int64_t work_required = std::max(n_i * stride, (acc_required + n_i) * dimn);
    work_size = std::max(work_size, work_required);
  }
  
  set_work_size(work_size, Workspace, Lwork);
  for (int64_t i = levels; i > 0; i--) {
    int64_t ibegin = 0, N_rows = 0, N_cols = 0;
    content_length(&N_cols, &N_rows, &ibegin, &comm[i]);
    int64_t nnz = A[i].lenA;
    int64_t dimn = i == levels ? leaf : (rank * branches);
    int64_t dims = std::min(rank, dimn);
    int64_t dimr = dimn - dims;
    int64_t dimn_up = rank * branches;
    int64_t ibegin_next = 0;
    content_length(NULL, NULL, &ibegin_next, &comm[i - 1]);

    std::vector<double*> A_next(nnz);
    for (int64_t x = 0; x < N_cols; x++)
      for (int64_t yx = rels_near[i].RowIndex[x]; yx < rels_near[i].RowIndex[x + 1]; yx++) {
        int64_t y = rels_near[i].ColIndex[yx];
        std::pair<int64_t, int64_t> px = basis[i].LocalParent[x + ibegin];
        std::pair<int64_t, int64_t> py = basis[i].LocalParent[y];
        int64_t ij = rels_near[i - 1].lookupIJ(std::get<0>(py), std::get<0>(px) - ibegin_next);
        A_next[yx] = &A[i - 1].A_ptr[(std::get<1>(py) * dimn_up + std::get<1>(px)) * dims + ij * dimn_up * dimn_up];
      }

    std::vector<double*> X_next(N_rows);
    for (int64_t x = 0; x < N_rows; x++) { 
      std::pair<int64_t, int64_t> p = basis[i].LocalParent[x];
      X_next[x] = &A[i - 1].X_ptr[std::get<1>(p) * dims + std::get<0>(p) * dimn_up];
    }

    batchParamsCreate(&A[i].params, dimr, dims, A[i].U_ptr, A[i].A_ptr, A[i].X_ptr, dimn_up, &A_next[0], &X_next[0],
      *Workspace, work_size, N_rows, N_cols, ibegin, &rels_near[i].ColIndex[0], &rels_near[i].RowIndex[0]);
  }

  int64_t child = std::get<0>(basis[0].LocalChild[0]);
  int64_t clen = std::get<1>(basis[0].LocalChild[0]);
  std::vector<int64_t> cdims(clen);
  if (child >= 0)
    for (int64_t i = 0; i < clen; i++)
      cdims[i] = basis[1].DimsLr[child + i];
  else
    cdims.emplace_back(basis[0].Dims[0]);
  int64_t low_s = clen > 0 ? rank : 0;
  lastParamsCreate(&A[0].params, A[0].A_ptr, A[0].X_ptr, 0 < levels ? (rank * branches) : leaf, low_s, cdims.size(), &cdims[0]);
}

void node_free(Node* node) {
  free(node->A_ptr);
  free(node->X_ptr);
  free(node->B_ptr);
  free(node->U_ptr);
  free(node->A);
  batchParamsDestory(&node->params);
}

void matVecA(Node A[], const Base basis[], const CSR rels_near[], double* X, const ColCommMPI comm[], int64_t levels) {
  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &comm[levels]);

  memcpy(A[levels].X[lbegin].A, X, llen * basis[levels].dimN * sizeof(double));

  for (int64_t i = levels; i > 0; i--) {
    int64_t ibegin = 0, iboxes = 0, xlen = 0;
    content_length(&iboxes, &xlen, &ibegin, &comm[i]);

    level_merge_cpu(A[i].X[0].A, xlen * basis[i].dimN, &comm[i]);
    neighbor_bcast_cpu(A[i].X[0].A, basis[i].dimN, &comm[i]);
    dup_bcast_cpu(A[i].X[0].A, xlen * basis[i].dimN, &comm[i]);

    for (int64_t j = 0; j < iboxes; j++)
      mmult('T', 'N', &basis[i].Uo[j + ibegin], &A[i].X[j + ibegin], &A[i].Xo[j + ibegin], 1., 0.);
  }

  level_merge_cpu(A[0].X[0].A, basis[0].dimN, &comm[0]);
  dup_bcast_cpu(A[0].X[0].A, basis[0].dimN, &comm[0]);
  mmult('N', 'N', &A[0].A[0], &A[0].X[0], &A[0].B[0], 1., 0.);
  memset(A[0].X[0].A, 0, sizeof(double) * basis[0].dimN);

  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = 0, iboxes = 0, ulen = 0;
    content_length(&iboxes, &ulen, &ibegin, &comm[i]);
    for (int64_t j = 0; j < iboxes; j++)
      mmult('N', 'N', &basis[i].Uo[j + ibegin], &A[i].Xc[j + ibegin], &A[i].B[j + ibegin], 1., 0.);
    for (int64_t y = 0; y < iboxes; y++)
      for (int64_t xy = rels_near[i].RowIndex[y]; xy < rels_near[i].RowIndex[y + 1]; xy++) {
        int64_t x = rels_near[i].ColIndex[xy];
        mmult('N', 'N', &A[i].A[xy], &A[i].X[x], &A[i].B[y + ibegin], 1., 1.);
      }
    memset(A[i].X_ptr, 0, sizeof(double) * ulen * basis[i].dimN);
  }
  memcpy(X, A[levels].B[lbegin].A, llen * basis[levels].dimN * sizeof(double));
}

void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX) {
  double err[2] = { 0., 0. };
  for (int64_t i = 0; i < lenX; i++) {
    double diff = X[i] - ref[i];
    err[0] = err[0] + diff * diff;
    err[1] = err[1] + ref[i] * ref[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = sqrt(err[0] / err[1]);
}


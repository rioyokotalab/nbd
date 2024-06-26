
#include <gpu_linalg.hpp>
#include <linalg.hpp>
#include <build_tree.hpp>
#include <basis.hpp>
#include <comm.hpp>

#include "cuda.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <tuple>

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


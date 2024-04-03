#pragma once

#include <vector>
#include <cstdint>

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

void* init_libs(int* argc, char*** argv);
void fin_libs();
void set_work_size(int64_t Lwork, double** D_DATA, int64_t* D_DATA_SIZE);

void batchParamsCreate(struct BatchedFactorParams* params, int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, double* X_ptr, int64_t N_up, double** A_up, double** X_up,
  double* Workspace, int64_t Lwork, int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]);
void batchParamsDestory(struct BatchedFactorParams* params);

void lastParamsCreate(struct BatchedFactorParams* params, double* A, double* X, int64_t N, int64_t S, int64_t clen, const int64_t cdims[]);

void allocBufferedList(void** A_ptr, void** A_buffer, int64_t element_size, int64_t count);
void flushBuffer(char dir, void* A_ptr, void* A_buffer, int64_t element_size, int64_t count);
void freeBufferedList(void* A_ptr, void* A_buffer);

void batchCholeskyFactor(struct BatchedFactorParams* params, const struct CellComm* comm);
void batchForwardULV(struct BatchedFactorParams* params, const struct CellComm* comm);
void batchBackwardULV(struct BatchedFactorParams* params, const struct CellComm* comm);
void chol_decomp(struct BatchedFactorParams* params, const struct CellComm* comm);
void chol_solve(struct BatchedFactorParams* params, const struct CellComm* comm);

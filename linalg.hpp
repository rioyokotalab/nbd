#pragma once

#include <complex>
#include <cstdint>
#include <vector>

class Matrix {
public:
  double* A; int64_t M, N, LDA; 
};

class BatchedFactorParams {
public:
  int64_t N_r, N_s, N_upper, L_diag, L_nnz, L_lower, L_rows, L_tmp;
  const double** U_r, **U_s, **V_x, **A_sx, **U_i, *U_d0;
  double** A_x, **A_s, **A_l, **B_x, **A_upper, *V_data, *A_data;
  int64_t* ipiv;
  int* info;
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
  std::vector<Matrix> X, Xc, Xo, B; 
  double* A_ptr;
  double* X_ptr, *B_ptr;
  double* U_ptr;
  int64_t sizeA, sizeU;
  BatchedFactorParams params; 
};

class Base;
class ColCommMPI;
class CSR;
class Cell;
class EvalDouble;

void gen_matrix(const EvalDouble& Eval, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda);

void mmult(char ta, char tb, const Matrix* A, const Matrix* B, Matrix* C, double alpha, double beta);

void mul_AS(const Matrix* RU, const Matrix* RV, Matrix* A);

void compute_basis(const EvalDouble& eval, int64_t rank_max, 
  int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t Nclose, const double Cbodies[], int64_t Nfar, const double Fbodies[]);

void mat_vec_reference(const EvalDouble& eval, int64_t begin, int64_t end, double B[], int64_t nbodies, const double* bodies, const double Xbodies[]);

void set_work_size(int64_t Lwork, double** D_DATA, int64_t* D_DATA_SIZE);

void batchParamsCreate(BatchedFactorParams* params, int64_t R_dim, int64_t S_dim, const double* U_ptr, double* A_ptr, double* X_ptr, int64_t N_up, double** A_up, double** X_up,
  double* Workspace, int64_t Lwork, int64_t N_rows, int64_t N_cols, int64_t col_offset, const int64_t row_A[], const int64_t col_A[]);
void batchParamsDestory(BatchedFactorParams* params);

void lastParamsCreate(BatchedFactorParams* params, double* A, double* X, int64_t N, int64_t S, int64_t clen, const int64_t cdims[]);

void batchCholeskyFactor(BatchedFactorParams* params, const ColCommMPI* comm);
void batchForwardULV(BatchedFactorParams* params, const ColCommMPI* comm);
void batchBackwardULV(BatchedFactorParams* params, const ColCommMPI* comm);
void chol_decomp(BatchedFactorParams* params, const ColCommMPI* comm);
void chol_solve(BatchedFactorParams* params, const ColCommMPI* comm);

void allocNodes(Node A[], double** Workspace, int64_t* Lwork, int64_t rank, int64_t leaf, int64_t branches, const Cell cells[], const Base basis[], const CSR rels_near[], const CSR rels_far[], const ColCommMPI comm[], int64_t levels);
void node_free(Node* node);

void matVecA(Node A[], const Base basis[], const CSR rels_near[], double* X, const ColCommMPI comm[], int64_t levels);

void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX);



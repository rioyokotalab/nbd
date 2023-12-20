
#pragma once

#include "mpi.h"
#include "cuda_runtime_api.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cstddef>

#include "comm.hxx"
#include "basis.hxx"

struct Matrix { double* A; int64_t M, N, LDA; };

struct Cell { int64_t Child[2], Body[2], Level; double R[3], C[3]; };
struct CSR { int64_t M, N, *RowIndex, *ColIndex; };

struct BatchedFactorParams { 
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

struct Node {
  int64_t lenA, lenS;
  struct Matrix *A, *S;
  double* A_ptr, *A_buf, *X_ptr, *X_buf;
  struct BatchedFactorParams params; 
};

struct EvalDouble;

void gen_matrix(const EvalDouble& Eval, int64_t m, int64_t n, const double* bi, const double* bj, double Aij[], int64_t lda);

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta);

void mul_AS(const struct Matrix* RU, const struct Matrix* RV, struct Matrix* A);

int64_t compute_basis(const EvalDouble& eval, double epi, int64_t rank_min, int64_t rank_max, 
  int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t Nclose, const double Cbodies[], int64_t Nfar, const double Fbodies[]);

cudaStream_t init_libs(int* argc, char*** argv);
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

void mat_vec_reference(const EvalDouble& eval, int64_t begin, int64_t end, double B[], int64_t nbodies, const double* bodies, const double Xbodies[]);

void buildTree(int64_t* ncells, struct Cell* cells, double* bodies, int64_t nbodies, int64_t levels);

void buildTreeBuckets(struct Cell* cells, const double* bodies, const int64_t buckets[], int64_t levels);

void traverse(char NoF, CSR* rels, int64_t ncells, const struct Cell* cells, double theta);

void csc_free(CSR* csc);

void lookupIJ(int64_t* ij, const CSR* rels, int64_t i, int64_t j);

void countMaxIJ(int64_t* max_i, int64_t* max_j, const CSR* rels);

void loadX(double* X, int64_t seg, const double Xbodies[], int64_t Xbegin, int64_t ncells, const struct Cell cells[]);

void evalD(const EvalDouble& eval, struct Matrix* D, const CSR* rels, const struct Cell* cells, const double* bodies, const struct CellComm* comm);

void evalS(const EvalDouble& eval, struct Matrix* S, const struct Base* basis, const CSR* rels, const struct CellComm* comm);

void allocNodes(struct Node A[], double** Workspace, int64_t* Lwork, const struct Base basis[], const CSR rels_near[], const CSR rels_far[], const struct CellComm comm[], int64_t levels);

void node_free(struct Node* node);

void factorA_mov_mem(char dir, struct Node A[], const struct Base basis[], int64_t levels);

void matVecA(const struct Node A[], const struct Base basis[], const CSR rels_near[], double* X, const struct CellComm comm[], int64_t levels);

void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX);



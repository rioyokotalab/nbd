
#include <nbd.hpp>
#include <kernel.hpp>

#include "mkl.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <tuple>

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

int64_t compute_basis(const EvalDouble& eval, double epi, int64_t rank_min, int64_t rank_max, 
  int64_t M, double* A, int64_t LDA, double Xbodies[], int64_t Nclose, const double Cbodies[], int64_t Nfar, const double Fbodies[]) {

  if (M > 0 && (Nclose > 0 || Nfar > 0)) {
    int64_t ldm = std::max(M, Nclose + Nfar);
    std::vector<double> Aall(M * ldm, 0.), U(M * M), S(M * 2);
    std::vector<long long> ipiv(M);
    gen_matrix(eval, Nclose, M, Cbodies, Xbodies, &Aall[0], ldm);
    gen_matrix(eval, Nfar, M, Fbodies, Xbodies, &Aall[Nclose], ldm);

    for (int64_t i = 0; i < Nclose; i += M) {
      int64_t len = std::min(M, Nclose - i);
      gen_matrix(eval, len, len, &Cbodies[i * 3], &Cbodies[i * 3], &U[0], M);
      //LAPACKE_dgesv(LAPACK_COL_MAJOR, len, M, &U[0], M, &ipiv[0], &Aall[i], ldm);
    }

    std::fill(ipiv.begin(), ipiv.end(), 0);
    LAPACKE_dgeqp3(LAPACK_COL_MAJOR, Nclose + Nfar, M, &Aall[0], ldm, &ipiv[0], &S[0]);
    LAPACKE_dlaset(LAPACK_COL_MAJOR, 'L', M - 1, M - 1, 0., 0., &Aall[1], ldm);

    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'A', M, M, &Aall[0], ldm, &S[0], NULL, M, &U[0], M, &S[M]);
    double s0 = S[0] * epi;
    rank_max = rank_max <= 0 ? M : std::min(rank_max, M);
    rank_min = rank_min <= 0 ? 0 : std::min(rank_min, M);
    int64_t rank = epi > 0. ?
      std::distance(S.begin(), std::find_if(S.begin() + rank_min, S.begin() + rank_max, [s0](double& s) { return s < s0; })) : rank_max;
    
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
    
    return rank;
  }
  return 0;
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




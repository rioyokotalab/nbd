

#include "linalg.hxx"

#if USE_MKL
#include "mkl.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cstdlib>

using namespace nbd;

void nbd::cMatrix(Matrix& mat, int64_t m, int64_t n) {
  mat.A.resize(m * n);
  mat.M = m;
  mat.N = n;
}

void nbd::cVector(Vector& vec, int64_t n) {
  vec.X.resize(n);
  vec.N = n;
}

void nbd::cpyFromMatrix(char trans, const Matrix& A, double* V) {
  int64_t iv = A.M;
  int64_t incv = 1;
  if (trans == 'T' || trans == 't') {
    iv = 1;
    incv = A.N;
  }
  for (int64_t j = 0; j < A.N; j++)
    cblas_dcopy(A.M, &A.A[j * A.M], 1, &V[j * iv], incv);
}

void nbd::cpyFromVector(const Vector& A, double* v) {
  std::copy(A.X.data(), A.X.data() + A.N, v);
}

void nbd::maxpby(Matrix& A, const double* v, double alpha, double beta) {
  int64_t size = A.M * A.N;
  if (beta == 0.)
    std::fill(A.A.data(), A.A.data() + size, 0.);
  else if (beta != 1.)
    cblas_dscal(size, beta, A.A.data(), 1);
  cblas_daxpy(size, alpha, v, 1, A.A.data(), 1);
}

void nbd::vaxpby(Vector& A, const double* v, double alpha, double beta) {
  int64_t size = A.N;
  if (beta == 0.)
    std::fill(A.X.data(), A.X.data() + size, 0.);
  else if (beta != 1.)
    cblas_dscal(size, beta, A.X.data(), 1);
  cblas_daxpy(size, alpha, v, 1, A.X.data(), 1);
}

void nbd::cpyMatToMat(int64_t m, int64_t n, const Matrix& m1, Matrix& m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
  if (m > 0 && n > 0)
    for (int64_t j = 0; j < n; j++) {
      int64_t j1 = y1 + (x1 + j) * m1.M;
      int64_t j2 = y2 + (x2 + j) * m2.M;
      std::copy(&m1.A[j1], &m1.A[j1] + m, &m2.A[j2]);
    }
}

void nbd::cpyVecToVec(int64_t n, const Vector& v1, Vector& v2, int64_t x1, int64_t x2) {
  if (n > 0)
    std::copy(&v1.X[x1], &v1.X[x1] + n, &v2.X[x2]);
}

void nbd::updateU(double epi, Matrix& A, Matrix& U, int64_t *rnk_out) {
  int64_t m = A.M;
  Matrix au, work;
  int64_t n = A.N + U.N;
  int64_t rank = std::min(m, n);
  rank = *rnk_out > 0 ? std::min(rank, *rnk_out) : rank;
  cMatrix(au, m, n);
  cMatrix(work, m, U.N);
  cpyMatToMat(m, A.N, A, au, 0, 0, 0, 0);
  cpyMatToMat(m, U.N, U, au, 0, 0, 0, A.N);
  cpyMatToMat(m, U.N, U, work, 0, 0, 0, 0);
  cMatrix(A, m, m);

  Vector s, superb;
  cVector(s, std::min(m, n));
  cVector(superb, s.N + 1);
  int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'N', m, n, au.A.data(), m, s.X.data(), A.A.data(), m, NULL, n, superb.X.data());

  if (epi > 0.) {
    rank = 0;
    double sepi = s.X[0] * epi;
    while(rank < s.N && s.X[rank] > sepi)
      rank += 1;
  }
  *rnk_out = rank;

  if (U.N > 0 && rank > 0) {
    cMatrix(U, rank, U.N);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, U.N, m, 1., A.A.data(), m, work.A.data(), m, 0., U.A.data(), rank);
  }
}

void nbd::updateSubU(Matrix& U, const Matrix& R1, const Matrix& R2) {
  if (U.M > 0 && U.N > 0) {
    int64_t m1 = R1.N;
    int64_t m2 = R2.N;
    int64_t n = U.N;
    Matrix ru1, ru2;
    cMatrix(ru1, R1.M, n);
    cMatrix(ru2, R2.M, n);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R1.M, n, m1, 1., R1.A.data(), R1.M, U.A.data(), U.M, 0., ru1.A.data(), R1.M);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, R2.M, n, m2, 1., R2.A.data(), R2.M, U.A.data() + m1, U.M, 0., ru2.A.data(), R2.M);

    cMatrix(U, R1.M + R2.M, n);
    cpyMatToMat(R1.M, n, ru1, U, 0, 0, 0, 0);
    cpyMatToMat(R2.M, n, ru2, U, 0, 0, R1.M, 0);
  }
}

void nbd::lraID(double epi, int64_t lenR, Matrix& A, Matrix& U, int64_t arows[], const double R[], int64_t* rnk_out) {
  int64_t rank = std::min(A.M, A.N);
  rank = *rnk_out > 0 ? std::min(*rnk_out, rank) : rank;
  zeroMatrix(U);
  msample('N', lenR, A, R, U);

  Vector s, superb;
  cVector(s, rank);
  cVector(superb, s.N + 1);
  int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'N', A.M, rank, U.A.data(), A.M, s.X.data(), NULL, A.M, NULL, rank, superb.X.data());
  if (epi > 0.) {
    rank = 0;
    double sepi = s.X[0] * epi;
    while(rank < s.N && s.X[rank] > sepi)
      rank += 1;
  }
  *rnk_out = rank;

  for (int64_t i = 0; i < rank; i++)
    cblas_dscal(A.M, s.X[i], &U.A[i * A.M], 1);
  cblas_dcopy(A.M * rank, U.A.data(), 1, A.A.data(), 1);

  std::vector<int> ipiv(rank);
  info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.M, rank, A.A.data(), A.M, ipiv.data());
  if (info > 0)
    rank = info - 1;
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A.M, rank, 1., A.A.data(), A.M, U.A.data(), A.M);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, A.M, rank, 1., A.A.data(), A.M, U.A.data(), A.M);

  std::vector<int64_t> rows(A.M);
  std::iota(rows.begin(), rows.end(), 0);
  for (int64_t i = 0; i < rank; i++) {
    int64_t ri = ipiv[i] - 1;
    std::iter_swap(&rows[i], &rows[ri]);
    arows[i] = rows[i];
  }
}

void nbd::zeroMatrix(Matrix& A) {
  std::fill(A.A.data(), A.A.data() + A.M * A.N, 0.);
}

void nbd::zeroVector(Vector& A) {
  std::fill(A.X.data(), A.X.data() + A.N, 0.);
}

void nbd::mmult(char ta, char tb, const Matrix& A, const Matrix& B, Matrix& C, double alpha, double beta) {
  int64_t k = (ta == 'N' || ta == 'n') ? A.N : A.M;
  if (C.M > 0 && C.N > 0 && k > 0) {
    CBLAS_TRANSPOSE tac = (ta == 'T' || ta == 't') ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tbc = (tb == 'T' || tb == 't') ? CblasTrans : CblasNoTrans;
    cblas_dgemm(CblasColMajor, tac, tbc, C.M, C.N, k, alpha, A.A.data(), A.M, B.A.data(), B.M, beta, C.A.data(), C.M);
  }
}

void nbd::msample(char ta, int64_t lenR, const Matrix& A, const double R[], Matrix& C) {
  if (lenR < C.N * 100) { 
    std::cerr << "Insufficent random vector: " << C.N << " x 100 needed " << lenR << " provided." << std::endl;
    return;
  }
  int64_t k = A.M;
  int64_t inca = 1;
  CBLAS_TRANSPOSE tac = CblasTrans;
  if (ta == 'N' || ta == 'n') {
    k = A.N;
    inca = A.M;
    tac = CblasNoTrans;
  }

  int64_t rk = lenR / C.N;
  int64_t lk = k % rk;
  if (lk > 0)
    cblas_dgemm(CblasColMajor, tac, CblasNoTrans, C.M, C.N, lk, 1., A.A.data(), A.M, R, lk, 1., C.A.data(), C.M);
  if (k > rk)
    for (int64_t i = lk; i < k; i += rk)
      cblas_dgemm(CblasColMajor, tac, CblasNoTrans, C.M, C.N, rk, 1., &A.A[i * inca], A.M, R, rk, 1., C.A.data(), C.M);
}

void nbd::msample_m(char ta, const Matrix& A, const Matrix& B, Matrix& C) {
  int64_t k = A.M;
  CBLAS_TRANSPOSE tac = CblasTrans;
  if (ta == 'N' || ta == 'n') {
    k = A.N;
    tac = CblasNoTrans;
  }
  int64_t nrhs = std::min(B.N, C.N);
  cblas_dgemm(CblasColMajor, tac, CblasNoTrans, C.M, nrhs, k, 1., A.A.data(), A.M, B.A.data(), B.M, 1., C.A.data(), C.M);
}

void nbd::minvl(const Matrix& A, Matrix& B) {
  if (A.M > 0 && A.N > 0) {
    Matrix work;
    cMatrix(work, A.M, A.N);
    cpyMatToMat(A.M, A.N, A, work, 0, 0, 0, 0);
    std::vector<int> ipiv(A.M);
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.M, A.N, work.A.data(), A.M, ipiv.data());
    LAPACKE_dgetrs(LAPACK_COL_MAJOR, 'N', A.M, B.N, work.A.data(), A.M, ipiv.data(), B.A.data(), B.M);
  }
}

void nbd::chol_decomp(Matrix& A) {
  if (A.M > 0)
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A.M, A.A.data(), A.M);
}

void nbd::trsm_lowerA(Matrix& A, const Matrix& L) {
  if (A.M > 0 && L.M > 0 && L.N > 0)
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, A.M, A.N, 1., L.A.data(), L.M, A.A.data(), A.M);
}

void nbd::utav(char tb, const Matrix& U, const Matrix& A, const Matrix& VT, Matrix& C) {
  Matrix work;
  cMatrix(work, C.M, A.N);
  if (tb == 'N' || tb == 'n') {
    mmult('T', 'N', U, A, work, 1., 0.);
    mmult('N', 'N', work, VT, C, 1., 0.);
  }
  else if (tb == 'T' || tb == 't') {
    mmult('N', 'N', U, A, work, 1., 0.);
    mmult('N', 'T', work, VT, C, 1., 0.);
  }
}

void nbd::chol_solve(Vector& X, const Matrix& A) {
  if (A.M > 0 && X.N > 0) {
    fw_solve(X, A);
    bk_solve(X, A);
  }
}

void nbd::fw_solve(Vector& X, const Matrix& L) {
  if (L.M > 0 && X.N > 0)
    cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, X.N, L.A.data(), L.M, X.X.data(), 1);
}

void nbd::bk_solve(Vector& X, const Matrix& L) {
  if (L.M > 0 && X.N > 0)
    cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, X.N, L.A.data(), L.M, X.X.data(), 1);
}

void nbd::mvec(char ta, const Matrix& A, const Vector& X, Vector& B, double alpha, double beta) {
  if (A.M > 0 && A.N > 0) {
    CBLAS_TRANSPOSE tac = (ta == 'T' || ta == 't') ? CblasTrans : CblasNoTrans;
    cblas_dgemv(CblasColMajor, tac, A.M, A.N, alpha, A.A.data(), A.M, X.X.data(), 1, beta, B.X.data(), 1);
  }
}

void nbd::vnrm2(const Vector& A, double* nrm) {
  *nrm = cblas_dnrm2(A.N, A.X.data(), 1);
}

void nbd::verr2(const Vector& A, const Vector& B, double* err) {
  Vector work;
  cVector(work, A.N);
  vaxpby(work, A.X.data(), 1., 0.);
  vaxpby(work, B.X.data(), -1., 1.);
  vnrm2(work, err);
}

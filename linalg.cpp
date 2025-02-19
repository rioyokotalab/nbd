
#include <linalg.hpp>
#include <build_tree.hpp>

#include "cblas.h"
#include "lapacke.h"

#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <tuple>
#include <random>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/QR>

DenseZMat::DenseZMat(int64_t M, int64_t N) : MatrixAcc(M, N), A(nullptr) {
  if (0 < M && 0 < N) {
    A = (std::complex<double>*)malloc(M * N * sizeof(std::complex<double>));
    std::fill(A, &A[M * N], 0.);
  }
}

DenseZMat::~DenseZMat() {
  if (A)
    free(A);
}

void DenseZMat::op_Aij_mulB(char opA, int64_t mA, int64_t nA, int64_t nrhs, int64_t iA, int64_t jA, const std::complex<double>* B_in, int64_t strideB, std::complex<double>* C_out, int64_t strideC) const {
  Eigen::Stride<Eigen::Dynamic, 1> lda(M, 1), ldb(strideB, 1), ldc(strideC, 1);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matC(C_out, mA, nrhs, ldc);
  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matB(B_in, nA, nrhs, ldb);
  if (opA == 'T' || opA == 't')
    matC.noalias() = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(&A[iA + jA * M], nA, mA, lda).transpose() * matB;
  else if (opA == 'C' || opA == 'c')
    matC.noalias() = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(&A[iA + jA * M], nA, mA, lda).adjoint() * matB;
  else
    matC.noalias() = Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>>(&A[iA + jA * M], mA, nA, lda) * matB;
}

LowRankMatrix::LowRankMatrix(int64_t m, int64_t n, const MatrixAcc& eval, int64_t iA, int64_t jA) : M(m), N(n), rank(n), U(m * rank) {
  Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(n, n);
  eval.op_Aij_mulB('N', m, n, n, iA, jA, id.data(), n, U.data(), m);
}

LowRankMatrix::LowRankMatrix(double epi, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters, const MatrixAcc& eval, int64_t iA, int64_t jA) : M(m), N(n), rank(k), U(m * rank), V(N * rank), S(rank) {
  Zrsvd(epi, m, n, &k, p, niters, eval, iA, jA, S.data(), U.data(), m, V.data(), n);

  if (k < rank) {
    rank = k;
    U.resize(m * rank);
    V.resize(n * rank);
    S.resize(rank);
  }
}

void LowRankMatrix::matVecMul(int64_t mA, int64_t nA, int64_t nrhs, int64_t iA, int64_t jA, const std::complex<double>* B_in, int64_t strideB, std::complex<double>* C_out, int64_t strideC) const {
  Eigen::Stride<Eigen::Dynamic, 1> ldb(strideB, 1), ldc(strideC, 1);
  Eigen::Map<const Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> B(B_in, mA, nrhs, ldb);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> C(C_out, nA, nrhs, ldc);

  Eigen::Map<const Eigen::MatrixXcd> matU(U.data() + iA, mA, rank);
  if (rank < N) {
    Eigen::Map<const Eigen::MatrixXcd> matV(V.data() + jA, nA, rank);
    Eigen::Map<const Eigen::VectorXd> vecS(S.data(), rank);
    C.noalias() += matU * (vecS.asDiagonal() * (matV.adjoint() * B));
  }
  else
    C.noalias() += matU * B;
}

Hmatrix::Hmatrix(const MatrixAcc& eval, int64_t lbegin, int64_t len, const Cell cells[], const CSR& Near) {
  int64_t lend = lbegin + len;
  int64_t llen = Near.RowIndex[lend] - Near.RowIndex[lbegin];
  yOffsets.reserve(llen);
  xOffsets.reserve(llen);
  L.reserve(llen);
  
  for (int64_t y = lbegin; y < lend; y++) {
    int64_t yi = cells[y].Body[0];
    int64_t M = cells[y].Body[1] - yi;

    for (int64_t yx = Near.RowIndex[y]; yx < Near.RowIndex[y + 1]; yx++) {
      int64_t x = Near.ColIndex[yx];
      int64_t xj = cells[x].Body[0];
      yOffsets.emplace_back(yi);
      xOffsets.emplace_back(xj);
      L.emplace_back(M, cells[x].Body[1] - xj, eval, yi, xj);
    }
  }
}

Hmatrix::Hmatrix(double epi, const MatrixAcc& eval, int64_t rank, int64_t p, int64_t niters, int64_t lbegin, int64_t len, const Cell cells[], const CSR& Far) {
  int64_t lend = lbegin + len;
  int64_t llen = Far.RowIndex[lend] - Far.RowIndex[lbegin];
  yOffsets.reserve(llen);
  xOffsets.reserve(llen);
  L.reserve(llen);
  
  for (int64_t y = lbegin; y < lend; y++) {
    int64_t yi = cells[y].Body[0];
    int64_t M = cells[y].Body[1] - yi;

    for (int64_t yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      int64_t x = Far.ColIndex[yx];
      int64_t xj = cells[x].Body[0];
      yOffsets.emplace_back(yi);
      xOffsets.emplace_back(xj);
      L.emplace_back(epi, M, cells[x].Body[1] - xj, rank, p, niters, eval, yi, xj);
    }
  }
}

void Hmatrix::matVecMul(int64_t mA, int64_t nA, int64_t nrhs, int64_t iA, int64_t jA, const std::complex<double>* B_in, int64_t strideB, std::complex<double>* C_out, int64_t strideC) const {
  for (int64_t i = 0; i < (int64_t)L.size(); i++) {
    int64_t ybegin = std::max(iA, yOffsets[i]);
    int64_t yend = std::min(iA + mA, yOffsets[i] + L[i].M);
    int64_t xbegin = std::max(jA, xOffsets[i]);
    int64_t xend = std::min(jA + nA, xOffsets[i] + L[i].N);

    L[i].matVecMul(yend - ybegin, xend - xbegin, nrhs, ybegin - yOffsets[i], xbegin - xOffsets[i], &B_in[xbegin - jA], strideB, &C_out[ybegin - iA], strideC);
  }
}

void Zrsvd(double epi, int64_t m, int64_t n, int64_t* k, int64_t p, int64_t niters, const MatrixAcc& A, int64_t iA, int64_t jA, double* S, std::complex<double>* U, int64_t ldu, std::complex<double>* V, int64_t ldv) {
  int64_t rank = std::min(*k, std::min(m, n));
  p = std::min(rank + p, std::min(m, n));
  Eigen::MatrixXcd R(n, p), Q(m, p);
  std::mt19937_64 gen;
  std::normal_distribution<double> norm_dist(0., 1.);
  std::generate(R.reshaped().begin(), R.reshaped().end(), [&]() { return std::complex<double>(norm_dist(gen), norm_dist(gen)); });

  A.op_Aij_mulB('N', m, n, p, iA, jA, R.data(), n, Q.data(), m);
  while (0 < --niters) {
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Q);
    Q = qr.householderQ() * Eigen::MatrixXcd::Identity(m, p);
    A.op_Aij_mulB('C', n, m, p, iA, jA, Q.data(), m, R.data(), n);
    A.op_Aij_mulB('N', m, n, p, iA, jA, R.data(), n, Q.data(), m);
  }

  Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Q);
  Q = qr.householderQ() * Eigen::MatrixXcd::Identity(m, p);
  A.op_Aij_mulB('C', n, m, p, iA, jA, Q.data(), m, R.data(), n);

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(R, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (0. < epi && epi < 1.)
  { svd.setThreshold(epi); *k = rank = std::min(rank, (int64_t)svd.rank()); }

  Eigen::Stride<Eigen::Dynamic, 1> ldU(ldu, 1), ldV(ldv, 1);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matU(U, m, rank, ldU);
  Eigen::Map<Eigen::MatrixXcd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, 1>> matV(V, n, rank, ldV);
  Eigen::Map<Eigen::VectorXd> vecS(S, rank);

  vecS = svd.singularValues().topRows(rank);
  matV = svd.matrixU().leftCols(rank);
  matU.noalias() = Q * svd.matrixV().leftCols(rank);
}

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
    std::vector<int32_t> ipiv(M);
    gen_matrix(eval, Nclose, M, Cbodies, Xbodies, &Aall[0], ldm);
    gen_matrix(eval, Nfar, M, Fbodies, Xbodies, &Aall[Nclose], ldm);

    std::fill(ipiv.begin(), ipiv.end(), 0);
    LAPACKE_dgeqp3(LAPACK_COL_MAJOR, Nclose + Nfar, M, &Aall[0], ldm, &ipiv[0], &S[0]);
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

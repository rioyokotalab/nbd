
#include <h2matrix.hpp>
#include <basis_builder.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <csr.hpp>
#include <dense.hpp>
#include <lapack_f2c.hpp>

#include <numeric>
#include <algorithm>

inline void agemv(char transA, int M, int N, const std::complex<double>* A, int lda, const std::complex<double>* X, int beta, std::complex<double>* B)
{ int incx = 1; std::complex<double> one(1., 0.), zero(0., 0.); zgemv_(&transA, &M, &N, &one, A, &lda, X, &incx, beta ? &one : &zero, B, &incx); }
inline void nngemm(int M, int N, int K, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, std::complex<double>* C, int ldc)
{ char transA = 'N'; std::complex<double> one(1., 0.), zero(0., 0.); zgemm_(&transA, &transA, &M, &N, &K, &one, A, &lda, B, &ldb, &zero, C, &ldc); }

void H2Matrix::eval(const Cell cells[], const CSR& Far, const BasisBuilder& row_side, const BasisBuilder& col_side, const CSR& Far_local, const H2MatrixComm& comm) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();

  std::vector<long long> Dims(xlen), DimsLr(xlen);
  DimsOffsets.resize(xlen + 1);
  DimsLrOffsets.resize(xlen + 1);
  DimsOffsets[0] = DimsLrOffsets[0] = 0;

  UOffsets.resize(nodes + 1); VOffsets.resize(nodes + 1);
  COffsets.resize(Far_local.RowIndex[nodes] + 1);
  UOffsets[0] = COffsets[0] = 0;

  std::copy(row_side.M.begin(), row_side.M.end(), &Dims[ibegin]);
  std::copy(row_side.Ranks.begin(), row_side.Ranks.end(), &DimsLr[ibegin]);
  std::transform(Dims.begin() + ibegin, Dims.begin() + ibegin + nodes, DimsLr.begin() + ibegin, UOffsets.begin() + 1, std::multiplies<long long>());

  std::vector<long long> seq(xlen + 1);
  std::iota(seq.begin(), seq.end(), 0ll);
  comm.neighbor_bcast(Dims.data(), seq.data());
  comm.neighbor_bcast(DimsLr.data(), seq.data());

  std::partial_sum(Dims.begin(), Dims.end(), DimsOffsets.begin() + 1);
  std::partial_sum(DimsLr.begin(), DimsLr.end(), DimsLrOffsets.begin() + 1);
  std::partial_sum(UOffsets.begin() + 1, UOffsets.end(), UOffsets.begin() + 1);
  std::copy(UOffsets.begin(), UOffsets.end(), VOffsets.begin());
  U.resize(UOffsets.back());
  V.resize(VOffsets.back());

  for (long long i = 0; i < nodes; i++)
    std::transform(Far_local.ColIndex.begin() + Far_local.RowIndex[i], Far_local.ColIndex.begin() + Far_local.RowIndex[i + 1], COffsets.begin() + Far_local.RowIndex[i] + 1,
      [&](long long col) { return DimsLr[i + ibegin] * DimsLr[col]; });

  std::partial_sum(COffsets.begin() + 1, COffsets.end(), COffsets.begin() + 1);
  C.resize(COffsets.back());

  std::vector<long long> basisOffsets(xlen + 1);
  basisOffsets[0] = 0;

  std::transform(DimsLr.begin() + ibegin, DimsLr.begin() + ibegin + nodes, &cells[ybegin], basisOffsets.begin() + ibegin + 1, [](long long d, const Cell& c) { return d * (c.Body[1] - c.Body[0]); });
  comm.neighbor_bcast(&basisOffsets[1], seq.data());
  std::partial_sum(basisOffsets.begin() + 1, basisOffsets.end(), basisOffsets.begin() + 1);

  std::vector<std::complex<double>> fs_basis(basisOffsets.back());
  for (long long i = 0; i < nodes; i++)
    std::copy(col_side.basis_full[i].begin(), col_side.basis_full[i].end(), fs_basis.begin() + basisOffsets[i + ibegin]);
  comm.neighbor_bcast(fs_basis.data(), basisOffsets.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long K = DimsLr[i + ibegin];
    std::copy(row_side.basis[i].begin(), row_side.basis[i].begin() + (M * K), &U[UOffsets[i]]);
    std::copy(col_side.basis[i].begin(), col_side.basis[i].begin() + (M * K), &V[VOffsets[i]]);

    long long col = 0;
    for (long long ij = Far_local.RowIndex[i]; ij < Far_local.RowIndex[i + 1] && 0 < K; ij++) {
      long long j = Far_local.ColIndex[ij];
      long long N = DimsLr[j];
      long long cj = Far.ColIndex[ij + Far.RowIndex[ybegin]];
      long long Nf = cells[cj].Body[1] - cells[cj].Body[0];

      nngemm(K, N, Nf, &(row_side.fmatrix[i])[col * K], K, &fs_basis[basisOffsets[j]], Nf, &C[COffsets[ij]], K);
      col += Nf;
    }
  }
}

void H2Matrix::evalLeafA(const DenseMatrix& matA, const Cell cells[], const H2MatrixComm& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();

  AOffsets.resize(nodes + 1);
  std::transform(DimsOffsets.begin() + ibegin, DimsOffsets.begin() + (ibegin + nodes), DimsOffsets.begin() + (ibegin + 1), AOffsets.begin() + 1, [](long long start, long long end) { return (end - start) * (end - start); });
  std::partial_sum(AOffsets.begin() + 1, AOffsets.end(), AOffsets.begin() + 1);
  AOffsets[0] = 0;
  A.resize(AOffsets.back());

  for (long long i = 0; i < nodes; ++i) {
    long long N = DimsOffsets[i + ibegin + 1] - DimsOffsets[i + ibegin];
    long long ci = i + ybegin;
    matA.AijBlock(N, N, cells[ci].Body[0], cells[ci].Body[0], &A[AOffsets[i]], N);
  }
}

void H2Matrix::matVecUpwardPass(const std::complex<double>* X_in, std::complex<double>* Z, const H2MatrixComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++) {
    long long M = DimsOffsets[i + ibegin + 1] - DimsOffsets[i + ibegin];
    long long N = DimsLrOffsets[i + ibegin + 1] - DimsLrOffsets[i + ibegin];
    if (0 < N) {
      const std::complex<double>* v = &V[UOffsets[i]];
      const std::complex<double>* x = X_in + DimsOffsets[i + ibegin] - DimsOffsets[ibegin];
      std::complex<double>* z = Z + DimsLrOffsets[i + ibegin];
      agemv('C', M, N, v, M, x, 0, z);
    }
  }
  comm.neighbor_bcast(Z, DimsLrOffsets.data());
}

void H2Matrix::matVecHorizontalandDownwardPass(std::complex<double>* Y_out, const std::complex<double>* Z, std::complex<double>* Y, const CSR& Far) const {
  long long ibegin = Far.DiagCol;
  long long nodes = Far.RowIndex.size() - 1;
  long long lenX = DimsOffsets[ibegin + nodes] - DimsOffsets[ibegin];
  std::fill(Y_out, Y_out + lenX, std::complex<double>(0., 0.));

  for (long long i = 0; i < nodes; i++) {
    long long M = DimsOffsets[i + ibegin + 1] - DimsOffsets[i + ibegin];
    long long K = DimsLrOffsets[i + ibegin + 1] - DimsLrOffsets[i + ibegin];
    if (0 < K) {
      std::complex<double>* y = Y + DimsLrOffsets[i + ibegin];
      for (long long ij = Far.RowIndex[i]; ij < Far.RowIndex[i + 1]; ij++) {
        long long j = Far.ColIndex[ij];
        long long N = DimsLrOffsets[j + 1] - DimsLrOffsets[j];

        const std::complex<double>* z = Z + DimsLrOffsets[j];
        const std::complex<double>* c = &C[COffsets[ij]];
        agemv('N', K, N, c, K, z, 1, y);
      }

      const std::complex<double>* u = &U[UOffsets[i]];
      std::complex<double>* x = Y_out + DimsOffsets[i + ibegin] - DimsOffsets[ibegin];
      agemv('N', M, K, u, M, y, 0, x);
    }
  }
}

void H2Matrix::matVecLeafHorizontalPass(std::complex<double>* X_io, const std::complex<double>* Z, std::complex<double>* Y, const CSR& Far, const H2MatrixComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long lenX = DimsOffsets[ibegin + nodes] - DimsOffsets[ibegin];

  std::vector<std::complex<double>> X(DimsOffsets.back(), std::complex<double>(0., 0.));
  std::copy(&X_io[0], &X_io[lenX], X.begin() + DimsOffsets[ibegin]);
  comm.neighbor_bcast(X.data(), DimsOffsets.data());
  std::fill(X_io, X_io + lenX, std::complex<double>(0., 0.));

  for (long long i = 0; i < nodes; i++) {
    long long M = DimsOffsets[i + ibegin + 1] - DimsOffsets[i + ibegin];
    long long K = DimsLrOffsets[i + ibegin + 1] - DimsLrOffsets[i + ibegin];
    std::complex<double>* w = X_io + DimsOffsets[i + ibegin] - DimsOffsets[ibegin];
    
    if (0 < K) {
      std::complex<double>* y = Y + DimsLrOffsets[i + ibegin];
      for (long long ij = Far.RowIndex[i]; ij < Far.RowIndex[i + 1]; ij++) {
        long long j = Far.ColIndex[ij];
        long long N = DimsLrOffsets[j + 1] - DimsLrOffsets[j];

        const std::complex<double>* z = Z + DimsLrOffsets[j];
        const std::complex<double>* c = &C[COffsets[ij]];
        agemv('N', K, N, c, K, z, 1, y);
      }

      const std::complex<double>* u = &U[UOffsets[i]];
      agemv('N', M, K, u, M, y, 1, w);
    }

    const std::complex<double>* x = X.data() + DimsOffsets[i + ibegin];
    const std::complex<double>* a = &A[AOffsets[i]];
    agemv('N', M, M, a, M, x, 1, w);
  }
}


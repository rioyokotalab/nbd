
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

void H2Matrix::eval(Cell cells[], const BasisBuilder& row_side, const BasisBuilder& col_side, const CSR& Far_local, const CSR& Far, const H2MatrixComm& comm) {
  long long xlen = comm.lenNeighbors();
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();

  std::vector<long long> Dims(xlen), DimsLr(xlen);
  URowPrefixes.resize(xlen + 1); UColPrefixes.resize(xlen + 1);
  VRowPrefixes.resize(xlen + 1); VColPrefixes.resize(xlen + 1);
  URowPrefixes[0] = UColPrefixes[0] = VRowPrefixes[0] = VColPrefixes[0] = 0;

  UPrefixes.resize(nodes + 1); VPrefixes.resize(nodes + 1);
  CPrefixes.resize(Far_local.RowIndex[nodes] + 1);
  UPrefixes[0] = CPrefixes[0] = 0;

  std::copy(row_side.M.begin(), row_side.M.end(), &Dims[ibegin]);
  std::copy(row_side.Ranks.begin(), row_side.Ranks.end(), &DimsLr[ibegin]);
  std::transform(Dims.begin() + ibegin, Dims.begin() + ibegin + nodes, DimsLr.begin() + ibegin, UPrefixes.begin() + 1, std::multiplies<long long>());

  std::vector<long long> seq(xlen + 1);
  std::iota(seq.begin(), seq.end(), 0ll);
  comm.neighbor_bcast(Dims.data(), seq.data());
  comm.neighbor_bcast(DimsLr.data(), seq.data());

  std::partial_sum(Dims.begin(), Dims.end(), URowPrefixes.begin() + 1);
  std::partial_sum(DimsLr.begin(), DimsLr.end(), UColPrefixes.begin() + 1);
  std::partial_sum(UPrefixes.begin() + 1, UPrefixes.end(), UPrefixes.begin() + 1);
  std::copy(UPrefixes.begin(), UPrefixes.end(), VPrefixes.begin());
  std::copy(URowPrefixes.begin(), URowPrefixes.end(), VRowPrefixes.begin());
  std::copy(UColPrefixes.begin(), UColPrefixes.end(), VColPrefixes.begin());
  U.resize(UPrefixes.back());
  V.resize(VPrefixes.back());

  for (long long i = 0; i < nodes; i++)
    std::transform(Far_local.ColIndex.begin() + Far_local.RowIndex[i], Far_local.ColIndex.begin() + Far_local.RowIndex[i + 1], CPrefixes.begin() + Far_local.RowIndex[i] + 1,
      [&](long long col) { return (URowPrefixes[i + ibegin + 1] - URowPrefixes[i + ibegin]) * (VRowPrefixes[col + 1] - VRowPrefixes[col]); });

  std::partial_sum(CPrefixes.begin() + 1, CPrefixes.end(), CPrefixes.begin() + 1);
  C.resize(CPrefixes.back());

  std::vector<long long> basisPrefixes(xlen + 1);
  basisPrefixes[0] = 0;

  std::transform(DimsLr.begin() + ibegin, DimsLr.begin() + ibegin + nodes, &cells[ybegin], basisPrefixes.begin() + ibegin + 1, [](long long d, const Cell& c) { return d * (c.M[1] - c.M[0]); });
  comm.neighbor_bcast(&basisPrefixes[1], seq.data());
  std::partial_sum(basisPrefixes.begin() + 1, basisPrefixes.end(), basisPrefixes.begin() + 1);

  std::vector<std::complex<double>> fs_basis(basisPrefixes.back());
  for (long long i = 0; i < nodes; i++)
    std::copy(col_side.basis_full[i].begin(), col_side.basis_full[i].end(), fs_basis.begin() + basisPrefixes[i + ibegin]);
  comm.neighbor_bcast(fs_basis.data(), basisPrefixes.data());

  for (long long i = 0; i < nodes; i++) {
    long long M = URowPrefixes[i + ibegin + 1] - URowPrefixes[i + ibegin];
    long long K = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
    std::copy(row_side.basis[i].begin(), row_side.basis[i].begin() + (M * K), &U[UPrefixes[i]]);
    std::copy(col_side.basis[i].begin(), col_side.basis[i].begin() + (M * K), &V[VPrefixes[i]]);

    long long col = 0;
    for (long long ij = Far_local.RowIndex[i]; ij < Far_local.RowIndex[i + 1] && 0 < K; ij++) {
      long long j = Far_local.ColIndex[ij];
      long long N = VColPrefixes[j + 1] - VColPrefixes[j];
      long long cj = Far.ColIndex[ij + Far.RowIndex[ybegin]];
      long long Nf = cells[cj].M[1] - cells[cj].M[0];

      nngemm(K, N, Nf, &(row_side.fmatrix[i])[col * K], K, &fs_basis[basisPrefixes[j]], Nf, &C[CPrefixes[ij]], K);
      col += Nf;
    }
  }
}

void H2Matrix::evalLeafA(DenseMatrix& matA, Cell cells[], const CSR& Far_local, const CSR& Far, const H2MatrixComm& comm) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long ybegin = comm.oGlobal();

  APrefixes.resize(nodes + 1);
  std::transform(URowPrefixes.begin() + ibegin, URowPrefixes.begin() + (ibegin + nodes), URowPrefixes.begin() + (ibegin + 1), APrefixes.begin() + 1, [](long long start, long long end) { return (end - start) * (end - start); });
  std::partial_sum(APrefixes.begin() + 1, APrefixes.end(), APrefixes.begin() + 1);
  APrefixes[0] = 0;
  A.resize(APrefixes.back());

  for (long long i = 0; i < nodes; ++i) {
    long long ci = i + ybegin;
    long long y = cells[ci].M[0], M = cells[ci].M[1] - y;
    long long x = cells[ci].N[0], N = cells[ci].N[1] - x;
    matA.AijBlock(M, N, y, x, &A[APrefixes[i]], M);
    matA.AijSetZero(M, N, y, x);
  }
}

void H2Matrix::matVecUpwardPass(const std::complex<double>* X_in, std::complex<double>* Z, const H2MatrixComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++) {
    long long M = VRowPrefixes[i + ibegin + 1] - VRowPrefixes[i + ibegin];
    long long N = VColPrefixes[i + ibegin + 1] - VColPrefixes[i + ibegin];
    if (0 < N) {
      const std::complex<double>* v = &V[VPrefixes[i]];
      const std::complex<double>* x = &X_in[VRowPrefixes[i + ibegin] - VRowPrefixes[ibegin]];
      std::complex<double>* z = &Z[VColPrefixes[i + ibegin]];
      agemv('C', M, N, v, M, x, 0, z);
    }
  }
  comm.neighbor_bcast(Z, VColPrefixes.data());
}

void H2Matrix::matVecHorizontalandDownwardPass(std::complex<double>* Y_out, const std::complex<double>* Z, std::complex<double>* Y, const CSR& Far) const {
  long long ibegin = Far.DiagCol;
  long long nodes = Far.RowIndex.size() - 1;
  long long lenX = URowPrefixes[ibegin + nodes] - URowPrefixes[ibegin];
  std::fill(Y_out, Y_out + lenX, std::complex<double>(0., 0.));

  for (long long i = 0; i < nodes; i++) {
    long long M = URowPrefixes[i + ibegin + 1] - URowPrefixes[i + ibegin];
    long long K = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
    if (0 < K) {
      std::complex<double>* y = &Y[UColPrefixes[i + ibegin]];
      for (long long ij = Far.RowIndex[i]; ij < Far.RowIndex[i + 1]; ij++) {
        long long j = Far.ColIndex[ij];
        long long N = VColPrefixes[j + 1] - VColPrefixes[j];

        const std::complex<double>* z = &Z[VColPrefixes[j]];
        const std::complex<double>* c = &C[CPrefixes[ij]];
        agemv('N', K, N, c, K, z, 1, y);
      }

      const std::complex<double>* u = &U[UPrefixes[i]];
      std::complex<double>* x = &Y_out[URowPrefixes[i + ibegin] - URowPrefixes[ibegin]];
      agemv('N', M, K, u, M, y, 0, x);
    }
  }
}

void H2Matrix::matVecLeafHorizontalPass(std::complex<double>* X_io, const std::complex<double>* Z, std::complex<double>* Y, const CSR& Far, const H2MatrixComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long lenX = URowPrefixes[ibegin + nodes] - URowPrefixes[ibegin];

  std::vector<std::complex<double>> X(URowPrefixes.back(), std::complex<double>(0., 0.));
  std::copy(&X_io[0], &X_io[lenX], X.begin() + URowPrefixes[ibegin]);
  comm.neighbor_bcast(X.data(), URowPrefixes.data());
  std::fill(&X_io[0], &X_io[lenX], std::complex<double>(0., 0.));

  for (long long i = 0; i < nodes; i++) {
    long long M = URowPrefixes[i + ibegin + 1] - URowPrefixes[i + ibegin];
    long long K = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
    std::complex<double>* w = X_io + URowPrefixes[i + ibegin] - URowPrefixes[ibegin];
    
    if (0 < K) {
      std::complex<double>* y = &Y[UColPrefixes[i + ibegin]];
      for (long long ij = Far.RowIndex[i]; ij < Far.RowIndex[i + 1]; ij++) {
        long long j = Far.ColIndex[ij];
        long long N = VColPrefixes[j + 1] - VColPrefixes[j];

        const std::complex<double>* z = &Z[VColPrefixes[j]];
        const std::complex<double>* c = &C[CPrefixes[ij]];
        agemv('N', K, N, c, K, z, 1, y);
      }

      const std::complex<double>* u = &U[UPrefixes[i]];
      agemv('N', M, K, u, M, y, 1, w);
    }

    const std::complex<double>* x = &X[URowPrefixes[i + ibegin]];
    const std::complex<double>* a = &A[APrefixes[i]];
    agemv('N', M, M, a, M, x, 1, w);
  }
}


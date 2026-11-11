
#include <h2matrix.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <csr.hpp>
#include <dense.hpp>

#include <numeric>
#include <algorithm>

extern "C" void zgemv_(const char*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*) noexcept;
inline void agemv(char transA, int M, int N, const std::complex<double>* A, int lda, const std::complex<double>* X, int beta, std::complex<double>* B)
{ int incx = 1; std::complex<double> one(1., 0.), zero(0., 0.); zgemv_(&transA, &M, &N, &one, A, &lda, X, &incx, beta ? &one : &zero, B, &incx); }

void H2Matrix::eval(double epi, DenseMatrix& matA, const long long localM[], const long long localN[], const CSR& rels, const H2MatrixComm& comm, long long localNoffset) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();

  URowPrefixes.resize(xlen + 1); UColPrefixes.resize(xlen + 1);
  VRowPrefixes.resize(xlen + 1); VColPrefixes.resize(xlen + 1);
  UPrefixes.resize(nodes + 1); VPrefixes.resize(nodes + 1);
  CPrefixes.resize(rels.RowIndex[nodes] + 1);

  if (0ll <= localNoffset) {
    APrefixes.resize(nodes + 1);
    std::transform(localM, &localM[nodes], localN, APrefixes.begin() + 1, std::multiplies<long long>());
    std::partial_sum(APrefixes.begin() + 1, APrefixes.end(), APrefixes.begin() + 1);
    APrefixes[0] = 0ll; A.resize(APrefixes.back());

    long long localMoffset = matA.rowOffsets[matA.prime_rank];
    for (long long i = 0ll; i < nodes; ++i) {
      long long M = localM[i], N = localN[i];
      matA.AijExtractBlock(M, N, localMoffset, localNoffset, &A[APrefixes[i]], M);
      localMoffset += M; localNoffset += N;
    }
  }

  if (rels.RowIndex[nodes]) {
    long long colOffset = 0ll;
    URowPrefixes[0] = UColPrefixes[0] = VRowPrefixes[0] = VColPrefixes[0] = UPrefixes[0] = VPrefixes[0] = CPrefixes[0] = 0ll;
    std::vector<long long> DimsMLr(&localM[0], &localM[nodes]), DimsNLr(&localN[0], &localN[nodes]);
    matA.compress(epi, DimsMLr, U);
    do {
      DenseMatrix matAH(std::reduce(&localN[0], &localN[nodes]), matA.rowOffsets.back(), matA.prime, matA.sub);
      matA.adjointA(matAH); matAH.compress(epi, DimsNLr, V);
      matA = DenseMatrix(std::reduce(DimsMLr.begin(), DimsMLr.end()), matAH.rowOffsets.back(), matAH.prime, matAH.sub);
      matAH.adjointA(matA); colOffset = matAH.rowOffsets[matAH.prime_rank];
    } while(0);

    std::transform(&localM[0], &localM[nodes], &DimsMLr[0], UPrefixes.begin() + 1, std::multiplies<long long>());
    std::transform(&localN[0], &localN[nodes], &DimsNLr[0], VPrefixes.begin() + 1, std::multiplies<long long>());
    std::inclusive_scan(UPrefixes.begin() + 1, UPrefixes.end(), UPrefixes.begin() + 1);
    std::inclusive_scan(VPrefixes.begin() + 1, VPrefixes.end(), VPrefixes.begin() + 1);

    std::vector<long long> seq(xlen + 1); std::vector<std::array<long long, 4>> dims(xlen + 1);
    for (long long i = 0ll; i < nodes; ++i) { dims[i + ibegin] = std::array<long long, 4>{ localM[i], localN[i], DimsMLr[i], DimsNLr[i] }; }
    for (long long i = 0ll; i <= xlen; ++i) { seq[i] = i * 4ll; }
    comm.neighbor_bcast((long long*)dims.data(), seq.data());
    std::inclusive_scan(dims.begin(), dims.end(), dims.begin(),
      [](const std::array<long long, 4>& a, const std::array<long long, 4>& b) { return std::array<long long, 4>{ a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3] }; });
    
    for (long long i = 0ll; i < xlen; ++i) {
      long long i1 = i + 1ll; const auto& [m, n, k, l] = dims[i];
      URowPrefixes[i1] = m; VRowPrefixes[i1] = n; UColPrefixes[i1] = k; VColPrefixes[i1] = l;
    }

    for (long long i = 0ll; i < nodes; ++i)
      std::transform(rels.ColIndex.begin() + rels.RowIndex[i], rels.ColIndex.begin() + rels.RowIndex[i + 1], CPrefixes.begin() + rels.RowIndex[i] + 1,
        [&](long long col) { return (UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin]) * (VColPrefixes[col + 1] - VColPrefixes[col]); });
    std::inclusive_scan(CPrefixes.begin() + 1, CPrefixes.end(), CPrefixes.begin() + 1);
    C.resize(CPrefixes.back());

    long long rowOffset = matA.rowOffsets[matA.prime_rank] - UColPrefixes[ibegin]; colOffset -= VColPrefixes[ibegin];
    for (long long i = 0; i < nodes; i++) {
      long long mC = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
      long long y = rowOffset + UColPrefixes[ibegin + i];
      for (long long ij = rels.RowIndex[i]; ij < rels.RowIndex[i + 1] && 0 < mC; ij++) {
        long long j = rels.ColIndex[ij];
        long long nC = VColPrefixes[j + 1] - VColPrefixes[j];
        long long x = colOffset + VColPrefixes[j];
        matA.AijExtractBlock(mC, nC, y, x, &C[CPrefixes[ij]], mC);
      }
    }
  } else {
    std::vector<long long> seq(xlen + 1); std::vector<std::array<long long, 2>> dims(xlen + 1);
    for (long long i = 0ll; i < nodes; ++i) { dims[i + ibegin] = std::array<long long, 2>{ localM[i], localN[i] }; }
    for (long long i = 0ll; i <= xlen; ++i) { seq[i] = i * 2ll; }
    comm.neighbor_bcast((long long*)dims.data(), seq.data());
    std::inclusive_scan(dims.begin(), dims.end(), dims.begin(),
      [](const std::array<long long, 2>& a, const std::array<long long, 2>& b) { return std::array<long long, 2>{ a[0] + b[0], a[1] + b[1] }; });
    
    URowPrefixes[0] = VRowPrefixes[0] = CPrefixes[0] = 0ll;
    for (long long i = 0ll; i < xlen; ++i) {
      long long i1 = i + 1ll; const auto& [m, n] = dims[i];
      URowPrefixes[i1] = m; VRowPrefixes[i1] = n;
    }

    std::fill(UColPrefixes.begin(), UColPrefixes.end(), 0ll);
    std::fill(VColPrefixes.begin(), VColPrefixes.end(), 0ll);
    std::fill(UPrefixes.begin(), UPrefixes.end(), 0ll);
    std::fill(VPrefixes.begin(), VPrefixes.end(), 0ll);
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

void H2Matrix::matVecHorizontalandDownwardPass(std::complex<double>* Y_out, const std::complex<double>* Z, std::complex<double>* Y, const CSR& rels, const H2MatrixComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long lenX = URowPrefixes[ibegin + nodes] - URowPrefixes[ibegin];
  std::fill(Y_out, Y_out + lenX, std::complex<double>(0., 0.));

  for (long long i = 0; i < nodes; i++) {
    long long M = URowPrefixes[i + ibegin + 1] - URowPrefixes[i + ibegin];
    long long K = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
    if (0 < K) {
      std::complex<double>* y = &Y[UColPrefixes[i + ibegin]];
      for (long long ij = rels.RowIndex[i]; ij < rels.RowIndex[i + 1]; ij++) {
        long long j = rels.ColIndex[ij];
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

void H2Matrix::matVecLeafHorizontalPass(std::complex<double>* X_io, const std::complex<double>* Z, std::complex<double>* Y, const CSR& rels, const H2MatrixComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long lenX = VRowPrefixes[ibegin + nodes] - VRowPrefixes[ibegin];

  std::vector<std::complex<double>> X(VRowPrefixes.back(), std::complex<double>(0., 0.));
  std::copy(&X_io[0], &X_io[lenX], X.begin() + VRowPrefixes[ibegin]);
  comm.neighbor_bcast(X.data(), VRowPrefixes.data());
  std::fill(&X_io[0], &X_io[lenX], std::complex<double>(0., 0.));

  for (long long i = 0; i < nodes; i++) {
    long long M = URowPrefixes[i + ibegin + 1] - URowPrefixes[i + ibegin];
    long long K = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
    std::complex<double>* w = &X_io[URowPrefixes[i + ibegin] - URowPrefixes[ibegin]];
    
    if (0 < K) {
      std::complex<double>* y = &Y[UColPrefixes[i + ibegin]];
      for (long long ij = rels.RowIndex[i]; ij < rels.RowIndex[i + 1]; ij++) {
        long long j = rels.ColIndex[ij];
        long long N = VColPrefixes[j + 1] - VColPrefixes[j];

        const std::complex<double>* z = &Z[VColPrefixes[j]];
        const std::complex<double>* c = &C[CPrefixes[ij]];
        agemv('N', K, N, c, K, z, 1, y);
      }

      const std::complex<double>* u = &U[UPrefixes[i]];
      agemv('N', M, K, u, M, y, 1, w);
    }

    long long N = VRowPrefixes[i + ibegin + 1] - VRowPrefixes[i + ibegin];
    const std::complex<double>* x = &X[VRowPrefixes[i + ibegin]];
    const std::complex<double>* a = &A[APrefixes[i]];
    agemv('N', M, N, a, M, x, 1, w);
  }
}

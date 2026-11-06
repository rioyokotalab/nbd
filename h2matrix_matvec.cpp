
#include <h2matrix.hpp>
#include <comm-mpi.hpp>
#include <csr.hpp>
#include <lapack_f2c.hpp>

inline void agemv(char transA, int M, int N, const std::complex<double>* A, int lda, const std::complex<double>* X, int beta, std::complex<double>* B)
{ int incx = 1; std::complex<double> one(1., 0.), zero(0., 0.); zgemv_(&transA, &M, &N, &one, A, &lda, X, &incx, beta ? &one : &zero, B, &incx); }

void H2Matrix::matVecUpwardPass(const std::complex<double>* X_in, std::complex<double>* Z, const H2MatrixComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();

  for (long long i = 0; i < nodes; i++) {
    long long M = Dims[i + ibegin];
    long long N = DimsLr[i + ibegin];
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
    long long M = Dims[i + ibegin];
    long long K = DimsLr[i + ibegin];
    if (0 < K) {
      std::complex<double>* y = Y + DimsLrOffsets[i + ibegin];
      for (long long ij = Far.RowIndex[i]; ij < Far.RowIndex[i + 1]; ij++) {
        long long j = Far.ColIndex[ij];
        long long N = DimsLr[j];

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
    long long M = Dims[i + ibegin];
    long long K = DimsLr[i + ibegin];
    std::complex<double>* w = X_io + DimsOffsets[i + ibegin] - DimsOffsets[ibegin];
    
    if (0 < K) {
      std::complex<double>* y = Y + DimsLrOffsets[i + ibegin];
      for (long long ij = Far.RowIndex[i]; ij < Far.RowIndex[i + 1]; ij++) {
        long long j = Far.ColIndex[ij];
        long long N = DimsLr[j];

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


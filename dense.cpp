
#include <dense.hpp>
#include <lapack_f2c.hpp>
#include <algorithm>
#include <numeric>
#include <cassert>

DenseMatrix::DenseMatrix(long long localM, long long N, MPI_Comm world) : N(N), world(world) {
  if (0 < N) {
    A.resize(localM * N, 0.); int mpi_size = 1;
    MPI_Comm_rank(world, &mpi_rank); MPI_Comm_size(world, &mpi_size);
    rowOffsets.resize(mpi_size + 1); rowOffsets[0] = 0;
    MPI_Allgather(&localM, 1, MPI_LONG_LONG_INT, &rowOffsets[1], 1, MPI_LONG_LONG_INT, world);
    std::inclusive_scan(rowOffsets.begin() + 1, rowOffsets.end(), rowOffsets.begin() + 1);
  }
}

void DenseMatrix::matVecA(const DenseMatrix& X, std::complex<double>* B) const {
  assert(X.rowOffsets.back() == N && X.N == 1ll);
  std::vector<std::complex<double>> X_in(N);
  std::vector<int> recvcounts(rowOffsets.size() - 1), displs(rowOffsets.size() - 1);
  std::transform(X.rowOffsets.begin() + 1, X.rowOffsets.end(), X.rowOffsets.begin(), recvcounts.begin(), [](long long end, long long begin) { return (int)(end - begin); });
  std::transform(X.rowOffsets.begin(), std::prev(X.rowOffsets.end()), displs.begin(), [](long long begin) { return (int)begin; });

  MPI_Allgatherv(&(X.A)[0], int(X.rowOffsets[mpi_rank + 1] - X.rowOffsets[mpi_rank]), MPI_C_DOUBLE_COMPLEX, &X_in[0], &recvcounts[0], &displs[0], MPI_C_DOUBLE_COMPLEX, world);
  int M = int(rowOffsets[mpi_rank + 1] - rowOffsets[mpi_rank]);
  auto gemv = [](int m, int n, const std::complex<double>* a, const std::complex<double>* x, std::complex<double>* b) {
    char transA = 'T'; int incx = 1; std::complex<double> one(1., 0.), zero(0., 0.); zgemv_(&transA, &n, &m, &one, a, &n, x, &incx, &zero, b, &incx);
  };
  gemv(M, N, &A[0], &X_in[0], B);
}

void DenseMatrix::AijBlock(long long mA, long long nA, long long iA, long long jA, std::complex<double>* A_out, long long stride) const {
  long long row_begin = std::max(rowOffsets[mpi_rank], iA), row_end = std::min(rowOffsets[mpi_rank + 1], iA + mA);
  long long col_begin = std::max(0ll, jA), col_end = std::min(N, jA + nA);
  if (row_begin < row_end && col_begin < col_end) {
    long long rows = row_end - row_begin, cols = col_end - col_begin;
    const std::complex<double>* src = &A[(row_begin - rowOffsets[mpi_rank]) * N + col_begin];
    std::complex<double>* dst = &A_out[(row_begin - iA) + (col_begin - jA) * stride];
    for (long long x = 0; x < cols; ++x) for (long long y = 0; y < rows; ++y)
    { dst[y + stride * x] = src[y * N + x]; }
  }
}

void DenseMatrix::adjointA(DenseMatrix& B) const {
  assert(B.rowOffsets.back() == N && rowOffsets.back() == B.N);
  long long MA = rowOffsets[mpi_rank + 1] - rowOffsets[mpi_rank], MB = B.rowOffsets[mpi_rank + 1] - B.rowOffsets[mpi_rank];
  for (int i = 0; i < int(rowOffsets.size()) - 1; ++i) {
    long long NA = B.rowOffsets[i + 1] - B.rowOffsets[i], NB = rowOffsets[i + 1] - rowOffsets[i];
    std::vector<std::complex<double>> AH(MA * NA), BH(MB * NB); AijBlock(MA, NA, rowOffsets[mpi_rank], B.rowOffsets[i], &AH[0], MA);
    MPI_Sendrecv(&AH[0], AH.size(), MPI_C_DOUBLE_COMPLEX, i, 0, &BH[0], BH.size(), MPI_C_DOUBLE_COMPLEX, i, 0, world, MPI_STATUS_IGNORE);
    std::complex<double>* dst = &(B.A)[rowOffsets[i]];
    for (long long x = 0; x < MB; ++x) for (long long y = 0; y < NB; ++y)
    { dst[y + B.N * x] = std::conj(BH[y + NB * x]); }
  }
}

void DenseMatrix::AijSetZero(long long mA, long long nA, long long iA, long long jA) {
  long long row_begin = std::max(rowOffsets[mpi_rank], iA), row_end = std::min(rowOffsets[mpi_rank + 1], iA + mA);
  long long col_begin = std::max(0ll, jA), col_end = std::min(N, jA + nA);
  if (row_begin < row_end && col_begin < col_end) {
    long long rows = row_end - row_begin, cols = col_end - col_begin;
    std::complex<double>* src = &A[(row_begin - rowOffsets[mpi_rank]) * N + col_begin];
    for (long long y = 0; y < rows; ++y) { std::fill_n(&src[y * N], cols, std::complex<double>(0., 0.)); }
  }
}

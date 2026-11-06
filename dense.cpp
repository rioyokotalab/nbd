
#include <dense.hpp>
#include <lapack_f2c.hpp>
#include <algorithm>

inline void copy2d(int M, int N, const std::complex<double>* A, int lda, std::complex<double>* B, int ldb)
{ char uplo = 'A'; zlacpy_(&uplo, &M, &N, A, &lda, B, &ldb); }
inline void ngemv(int M, int N, const std::complex<double>* A, int lda, const std::complex<double>* X, std::complex<double>* B)
{ char transA = 'N'; int incx = 1; std::complex<double> one(1., 0.), zero(0., 0.); zgemv_(&transA, &M, &N, &one, A, &lda, X, &incx, &zero, B, &incx); }

DenseMatrix::DenseMatrix(long long RowStart, long long RowEnd, long long N, MPI_Comm world) : RowStart(RowStart), RowEnd(RowEnd), N(N), world(world) {
  if (RowStart < RowEnd && 0 < N) {
    A.resize((RowEnd - RowStart) * N);
    std::fill(A.begin(), A.end(), std::complex<double>(0., 0.));

    int mpi_size = 1;
    MPI_Comm_size(world, &mpi_size);
    rowOffsets.resize(mpi_size + 1);
    rowOffsets[0] = 0;
    MPI_Allgather(&RowEnd, 1, MPI_LONG_LONG_INT, &rowOffsets[1], 1, MPI_LONG_LONG_INT, world);
  }
}

void DenseMatrix::Aij_block(long long mA, long long nA, long long iA, long long jA, std::complex<double>* A_out, long long stride) const {
  long long lda = RowEnd - RowStart;
  long long row_begin = std::max(RowStart, iA);
  long long row_end = std::min(RowEnd, iA + mA);
  long long col_begin = std::max(0ll, jA);
  long long col_end = std::min(N, jA + nA);

  const std::complex<double>* src = &A[(row_begin - RowStart) + col_begin * lda];
  std::complex<double>* dst = &A_out[(row_begin - iA) + (col_begin - jA) * stride];
  copy2d(row_end - row_begin, col_end - col_begin, src, lda, dst, stride);
}

void DenseMatrix::matVecA(std::complex<double>* X) const {
  long long M = RowEnd - RowStart;
  std::vector<std::complex<double>> X_in(N);
  std::vector<int> recvcounts(rowOffsets.size() - 1), displs(rowOffsets.size() - 1);
  std::transform(rowOffsets.begin() + 1, rowOffsets.end(), rowOffsets.begin(), recvcounts.begin(), [](long long end, long long begin) { return (int)(end - begin); });
  std::transform(rowOffsets.begin(), std::prev(rowOffsets.end()), displs.begin(), [](long long begin) { return (int)begin; });
  MPI_Allgatherv(X, M, MPI_C_DOUBLE_COMPLEX, &X_in[0], &recvcounts[0], &displs[0], MPI_C_DOUBLE_COMPLEX, world);
  ngemv(M, N, &A[0], M, &X_in[0], X);
}

void DenseMatrix::adjointA(std::complex<double>* AH) const {
  int mpi_rank, mpi_size; MPI_Comm_rank(world, &mpi_rank); MPI_Comm_size(world, &mpi_size);
  std::vector<int> recvcounts(mpi_size), displs(mpi_size);
  long long M = RowEnd - RowStart;
  std::transform(rowOffsets.begin() + 1, rowOffsets.end(), rowOffsets.begin(), recvcounts.begin(), [=](long long end, long long begin) { return (int)(M * (end - begin)); });
  std::transform(rowOffsets.begin(), std::prev(rowOffsets.end()), displs.begin(), [=](long long begin) { return (int)(M * begin); });

  for (int i = 0; i < mpi_size; ++i) {
    long long Aoffset = M * rowOffsets[i];
    long long Ni = rowOffsets[i + 1] - rowOffsets[i];
    const std::complex<double>* Ai = &A[Aoffset];
    std::complex<double>* AHi = &AH[Aoffset];
    std::vector<std::complex<double>> AHc(M * Ni);
    for (long long y = 0; y < Ni; ++y) for (long long x = 0; x < M; ++x)
    { AHc[y + x * Ni] = std::conj(Ai[x + y * M]); }
    MPI_Sendrecv(&AHc[0], AHc.size(), MPI_C_DOUBLE_COMPLEX, i, i ^ mpi_rank, AHi, AHc.size(), MPI_C_DOUBLE_COMPLEX, i, i ^ mpi_rank, world, MPI_STATUS_IGNORE);
  }
}

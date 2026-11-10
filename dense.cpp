
#include <dense.hpp>
#include <lapack_f2c.hpp>
#include <algorithm>
#include <numeric>
#include <cassert>

DenseMatrix::DenseMatrix(long long localM, long long N, MPI_Comm prime, MPI_Comm sub) : N(N), prime_rank(0), prime(prime), sub(sub) {
  if (0ll < N) {
    A.resize(localM * N);
    if (prime != MPI_COMM_NULL) {
      int mpi_size = 1;
      MPI_Comm_rank(prime, &prime_rank); MPI_Comm_size(prime, &mpi_size);
      rowOffsets.resize(mpi_size + 1); rowOffsets[0] = 0;
      MPI_Allgather(&localM, 1, MPI_LONG_LONG_INT, &rowOffsets[1], 1, MPI_LONG_LONG_INT, prime);
      std::partial_sum(rowOffsets.begin() + 1, rowOffsets.end(), rowOffsets.begin() + 1);
    }
    if (sub != MPI_COMM_NULL) {
      int comm[2]{ prime_rank, int(rowOffsets.size()) - 1 };
      MPI_Bcast(comm, 2, MPI_INT, 0, sub);
      prime_rank = comm[0]; rowOffsets.resize(comm[1] + 1);
      MPI_Bcast(&rowOffsets[0], int(rowOffsets.size()), MPI_LONG_LONG_INT, 0, sub);
    }
  }
}

static inline void MPI_Bcast_large(std::complex<double>* data, long long len, int root, MPI_Comm comm) {
  if (0x40000000ll < len) {
    MPI_Datatype block_type;
    MPI_Type_contiguous(1 << 30, MPI_C_DOUBLE_COMPLEX, &block_type); MPI_Type_commit(&block_type);
    int blocks = int(((unsigned long long)len) >> 30) & 0x3fffffff, rem = int(len) & 0x3fffffff;
    MPI_Bcast(data, blocks, block_type, root, comm); MPI_Type_free(&block_type);
    if (rem) { MPI_Bcast(&data[((long long)blocks) << 30], rem, MPI_C_DOUBLE_COMPLEX, root, comm); }
  } else { MPI_Bcast(data, int(len), MPI_C_DOUBLE_COMPLEX, root, comm); }
}

void DenseMatrix::matVecA(const DenseMatrix& X, std::complex<double>* B) const {
  assert(X.rowOffsets.back() == N && X.N == 1ll);
  if (prime != MPI_COMM_NULL) {
    std::vector<std::complex<double>> X_in(N);
    std::vector<int> recvcounts(rowOffsets.size() - 1), displs(rowOffsets.size() - 1);
    std::transform(X.rowOffsets.begin() + 1, X.rowOffsets.end(), X.rowOffsets.begin(), recvcounts.begin(), [](long long end, long long begin) { return (int)(end - begin); });
    std::transform(X.rowOffsets.begin(), std::prev(X.rowOffsets.end()), displs.begin(), [](long long begin) { return (int)begin; });

    MPI_Allgatherv(&(X.A)[0], int(X.rowOffsets[prime_rank + 1] - X.rowOffsets[prime_rank]), MPI_C_DOUBLE_COMPLEX, &X_in[0], &recvcounts[0], &displs[0], MPI_C_DOUBLE_COMPLEX, prime);
    int M = int(rowOffsets[prime_rank + 1] - rowOffsets[prime_rank]);
    auto gemv = [](int m, int n, const std::complex<double>* a, const std::complex<double>* x, std::complex<double>* b) {
      char transA = 'C'; int incx = 1; std::complex<double> one(1., 0.), zero(0., 0.); zgemv_(&transA, &n, &m, &one, a, &n, x, &incx, &zero, b, &incx);
    };
    gemv(M, N, &A[0], &X_in[0], B);
  }
  if (sub != MPI_COMM_NULL)
  { MPI_Bcast_large(B, rowOffsets[prime_rank + 1] - rowOffsets[prime_rank], 0, sub); }
}

void DenseMatrix::adjointA(DenseMatrix& B) const {
  assert(B.rowOffsets.back() == N && rowOffsets.back() == B.N);
  if (prime != MPI_COMM_NULL) {
    long long MA = rowOffsets[prime_rank + 1] - rowOffsets[prime_rank], MB = B.rowOffsets[B.prime_rank + 1] - B.rowOffsets[B.prime_rank];
    MPI_Datatype typeA, typeB;
    MPI_Type_contiguous(MA, MPI_C_DOUBLE_COMPLEX, &typeA); MPI_Type_commit(&typeA);
    MPI_Type_contiguous(MB, MPI_C_DOUBLE_COMPLEX, &typeB); MPI_Type_commit(&typeB);
    for (int i = 0; i < int(rowOffsets.size()) - 1; ++i) {
      long long NA = B.rowOffsets[i + 1] - B.rowOffsets[i], NB = rowOffsets[i + 1] - rowOffsets[i];
      std::vector<std::complex<double>> AH(MA * NA), BH(MB * NB); AijBlock(MA, NA, rowOffsets[prime_rank], B.rowOffsets[i], &AH[0], MA);
      MPI_Sendrecv(&AH[0], NA, typeA, i, 0, &BH[0], NB, typeB, i, 0, prime, MPI_STATUS_IGNORE);
      std::complex<double>* dst = &(B.A)[rowOffsets[i]];
      for (long long x = 0; x < MB; ++x) for (long long y = 0; y < NB; ++y)
      { dst[y + B.N * x] = BH[y + NB * x]; }
    }
    MPI_Type_free(&typeA); MPI_Type_free(&typeB);
  }
  if (sub != MPI_COMM_NULL)
  { MPI_Bcast_large(&B.A[0], B.N * (B.rowOffsets[B.prime_rank + 1] - B.rowOffsets[B.prime_rank]), 0, sub); }
}

static inline int svd_fit_transform_right(double epi, int M, int N, std::complex<double>* A, std::complex<double>* V) {
  char jobu = 'O', jobvt = 'S'; int Lwork = -1, info = 0, n1 = 1, ldv = std::min(M, N); std::complex<double> work_s;
  std::vector<std::complex<double>> VH(((long long)ldv) * ((long long)N));
  std::vector<double> rwork(std::max(1, 5 * std::min(M, N))), S(std::min(M, N));
  zgesvd_(&jobu, &jobvt, &M, &N, A, &M, &S[0], nullptr, &n1, &VH[0], &ldv, &work_s, &Lwork, &rwork[0], &info);
  Lwork = int(std::ceil(work_s.real())); std::vector<std::complex<double>> work_a(Lwork);
  zgesvd_(&jobu, &jobvt, &M, &N, A, &M, &S[0], nullptr, &n1, &VH[0], &ldv, &work_a[0], &Lwork, &rwork[0], &info);

  std::vector<double> Snrm(S.size());
  std::transform_inclusive_scan(S.rbegin(), S.rend(), Snrm.rbegin(), std::plus<double>(), [](double x) { return x * x; });
  if (Snrm[0] < std::numeric_limits<double>::min()) { return 0; } else {
    double s0 = Snrm[0] * std::pow(epi, 2);
    int rank = int(std::distance(Snrm.begin(), std::find_if(Snrm.begin(), Snrm.end(), [=](double s) { return s < s0; })));
    for (int i = 0; i < rank; ++i) {
      double s = S[i]; std::for_each(&A[int64_t(i) * int64_t(M)], &A[int64_t(i + 1) * int64_t(M)], [=](std::complex<double>& x) { x *= s; });
      for (int j = 0; j < N; ++j)
      { V[((long long)j) + ((long long)N) * ((long long)i)] = std::conj(VH[((long long)j) * ((long long)ldv) + ((long long)i)]); }
    }
    return rank;
  }
}

void DenseMatrix::compress(double epi, std::vector<long long>& recipe, std::vector<std::complex<double>>& basis) {
  if (0ll < N) {
    long long localM = std::reduce(recipe.begin(), recipe.end());
    assert(localM == rowOffsets[prime_rank + 1] - rowOffsets[prime_rank]);

    if (prime != MPI_COMM_NULL) {
      long long iterF = 0ll, iterL = 0ll;
      basis.clear(); std::vector<long long>::iterator recipe_iter = recipe.begin();
      while (recipe_iter != recipe.end()) {
        long long block = *recipe_iter;
        if (0ll < block) {
          if (iterL < iterF) {
            std::vector<std::complex<double>> Acpy(&A[iterF * N], &A[(iterF + block) * N]);
            std::copy(Acpy.begin(), Acpy.end(), &A[iterL * N]);
          }
          std::vector<std::complex<double>> V(std::min(N, block) * block);
          int rank = *recipe_iter = svd_fit_transform_right(epi, N, block, &A[iterL * N], &V[0]);
          basis.insert(basis.end(), &V[0], &V[(long long)rank * block]);
          iterL += rank;
        }
        iterF += block; recipe_iter = std::next(recipe_iter);
      }
      A.resize(iterL * N);
      MPI_Allgather(&iterL, 1, MPI_LONG_LONG_INT, &rowOffsets[1], 1, MPI_LONG_LONG_INT, prime);
      std::partial_sum(rowOffsets.begin() + 1, rowOffsets.end(), rowOffsets.begin() + 1);
    }
    if (sub != MPI_COMM_NULL) {
      long long comm[2]{ (long long)A.size(), (long long)basis.size() };
      MPI_Bcast(comm, 2, MPI_LONG_LONG_INT, 0, sub);
      MPI_Bcast(&rowOffsets[0], int(rowOffsets.size()), MPI_LONG_LONG_INT, 0, sub);
      A.resize(comm[0]); basis.resize(comm[1]);
      MPI_Bcast_large(&A[0], comm[0], 0, sub);
      MPI_Bcast_large(&basis[0], comm[1], 0, sub);
    }
    A.shrink_to_fit();
  }
}

void DenseMatrix::AijBlock(long long mA, long long nA, long long iA, long long jA, std::complex<double>* A_out, long long stride) const {
  long long row_begin = std::max(rowOffsets[prime_rank], iA), row_end = std::min(rowOffsets[prime_rank + 1], iA + mA);
  long long col_begin = std::max(0ll, jA), col_end = std::min(N, jA + nA);
  if (row_begin < row_end && col_begin < col_end) {
    long long rows = row_end - row_begin, cols = col_end - col_begin;
    const std::complex<double>* src = &A[(row_begin - rowOffsets[prime_rank]) * N + col_begin];
    std::complex<double>* dst = &A_out[(row_begin - iA) + (col_begin - jA) * stride];
    for (long long x = 0; x < cols; ++x) for (long long y = 0; y < rows; ++y)
    { dst[y + stride * x] = std::conj(src[y * N + x]); }
  }
}

void DenseMatrix::AijSetZero(long long mA, long long nA, long long iA, long long jA) {
  long long row_begin = std::max(rowOffsets[prime_rank], iA), row_end = std::min(rowOffsets[prime_rank + 1], iA + mA);
  long long col_begin = std::max(0ll, jA), col_end = std::min(N, jA + nA);
  if (row_begin < row_end && col_begin < col_end) {
    long long rows = row_end - row_begin, cols = col_end - col_begin;
    std::complex<double>* src = &A[(row_begin - rowOffsets[prime_rank]) * N + col_begin];
    for (long long y = 0; y < rows; ++y) { std::fill_n(&src[y * N], cols, std::complex<double>(0., 0.)); }
  }
}

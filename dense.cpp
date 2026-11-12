
#include <dense.hpp>
#include <algorithm>
#include <numeric>
#include <cassert>

extern "C" void daxpy_(const int*, const double*, const double*, const int*, double*, const int*) noexcept;
extern "C" void zgemv_(const char*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*) noexcept;
extern "C" void zgesvd_(const char*, const char*, const int*, const int*, std::complex<double>*, const int*, double*, std::complex<double>*, const int*, std::complex<double>*, const int*, std::complex<double>*, const int*, double*, int*) noexcept;

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
      prime_rank = comm[0]; rowOffsets.resize(comm[1] + 1); rowOffsets[0] = 0ll;
      MPI_Bcast(&rowOffsets[1], comm[1], MPI_LONG_LONG_INT, 0, sub);
    }
  }
}

constexpr int kBlockDoubles = 1 << 27;  // 1 GiB with 8-byte doubles.

static inline void bcastLarge(double* data, long long count, int root, MPI_Comm comm) {
  if (kBlockDoubles <= count) {
    MPI_Datatype block;
    MPI_Type_contiguous(kBlockDoubles, MPI_DOUBLE, &block); MPI_Type_commit(&block);

    while (kBlockDoubles <= count) {
      long long blocks = std::min(count / kBlockDoubles, (long long)std::numeric_limits<int>::max());
      MPI_Bcast(data, int(blocks), block, root, comm); 
      long long done = blocks * kBlockDoubles;
      data += done; count -= done;
    }
    MPI_Type_free(&block);
  }
  if (count != 0) { MPI_Bcast(data, int(count), MPI_DOUBLE, root, comm); }
}

extern "C" void sumDoubleBlocks(void* in, void* inout, int* len, MPI_Datatype*) {
  const double* src = (const double*)in, one = 1.; double* dst = (double*)inout;
  const int n = *len, incx = 1; for (int i = 0; i < n; ++i)
  { daxpy_(&kBlockDoubles, &one, src, &incx, dst, &incx); src += kBlockDoubles; dst += kBlockDoubles; }
}

static inline void allreduceSumLarge(double* data, long long count, MPI_Comm comm) {
  if (kBlockDoubles <= count) {
    MPI_Datatype block; MPI_Op sum;
    MPI_Type_contiguous(kBlockDoubles, MPI_DOUBLE, &block); MPI_Type_commit(&block);
    MPI_Op_create(sumDoubleBlocks, 1, &sum);

    while (kBlockDoubles <= count) {
      const long long blocks = std::min(count / kBlockDoubles, (long long)std::numeric_limits<int>::max());
      MPI_Allreduce(MPI_IN_PLACE, data, int(blocks), block, sum, comm);
      const long long done = blocks * kBlockDoubles;
      data += done; count -= done;
    }
    MPI_Op_free(&sum); MPI_Type_free(&block);
  }
  if (count != 0) { MPI_Allreduce(MPI_IN_PLACE, data, int(count), MPI_DOUBLE, MPI_SUM, comm); }
}

void DenseMatrix::matVecA(const DenseMatrix& matA, std::complex<double>* B) const {
  assert(rowOffsets.back() == matA.N && N == 1ll);
  assert(prime == matA.prime && sub == matA.sub);
  if (prime != MPI_COMM_NULL) {
    long long maxN = std::transform_reduce(rowOffsets.begin() + 1, rowOffsets.end(), rowOffsets.begin(), 0ll, [](long long a, long long b) { return std::max(a, b); }, std::minus<long long>());
    long long prime_size = (long long)rowOffsets.size() - 1ll;
    std::vector<std::complex<double>> X_all(prime_size * maxN);
    std::transform(A.begin(), A.end(), &X_all[prime_rank * maxN], [](std::complex<double> a) { return std::conj(a); });
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &X_all[0], maxN, MPI_C_DOUBLE_COMPLEX, prime);

    const int m = int(matA.rowOffsets[matA.prime_rank + 1] - matA.rowOffsets[matA.prime_rank]), lda = int(matA.N), incx = 1;
    const char transA = 'C'; const std::complex<double> one(1., 0.), zero(0., 0.);
    for (long long i = 0; i < prime_size; ++i) {
      const int n = int(rowOffsets[i + 1] - rowOffsets[i]);
      zgemv_(&transA, &n, &m, &one, &(matA.A)[rowOffsets[i]], &lda, &X_all[i * maxN], &incx, i == 0ll ? &zero : &one, B, &incx);
    }
  }
  if (sub != MPI_COMM_NULL)
  { bcastLarge((double*)B, (rowOffsets[prime_rank + 1] - rowOffsets[prime_rank]) * 2ll, 0, sub); }
}

void DenseMatrix::adjointA(DenseMatrix& B) {
  assert(B.rowOffsets.back() == N && rowOffsets.back() == B.N);
  assert(prime == B.prime && sub == B.sub);
  if (prime != MPI_COMM_NULL) {
    long long maxM = std::transform_reduce(rowOffsets.begin() + 1, rowOffsets.end(), rowOffsets.begin(), 0ll, [](long long a, long long b) { return std::max(a, b); }, std::minus<long long>());
    long long maxN = std::transform_reduce(B.rowOffsets.begin() + 1, B.rowOffsets.end(), B.rowOffsets.begin(), 0ll, [](long long a, long long b) { return std::max(a, b); }, std::minus<long long>());
    long long count = maxM * maxN, prime_size = (long long)rowOffsets.size() - 1ll; B.A.resize(prime_size * count);

    long long MA = rowOffsets[prime_rank + 1] - rowOffsets[prime_rank];
    for (long long i = 0; i < prime_size; ++i) {
      long long NA = B.rowOffsets[i + 1] - B.rowOffsets[i];
      const std::complex<double>* src = &A[B.rowOffsets[i]]; std::complex<double>* dst = &(B.A)[i * count];
      for (long long y = 0; y < MA; ++y) for (long long x = 0; x < NA; ++x) { dst[y + maxM * x] = std::conj(src[y * N + x]); }
    }

    A.resize(prime_size * count);
    MPI_Datatype block; MPI_Type_contiguous(maxM, MPI_C_DOUBLE_COMPLEX, &block); MPI_Type_commit(&block);
    MPI_Alltoall(B.A.data(), maxN, block, A.data(), maxN, block, prime); MPI_Type_free(&block);
    
    long long MB = B.rowOffsets[B.prime_rank + 1] - B.rowOffsets[B.prime_rank]; B.A.resize(MB * B.N);
    for (long long i = 0; i < prime_size; ++i) {
      long long NB = rowOffsets[i + 1] - rowOffsets[i];
      const std::complex<double>* src = &A[i * count]; std::complex<double>* dst = &(B.A)[rowOffsets[i]];
      for (long long x = 0; x < MB; ++x) for (long long y = 0; y < NB; ++y) { dst[y + B.N * x] = src[y + maxM * x]; }
    }
  }
  if (sub != MPI_COMM_NULL)
  { bcastLarge((double*)&B.A[0], B.N * (B.rowOffsets[B.prime_rank + 1] - B.rowOffsets[B.prime_rank]) * 2ll, 0, sub); }
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
  double s0 = Snrm[0] * std::pow(epi, 2);
  int rank = int(std::distance(Snrm.begin(), std::find_if(Snrm.begin(), Snrm.end(), [=](double s) { return s < s0; })));
  for (int i = 0; i < rank; ++i) {
    double s = S[i]; std::for_each(&A[int64_t(i) * int64_t(M)], &A[int64_t(i + 1) * int64_t(M)], [=](std::complex<double>& x) { x *= s; });
    for (int j = 0; j < N; ++j)
    { V[((long long)j) + ((long long)N) * ((long long)i)] = std::conj(VH[((long long)j) * ((long long)ldv) + ((long long)i)]); }
  }
  return rank;
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
      rowOffsets[0] = (long long)basis.size();
      MPI_Bcast(&rowOffsets[0], int(rowOffsets.size()), MPI_LONG_LONG_INT, 0, sub);
      long long basis_len = rowOffsets[0]; basis.resize(basis_len); rowOffsets[0] = 0ll;
      long long A_len = N * (rowOffsets[prime_rank + 1] - rowOffsets[prime_rank]); A.resize(A_len);
      bcastLarge((double*)&basis[0], basis_len * 2ll, 0, sub);
      bcastLarge((double*)&A[0], A_len * 2ll, 0, sub);
    }
    A.shrink_to_fit();
  }
}

void DenseMatrix::copyFromA(const DenseMatrix& matA) {
  assert(rowOffsets.back() == matA.rowOffsets.back() && N == matA.N);
  long long dst_begin = rowOffsets[prime_rank], src_begin = matA.rowOffsets[matA.prime_rank];
  long long lo = std::max(dst_begin, src_begin), hi = std::min(rowOffsets[prime_rank + 1], matA.rowOffsets[matA.prime_rank + 1]);

  bool elected = true;
  if (sub != MPI_COMM_NULL) {
    int sub_rank, sub_size; MPI_Comm_rank(sub, &sub_rank); MPI_Comm_size(sub, &sub_size);
    std::vector<int> src_groups(sub_size);
    MPI_Allgather(&matA.prime_rank, 1, MPI_INT, src_groups.data(), 1, MPI_INT, sub);
    std::vector<int>::iterator my_pos = src_groups.begin() + sub_rank;
    elected = std::find(src_groups.begin(), my_pos, matA.prime_rank) == my_pos;
  }

  bool contributes = elected && lo < hi;
  if (contributes) {
    long long src_offset = (lo - src_begin) * N, dst_offset = (lo - dst_begin) * N, count = (hi - lo) * N;
    std::copy_n(&matA.A[src_offset], count, &A[dst_offset]);
  } else { std::fill(A.begin(), A.end(), std::complex<double>(0., 0.)); }
  if (sub != MPI_COMM_NULL) { allreduceSumLarge((double*)A.data(), 2ll * A.size(), sub); }
}

void DenseMatrix::AijSetBlock(long long mA, long long nA, long long iA, long long jA, const std::complex<double>* A_in, long long stride) {
  long long row_begin = std::max(rowOffsets[prime_rank], iA), row_end = std::min(rowOffsets[prime_rank + 1], iA + mA);
  long long col_begin = std::max(0ll, jA), col_end = std::min(N, jA + nA);
  if (row_begin < row_end && col_begin < col_end) {
    long long rows = row_end - row_begin, cols = col_end - col_begin;
    const std::complex<double>* src = &A_in[(row_begin - iA) + (col_begin - jA) * stride];
    std::complex<double>* dst = &A[(row_begin - rowOffsets[prime_rank]) * N + col_begin];
    for (long long y = 0; y < rows; ++y) for (long long x = 0; x < cols; ++x) { dst[y * N + x] = std::conj(src[y + x * stride]); }
  }
}

void DenseMatrix::AijExtractBlock(long long mA, long long nA, long long iA, long long jA, std::complex<double>* A_out, long long stride) {
  long long row_begin = std::max(rowOffsets[prime_rank], iA), row_end = std::min(rowOffsets[prime_rank + 1], iA + mA);
  long long col_begin = std::max(0ll, jA), col_end = std::min(N, jA + nA);
  if (row_begin < row_end && col_begin < col_end) {
    long long rows = row_end - row_begin, cols = col_end - col_begin;
    std::complex<double>* src = &A[(row_begin - rowOffsets[prime_rank]) * N + col_begin];
    std::complex<double>* dst = &A_out[(row_begin - iA) + (col_begin - jA) * stride];
    for (long long y = 0; y < rows; ++y) {
      for (long long x = 0; x < cols; ++x) { dst[y + stride * x] = std::conj(src[y * N + x]); }
      std::fill_n(&src[y * N], cols, std::complex<double>(0., 0.));
    }
  }
}

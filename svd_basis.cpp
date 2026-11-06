
#include <svd_basis.hpp>
#include <lapack_f2c.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>

inline void angemm(char transA, int M, int N, int K, int32_t sign, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, int32_t beta, std::complex<double>* C, int ldc)
{ char transB = 'N'; std::complex<double> one(1., 0.), zero(0., 0.), minus_one(-1., 0.); zgemm_(&transA, &transB, &M, &N, &K, sign ? &minus_one : &one, A, &lda, B, &ldb, beta ? &one : &zero, C, &ldc); }

inline void orth(int M, int N, int K, std::complex<double>* A, int lda) {
  int Lwork = -1, info; std::vector<std::complex<double>> tau(std::min(M, K)); std::complex<double> work_s0, work_s1;
  zgeqrf_(&M, &K, A, &lda, &tau[0], &work_s0, &Lwork, &info);
  zungqr_(&M, &N, &K, A, &lda, &tau[0], &work_s1, &Lwork, &info);
  Lwork = std::max(int(std::ceil(work_s0.real())), int(std::ceil(work_s1.real()))); 
  std::vector<std::complex<double>> work_a(Lwork);
  zgeqrf_(&M, &K, A, &lda, &tau[0], &work_a[0], &Lwork, &info);
  zungqr_(&M, &N, &K, A, &lda, &tau[0], &work_a[0], &Lwork, &info);
}

inline int svd_fit_left(double epi, int M, int N, std::complex<double>* A) {
  char jobu = 'O', jobvt = 'N'; int Lwork = -1, info = 0, n1 = 1; std::complex<double> work_s;
  std::vector<double> rwork(std::max(1, 5 * std::min(M, N))), S(std::min(M, N));
  zgesvd_(&jobu, &jobvt, &M, &N, A, &M, &S[0], nullptr, &n1, nullptr, &n1, &work_s, &Lwork, &rwork[0], &info);
  Lwork = int(std::ceil(work_s.real())); std::vector<std::complex<double>> work_a(Lwork);
  zgesvd_(&jobu, &jobvt, &M, &N, A, &M, &S[0], nullptr, &n1, nullptr, &n1, &work_a[0], &Lwork, &rwork[0], &info);

  std::transform_inclusive_scan(S.rbegin(), S.rend(), S.rbegin(), std::plus<double>(), [](double x) { return x * x; });
  if (S[0] < std::numeric_limits<double>::min()) { return 0; } else {
    double s0 = S[0] * std::pow(epi, 2);
    return std::distance(S.begin(), std::find_if(S.begin(), S.end(), [=](double s) { return s < s0; }));
  }
}

#include <hyacin.h>
hyacinHandle_t handle = {};
void svd_lib_init() { cudaSetDevice(0); hyacinCreate(&handle); }
void svd_lib_finalize() { hyacinDestroy(&handle); }

inline int svd_fit_left_gpu(double epi, int M, int N, std::complex<double>* A) {
  std::complex<double>* devA = nullptr; double* devS = nullptr; 
  cudaMalloc(&devA, (long long)M * (long long)N * sizeof(std::complex<double>));
  cudaMalloc(&devS, (long long)M * sizeof(double));
  cudaMemcpy(devA, A, (long long)M * (long long)N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
  int rank = hyacinTsvdFitSegmentN(&handle, epi, 'C', N, M, M, devA, M, devS, devA, M, N, HYACIN_F64_COMPLEX);
  cudaMemcpy(A, devA, (long long)M * (long long)rank * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  cudaFree(devA); cudaFree(devS);
  return rank;
}

long long computeBasis(double epi, long long requested_rank, char eco_u, long long M, long long N, long long K, std::complex<double>* U, std::complex<double>* A) {
  std::vector<std::complex<double>> B(M * (N + K));

  std::copy_n(A, M * N, &B[M * K]);
  if (0 < K) {
    std::vector<std::complex<double>> uB(K * N);
    angemm('N', K, N, M, 0, U, M, &B[M * K], M, 0, &uB[0], K);
    angemm('N', M, N, K, 1, U, M, &uB[0], K, 1, &B[M * K], M);
  }

  long long rank = svd_fit_left(epi, M, N, &B[M * K]);
  requested_rank = std::min(0 < requested_rank ? requested_rank : rank + K, M);

  int32_t econ = int32_t(eco_u == 'E' || eco_u == 'e');
  long long nU = econ ? requested_rank : M;
  std::copy_n(U, M * K, B.begin());
  orth(M, nU, std::min(nU, rank + K), &B[0], M);
  std::copy_n(&B[M * K], M * (nU - K), &U[M * K]);

  std::copy_n(&A[0], M * N, &B[0]);
  angemm('C', requested_rank, N, M, 0, U, M, &B[0], M, 0, A, requested_rank);
  return requested_rank;
}


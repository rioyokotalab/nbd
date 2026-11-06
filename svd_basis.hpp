#pragma once

#include <complex>

void svd_lib_init();
void svd_lib_finalize();

long long computeBasis(double epi, long long requested_rank, char eco_u, long long M, long long N, long long K, std::complex<double> U[], std::complex<double> A[]);


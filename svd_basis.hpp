#pragma once

#include <complex>

void svd_lib_init();
void svd_lib_finalize();

long long computeBasis(double epi, long long requested_rank, long long M, long long N, std::complex<double> U[], std::complex<double> A[]);


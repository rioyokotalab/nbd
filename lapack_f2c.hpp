#pragma once

#include <complex>

extern "C" void zgemv_(const char*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*) noexcept;
extern "C" void zgemm_(const char*, const char*, const int*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*) noexcept;

extern "C" void zlacpy_(const char*, const int*, const int*, const std::complex<double>*, const int*, std::complex<double>*, const int*) noexcept;
extern "C" void zlaset_(const char*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, std::complex<double>*, const int*) noexcept;
extern "C" void zgeqrf_(const int*, const int*, std::complex<double>*, const int*, std::complex<double>*, std::complex<double>*, const int*, int*) noexcept;
extern "C" void zungqr_(const int*, const int*, const int*, std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*, int*) noexcept;
extern "C" void zgesvd_(const char*, const char*, const int*, const int*, std::complex<double>*, const int*, double*, std::complex<double>*, const int*, std::complex<double>*, const int*, std::complex<double>*, const int*, double*, int*) noexcept;

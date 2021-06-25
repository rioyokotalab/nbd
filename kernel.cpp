
#include "kernel.h"
#include "build_tree.h"
#include "aca.h"

#include <cmath>

using namespace nbd;

eval_func_t nbd::r2() {
  return [](real_t& r2) -> void {};
}

eval_func_t nbd::l2d() {
  return [](real_t& r2) -> void {
    r2 = r2 == 0 ? 1.e5 : std::log(std::sqrt(r2));
  };
}

eval_func_t nbd::l3d() {
  return [](real_t& r2) -> void {
    r2 = r2 == 0 ? 1.e5 : 1. / std::sqrt(r2);
  };
}


void nbd::eval(eval_func_t r2f, const Body* bi, const Body* bj, int dim, real_t* out) {
  real_t& r2 = *out;
  r2 = 0.;
  for (int i = 0; i < dim; i++) {
    real_t dX = bi->X[i] - bj->X[i];
    r2 += dX * dX;
  }
  r2f(r2);
}


void nbd::mvec_kernel(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, const real_t* x_vec, real_t* b_vec) {
  int m = ci->NBODY, n = cj->NBODY;

  for (int y = 0; y < m; y++) {
    real_t sum = 0.;
    for (int x = 0; x < n; x++) {
      real_t r2;
      eval(r2f, ci->BODY + y, cj->BODY + x, dim, &r2);
      sum += r2 * x_vec[x];
    }
    b_vec[y] = sum;
  }
}


void nbd::P2Pnear(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, Matrix& a) {
  int m = ci->NBODY, n = cj->NBODY;
  a = Matrix(m, n, m);

  for (int i = 0; i < m * n; i++) {
    int x = i / m, y = i - x * m;
    real_t r2;
    eval(r2f, ci->BODY + y, cj->BODY + x, dim, &r2);
    a[(size_t)x * a.LDA + y] = r2;
  }
}

void nbd::P2Pfar(eval_func_t r2f, const Cell* ci, const Cell* cj, int dim, Matrix& a, int rank) {
  int m = ci->NBODY, n = cj->NBODY;
  a = Matrix(m, n, rank, m, n);

  int iters;
  daca_cells(r2f, ci, cj, dim, rank, a, a.LDA, a.B.data(), a.LDB, &iters);
  if (iters != rank) {
    a.A.resize((size_t)m * iters);
    a.B.resize((size_t)n * iters);
    a.R = iters;
  }
}

void nbd::SampleP2Pi(Matrix& s, const Matrix& a) {
  if (a.R > 0)
    drspl(s.M, a.N, a.R, a.A.data(), a.LDA, a.B.data(), a.LDB, s.N, s, s.LDA);
}

void nbd::SampleP2Pj(Matrix& s, const Matrix& a) {
  if (a.R > 0)
    drspl(s.M, a.M, a.R, a.B.data(), a.LDB, a.A.data(), a.LDA, s.N, s, s.LDA);
}

void nbd::SampleParent(Matrix& sc, const Matrix& sp, int c_off) {
  ddspl(sc.M, sp.N, sp.A.data() + c_off, sp.LDA, sc.N, sc, sc.LDA);
}

void nbd::BasisOrth(Matrix& s) {
  if (s.M && s.N)
    dorth(s.M, s.N, s, s.LDA);
}

void nbd::BasisInvLeft(const Matrix& s, Matrix& a) {
  if (a.R > 0) {
    int m = a.M, n = a.R, k = s.N;
    std::vector<real_t> b = a.A;

    a.A.resize((size_t)k * n);
    dmul_ut(m, n, k, s.A.data(), s.LDA, b.data(), a.LDA, a.A.data(), k);
    a.LDA = a.M = k;
  }
}

void nbd::BasisInvRight(const Matrix& s, Matrix& a) {
  if (a.R > 0) {
    int m = a.N, n = a.R, k = s.N;
    std::vector<real_t> b = a.B;

    a.B.resize((size_t)k * n);
    dmul_ut(m, n, k, s.A.data(), s.LDA, b.data(), a.LDB, a.B.data(), k);
    a.LDB = a.N = k;
  }
}

void nbd::BasisInvMultipleLeft(const Matrix* s, int ls, Matrix& a) {
  int m = a.M, n = a.N, k = 0;
  for (auto p = s; p != s + ls; p++)
    k += p->N;
  std::vector<real_t> b = a.A;

  a.A.resize((size_t)k * n);
  int off_a = 0, off_b = 0;
  for (auto p = s; p != s + ls; p++) {
    dmul_ut(p->M, n, p->N, p->A.data(), p->LDA, b.data() + off_b, a.LDA, a + off_a, k);
    off_a += p->N;
    off_b += p->M;
  }
  a.LDA = a.M = k;
}

void nbd::MergeS(Matrix& a) {
  if (a.R > 0) {
    int m = a.M, n = a.N, k = a.R;
    std::vector<real_t> ua = a.A;
    if (n != k)
      a.A.resize((size_t)a.LDA * n);
    dmul_s(m, n, k, ua.data(), a.LDA, a.B.data(), a.LDB, a, a.LDA);
    a.R = a.LDB = 0;
    a.B.clear();
  }
}

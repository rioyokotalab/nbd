
#include "build_tree.h"
#include "kernel.h"

#include <cmath>
#include <iterator>
#include <random>
#include <numeric>
#include <cstdio>

using namespace nbd;

Bodies::iterator spart(Bodies::iterator first, Bodies::iterator last, int sdim, real_t pivot) {

  auto l = [pivot, sdim](const Body& element) { return element.X[sdim] < pivot; };
  return std::partition(first, last, l);
}

Bodies::iterator spart_const_k(Bodies::iterator first, Bodies::iterator last, int sdim, int k) {

  for (int i = 0; i <= k; i++) {
    Bodies::iterator min = std::min_element(first + i, last, 
      [sdim](const Body& e1, const Body& e2) { return e1.X[sdim] < e2.X[sdim]; });
    if (min != first + i)
      std::iter_swap(min, first + i);
  }

  return first + k;
}

Bodies::iterator spart_size_k(Bodies::iterator first, Bodies::iterator last, int sdim, int k) {
  
  if (k < 10)
    return spart_const_k(first, last, sdim, k);
  real_t pivot = (first + std::distance(first, last) / 2)->X[sdim];

  Bodies::iterator i1 = std::partition(first, last,
    [pivot, sdim](const Body& element) { return element.X[sdim] < pivot; });
  Bodies::iterator i2 = std::partition(i1, last,
    [pivot, sdim](const Body& element) { return !(pivot < element.X[sdim]); });
  auto d1 = i1 - first, d2 = i2 - first;
  
  if (d1 > k)
    return spart_size_k(first, i1, sdim, k);
  else if (d2 <= k)
    return spart_size_k(i2, last, sdim, (int)(k - d2));
  else
    return first + k;
}

void spart_sdim(const real_t R[], int dim, int& sdim) {
  sdim = 0;
  real_t dmax = 0.;
  for (int d = 0; d < dim; d++) {
    if (R[d] > dmax)
    { dmax = R[d]; sdim = d; }
  }
}

void spart_median(real_t med, real_t& C1, real_t& C2, real_t& R1, real_t& R2) {
  real_t Xmin = C1 - R1;
  real_t Xmax = C1 + R1;
  C1 = (Xmin + med) / 2;
  C2 = (med + Xmax) / 2;
  R1 = (med - Xmin) / 2;
  R2 = (Xmax - med) / 2;
}


Cells nbd::buildTree(Bodies& bodies, int ncrit, int dim) {

  Cells cells(1);
  cells.reserve(bodies.size());

  cells[0].BODY = bodies.data();
  cells[0].NBODY = (int)bodies.size();
  cells[0].NCHILD = 0;

  real_t Xmin[nbd::dim], Xmax[nbd::dim];
  for (int d = 0; d < dim; d++) 
    Xmin[d] = Xmax[d] = bodies[0].X[d];
  for (auto& b : bodies) {
    for (int d = 0; d < dim; d++) 
      Xmin[d] = std::fmin(b.X[d], Xmin[d]);
    for (int d = 0; d < dim; d++) 
      Xmax[d] = std::fmax(b.X[d], Xmax[d]);
  }


  for (int d = 0; d < dim; d++) {
    cells[0].C[d] = (Xmin[d] + Xmax[d]) / 2;
    cells[0].R[d] = std::fabs(Xmin[d] - Xmax[d]) / 2;
  }

  int nlis = ((int)bodies.size() + ncrit - 1) / ncrit, iters = 0;
  while (nlis >>= 1) ++iters;
  int last_off = 0, last_len = 1;

  for (int i = 1; i <= iters; i++) {

    int len = 0;

    for (int j = last_off; j < last_off + last_len; j++) {
      Cell& cell = cells[j];
      Bodies::iterator cell_b = bodies.begin() + std::distance(bodies.data(), cell.BODY);

      int sdim;
      spart_sdim(cell.R, dim, sdim);

#ifdef PART_EQ_SIZE
      auto p = spart_size_k(cell_b, cell_b + cell.NBODY, sdim, cell.NBODY / 2);
      real_t med = p->X[sdim];
#else
      real_t med = cell.C[sdim];
      auto p = spart(cell_b, cell_b + cell.NBODY, sdim, med);
#endif
      int size[2];
      size[0] = (int)(p - cell_b);
      size[1] = cell.NBODY - size[0];

      cell.NCHILD = (int)(size[0] > 0) + (int)(size[1] > 0);
      cells.resize(cells.size() + cell.NCHILD);
      Cell* child = &cells.back() - cell.NCHILD + 1;
      cell.CHILD = child;

      int offset = 0;
      for (int k = 0; k < 2; k++) {
        if (size[k]) {
          child->BODY = cell.BODY + offset;
          child->NBODY = size[k];
          child->NCHILD = 0;
          for (int d = 0; d < dim; d++) {
            child->C[d] = cell.C[d];
            child->R[d] = cell.R[d];
          }
          child = child + 1;
          offset += size[k];
        }
      }

      if (cell.NCHILD == 2)
        spart_median(med, cell.CHILD[0].C[sdim], cell.CHILD[1].C[sdim], cell.CHILD[0].R[sdim], cell.CHILD[1].R[sdim]);

      len += cell.NCHILD;
    }

    last_off += last_len;
    last_len = len;
  }

  return cells;
}


void nbd::getList(Cell * Ci, Cell * Cj, int dim, real_t theta, bool symm) {
  real_t dX = 0., CiR = 0., CjR = 0.;
  for (int d = 0; d < dim; d++) {
    real_t diff = Ci->C[d] - Cj->C[d];
    dX += diff * diff;

    CiR = std::fmax(CiR, Ci->R[d]);
    CjR = std::fmax(CjR, Cj->R[d]);
  }
  real_t R2 = dX * theta * theta;

  if (R2 > (CiR + CjR) * (CiR + CjR)) {
    Ci->listFar.push_back(Cj);
    if (!symm)
      Cj->listFar.push_back(Ci);
  } else if (Ci->NCHILD == 0 && Cj->NCHILD == 0) {
    Ci->listNear.push_back(Cj);
    if (!symm)
      Cj->listNear.push_back(Ci);
  } else { 
    if (Cj->NCHILD == 0 || (CiR >= CjR && Ci->NCHILD != 0))
      for (Cell * ci=Ci->CHILD; ci!=Ci->CHILD+Ci->NCHILD; ci++)
        getList(ci, Cj, dim, theta, symm);
    else
      for (Cell * cj=Cj->CHILD; cj!=Cj->CHILD+Cj->NCHILD; cj++)
        getList(Ci, cj, dim, theta, symm);
  }
}

Matrices nbd::evaluate(EvalFunc ef, const Cells& icells, const Cells& jcells, int dim, int rank, bool eval_near) {
  Matrices d(icells.size() * jcells.size());
#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    if (rank > 0)
      for (auto& j : i.listFar) {
        auto x = j - &jcells[0];
        P2Pfar(ef, &i, j, dim, d[y + x * icells.size()], rank);
      }
    if (eval_near)
      for (auto& j : i.listNear) {
        auto x = j - &jcells[0];
        P2Pnear(ef, &i, j, dim, d[y + x * icells.size()]);
      }
  }
  return d;
}

Matrices nbd::traverse(EvalFunc ef, Cells& icells, Cells& jcells, int dim, real_t theta, int rank, bool eval_near) {
  getList(&icells[0], &jcells[0], dim, theta, &icells == &jcells);
  return evaluate(ef, icells, jcells, dim, rank, eval_near);
}


Matrices nbd::sample_base_i(const Cells& icells, const Cells& jcells, Matrices& d, int p) {
  Matrices base(icells.size());
  int ld = (int)icells.size();
#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    int r = 0;
    for (auto& j : i.listFar) {
      auto x = j - jcells.data();
      Matrix& m = d[y + (size_t)x * ld];
      int rm = m.A.size() / (m.M + m.N);
      r = std::max(r, rm);
    }

    if (r > 0) {
      int osp = std::min(i.NBODY, r + p);
      base[y].M = i.NBODY;
      base[y].N = osp;
      base[y].A.resize((size_t)i.NBODY * osp);
      std::fill(base[y].A.begin(), base[y].A.end(), 0);
    }

    for (auto& j : i.listFar) {
      auto x = j - jcells.data();
      Matrix& m = d[y + (size_t)x * ld];
      SampleP2Pi(base[y], m);
    }
  }

  return base;
}

Matrices nbd::sample_base_j(const Cells& icells, const Cells& jcells, Matrices& d, int p) {
  Matrices base(jcells.size());
  int ld = (int)icells.size();
#pragma omp parallel for
  for (int x = 0; x < jcells.size(); x++) {
    auto j = jcells[x];
    int r = 0;
    for (auto& i : j.listFar) {
      auto y = i - icells.data();
      Matrix& m = d[y + (size_t)x * ld];
      int rm = m.A.size() / (m.M + m.N);
      r = std::max(r, rm);
    }

    if (r > 0) {
      int osp = std::min(j.NBODY, r + p);
      base[x].M = j.NBODY;
      base[x].N = osp;
      base[x].A.resize((size_t)j.NBODY * osp);
      std::fill(base[x].A.begin(), base[x].A.end(), 0);
    }

    for (auto& i : j.listFar) {
      auto y = i - icells.data();
      Matrix& m = d[y + (size_t)x * ld];
      SampleP2Pj(base[x], m);
    }
  }

  return base;
}


void nbd::sample_base_recur(Cell* cell, Matrix* base) {

  int c_off = 0;
  for (Cell* c = cell->CHILD; c != cell->CHILD + cell->NCHILD; c++) {
    auto i = c - cell;
    if (base[i].N == 0 && base->N > 0) {
      int r = std::min(c->NBODY, base->N);
      base[i].M = c->NBODY;
      base[i].N = r;
      base[i].A.resize((size_t)c->NBODY * r);
      std::fill(base[i].A.begin(), base[i].A.end(), 0);
    }
    if (base->N > 0)
      SampleParent(base[i], *base, c_off);
    sample_base_recur(c, base + i);
    c_off += c->NBODY;
  }

}

void nbd::orth_base(Matrices& base) {
#pragma omp parallel for
  for (int x = 0; x < base.size(); x++) {
    Matrix r;
    BasisOrth(base[x], r);
  }
}

void nbd::shared_base_i(const Cells& icells, const Cells& jcells, Matrices& d, Matrices& base) {
  int ld = (int)icells.size();
  
  if (&icells == &jcells)
#pragma omp parallel for
    for (int y = 0; y < icells.size(); y++) {
      auto i = icells[y];
      for (auto& j : i.listFar) {
        auto x = j - jcells.data();
        Matrix& m = d[y + (size_t)x * ld];
        BasisInvRightAndMerge(base[x], m);
      }
    }

#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    for (auto& j : i.listFar) {
      auto x = j - jcells.data();
      Matrix& m = d[y + (size_t)x * ld];
      BasisInvLeft(base[y], m);
    }
  }

}

void nbd::shared_base_j(const Cells& icells, const Cells& jcells, Matrices& d, Matrices& base) {
  int ld = (int)icells.size();
#pragma omp parallel for
  for (int x = 0; x < jcells.size(); x++) {
    auto j = jcells[x];
    for (auto& i : j.listFar) {
      auto y = i - icells.data();
      Matrix& m = d[y + (size_t)x * ld];
      BasisInvRightAndMerge(base[x], m);
    }
  }
}

void nbd::nest_base(const Cell* icell, Matrix* base) {
  if (icell->NCHILD == 0)
    return;

  if (base->N > 0) {
    Matrix* m = base + (icell->CHILD - icell);
    BasisInvMultipleLeft(m, icell->NCHILD, *base);
  }

  for (Cell* c = icell->CHILD; c != icell->CHILD + icell->NCHILD; c++) {
    Matrix* m = base + (c - icell);
    nest_base(c, m);
  }
}

Matrices nbd::traverse_i(Cells& icells, Cells& jcells, Matrices& d, int p) {
  Matrices base = sample_base_i(icells, jcells, d, p);
  sample_base_recur(&icells[0], &base[0]);
  orth_base(base);
  return base;
}

Matrices nbd::traverse_j(Cells& icells, Cells& jcells, Matrices& d, int p) {
  Matrices base = sample_base_j(icells, jcells, d, p);
  sample_base_recur(&jcells[0], &base[0]);
  orth_base(base);
  return base;
}

void nbd::traverse_b(const Cells& icells, const Cells& jcells, Matrices& ibase, Matrices& jbase, Matrices& d) {
  if (&icells != &jcells) {
    shared_base_j(icells, jcells, d, jbase);
    nest_base(&jcells[0], &jbase[0]);
  }
  shared_base_i(icells, jcells, d, ibase);
  nest_base(&icells[0], &ibase[0]);
}


Cells nbd::getLeaves(const Cells& cells) {
  Cells l;
  l.emplace_back(cells[0]);

  for (const auto& c : cells)
    if (c.NCHILD == 0) {
      l.emplace_back(c);
      auto& cl = l.back();
      cl.listFar.clear();
      cl.listNear.clear();
    }

  l[0].NCHILD = l.size() - 1;
  l[0].CHILD = l[0].NCHILD ? &l[1] : nullptr;
  return l;
}

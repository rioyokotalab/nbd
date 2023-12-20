#include "nbd.hxx"

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"

#include <algorithm>
#include <array>

void get_bounds(const double* bodies, int64_t nbodies, double R[], double C[]) {
  double Xmin[3];
  double Xmax[3];
  Xmin[0] = Xmax[0] = bodies[0];
  Xmin[1] = Xmax[1] = bodies[1];
  Xmin[2] = Xmax[2] = bodies[2];

  for (int64_t i = 1; i < nbodies; i++) {
    const double* x_bi = &bodies[i * 3];
    Xmin[0] = fmin(x_bi[0], Xmin[0]);
    Xmin[1] = fmin(x_bi[1], Xmin[1]);
    Xmin[2] = fmin(x_bi[2], Xmin[2]);

    Xmax[0] = fmax(x_bi[0], Xmax[0]);
    Xmax[1] = fmax(x_bi[1], Xmax[1]);
    Xmax[2] = fmax(x_bi[2], Xmax[2]);
  }

  C[0] = (Xmin[0] + Xmax[0]) / 2.;
  C[1] = (Xmin[1] + Xmax[1]) / 2.;
  C[2] = (Xmin[2] + Xmax[2]) / 2.;

  double d0 = Xmax[0] - Xmin[0];
  double d1 = Xmax[1] - Xmin[1];
  double d2 = Xmax[2] - Xmin[2];

  R[0] = (d0 == 0. && Xmin[0] == 0.) ? 0. : (1.e-8 + d0 / 2.);
  R[1] = (d1 == 0. && Xmin[1] == 0.) ? 0. : (1.e-8 + d1 / 2.);
  R[2] = (d2 == 0. && Xmin[2] == 0.) ? 0. : (1.e-8 + d2 / 2.);
}

void sort_bodies(double* bodies, int64_t nbodies, int64_t sdim) {
  std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(bodies);
  std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[3 * nbodies]);
  std::sort(bodies3, bodies3_end, [=](std::array<double, 3>& i, std::array<double, 3>& j)->bool {
    double x = i[sdim];
    double y = j[sdim];
    return x < y;
  });
}

void buildTree(int64_t* ncells, struct Cell* cells, double* bodies, int64_t nbodies, int64_t levels) {
  struct Cell* root = &cells[0];
  root->Body[0] = 0;
  root->Body[1] = nbodies;
  root->Level = 0;
  get_bounds(bodies, nbodies, root->R, root->C);

  int64_t len = 1;
  int64_t i = 0;
  while (i < len) {
    struct Cell* ci = &cells[i];
    ci->Child[0] = -1;
    ci->Child[1] = -1;

    if (ci->Level < levels) {
      int64_t sdim = 0;
      double maxR = ci->R[0];
      if (ci->R[1] > maxR)
      { sdim = 1; maxR = ci->R[1]; }
      if (ci->R[2] > maxR)
      { sdim = 2; maxR = ci->R[2]; }

      int64_t i_begin = ci->Body[0];
      int64_t i_end = ci->Body[1];
      int64_t nbody_i = i_end - i_begin;
      sort_bodies(&bodies[i_begin * 3], nbody_i, sdim);
      int64_t loc = i_begin + nbody_i / 2;

      struct Cell* c0 = &cells[len];
      struct Cell* c1 = &cells[len + 1];
      ci->Child[0] = len;
      ci->Child[1] = len + 2;
      len = len + 2;

      c0->Body[0] = i_begin;
      c0->Body[1] = loc;
      c1->Body[0] = loc;
      c1->Body[1] = i_end;
      
      c0->Level = ci->Level + 1;
      c1->Level = ci->Level + 1;

      get_bounds(&bodies[i_begin * 3], loc - i_begin, c0->R, c0->C);
      get_bounds(&bodies[loc * 3], i_end - loc, c1->R, c1->C);
    }
    i++;
  }
  *ncells = len;
}

void buildTreeBuckets(struct Cell* cells, const double* bodies, const int64_t buckets[], int64_t levels) {
  int64_t nleaf = (int64_t)1 << levels;
  int64_t count = 0;
  for (int64_t i = 0; i < nleaf; i++) {
    int64_t ci = i + nleaf - 1;
    cells[ci].Child[0] = -1;
    cells[ci].Child[1] = -1;
    cells[ci].Body[0] = count;
    cells[ci].Body[1] = count + buckets[i];
    cells[ci].Level = levels;
    get_bounds(&bodies[count * 3], buckets[i], cells[ci].R, cells[ci].C);
    count = count + buckets[i];
  }

  for (int64_t i = nleaf - 2; i >= 0; i--) {
    int64_t c0 = (i << 1) + 1;
    int64_t c1 = (i << 1) + 2;
    int64_t begin = cells[c0].Body[0];
    int64_t len = cells[c1].Body[1] - begin;
    cells[i].Child[0] = c0;
    cells[i].Child[1] = c0 + 2;
    cells[i].Body[0] = begin;
    cells[i].Body[1] = begin + len;
    cells[i].Level = cells[c0].Level - 1;
    get_bounds(&bodies[begin * 3], len, cells[i].R, cells[i].C);
  }
}

int admis_check(double theta, const double C1[], const double C2[], const double R1[], const double R2[]) {
  double dCi[3];
  dCi[0] = C1[0] - C2[0];
  dCi[1] = C1[1] - C2[1];
  dCi[2] = C1[2] - C2[2];

  dCi[0] = dCi[0] * dCi[0];
  dCi[1] = dCi[1] * dCi[1];
  dCi[2] = dCi[2] * dCi[2];

  double dRi[3];
  dRi[0] = R1[0] * R1[0];
  dRi[1] = R1[1] * R1[1];
  dRi[2] = R1[2] * R1[2];

  double dRj[3];
  dRj[0] = R2[0] * R2[0];
  dRj[1] = R2[1] * R2[1];
  dRj[2] = R2[2] * R2[2];

  double dC = dCi[0] + dCi[1] + dCi[2];
  double dR = (dRi[0] + dRi[1] + dRi[2] + dRj[0] + dRj[1] + dRj[2]) * theta;
  return (int)(dC > dR);
}

void getList(char NoF, int64_t* len, int64_t rels[], int64_t ncells, const struct Cell cells[], int64_t i, int64_t j, double theta) {
  const struct Cell* Ci = &cells[i];
  const struct Cell* Cj = &cells[j];
  int64_t ilevel = Ci->Level;
  int64_t jlevel = Cj->Level; 
  if (ilevel == jlevel) {
    int admis = admis_check(theta, Ci->C, Cj->C, Ci->R, Cj->R);
    int write_far = NoF == 'F' || NoF == 'f';
    int write_near = NoF == 'N' || NoF == 'n';
    if (admis ? write_far : write_near) {
      int64_t n = *len;
      rels[n] = i + j * ncells;
      *len = n + 1;
    }
    if (admis)
      return;
  }
  if (ilevel <= jlevel && Ci->Child[0] >= 0)
    for (int64_t k = Ci->Child[0]; k < Ci->Child[1]; k++)
      getList(NoF, len, rels, ncells, cells, k, j, theta);
  else if (jlevel <= ilevel && Cj->Child[0] >= 0)
    for (int64_t k = Cj->Child[0]; k < Cj->Child[1]; k++)
      getList(NoF, len, rels, ncells, cells, i, k, theta);
}

void traverse(char NoF, struct CSC* rels, int64_t ncells, const struct Cell* cells, double theta) {
  rels->M = ncells;
  rels->N = ncells;
  int64_t* rel_arr = (int64_t*)malloc(sizeof(int64_t) * (ncells * ncells + ncells + 1));
  int64_t len = 0;
  getList(NoF, &len, &rel_arr[ncells + 1], ncells, cells, 0, 0, theta);

  if (len < ncells * ncells)
    rel_arr = (int64_t*)realloc(rel_arr, sizeof(int64_t) * (len + ncells + 1));
  int64_t* rel_rows = &rel_arr[ncells + 1];
  std::sort(rel_rows, rel_rows + len);
  rels->ColIndex = rel_arr;
  rels->RowIndex = rel_rows;

  int64_t loc = -1;
  for (int64_t i = 0; i < len; i++) {
    int64_t r = rel_rows[i];
    int64_t x = r / ncells;
    int64_t y = r - x * ncells;
    rel_rows[i] = y;
    while (x > loc)
      rel_arr[++loc] = i;
  }
  for (int64_t i = loc + 1; i <= ncells; i++)
    rel_arr[i] = len;
}

void csc_free(struct CSC* csc) {
  free(csc->ColIndex);
}

void lookupIJ(int64_t* ij, const struct CSC* rels, int64_t i, int64_t j) {
  if (j < 0 || j >= rels->N)
  { *ij = -1; return; }
  const int64_t* row = rels->RowIndex;
  int64_t jbegin = rels->ColIndex[j];
  int64_t jend = rels->ColIndex[j + 1];
  const int64_t* row_iter = &row[jbegin];
  while (row_iter != &row[jend] && *row_iter != i)
    row_iter = row_iter + 1;
  int64_t k = row_iter - row;
  *ij = (k < jend) ? k : -1;
}

void countMaxIJ(int64_t* max_i, int64_t* max_j, const struct CSC* rels) {
  std::vector<int64_t> countx(rels->N, 0), county(rels->M, 0);
  for (int64_t x = 0; x < rels->N; x++)
    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      countx[x] = countx[x] + 1;
      county[y] = county[y] + 1;
    }
  if (max_i)
    *max_i = *std::max_element(county.begin(), county.end());
  if (max_j)
    *max_j = *std::max_element(countx.begin(), countx.end());
}

void loadX(double* X, int64_t seg, const double Xbodies[], int64_t Xbegin, int64_t ncells, const struct Cell cells[]) {
  for (int64_t i = 0; i < ncells; i++) {
    int64_t b0 = cells[i].Body[0] - Xbegin;
    int64_t lenB = cells[i].Body[1] - cells[i].Body[0];
    for (int64_t j = 0; j < lenB; j++)
      X[i * seg + j] = Xbodies[j + b0];
  }
}

void evalD(const EvalDouble& eval, struct Matrix* D, const struct CSC* rels, const struct Cell* cells, const double* bodies, const struct CellComm* comm) {
  int64_t ibegin = 0, nodes = 0;
  content_length(&nodes, NULL, &ibegin, comm);
  ibegin = comm->iGlobal(ibegin);

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    int64_t lc = ibegin + i;
    const struct Cell* ci = &cells[lc];
    int64_t nbegin = rels->ColIndex[lc];
    int64_t nlen = rels->ColIndex[lc + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];
    int64_t x_begin = ci->Body[0];
    int64_t n = ci->Body[1] - x_begin;
    int64_t offsetD = nbegin - rels->ColIndex[ibegin];

    for (int64_t j = 0; j < nlen; j++) {
      int64_t lj = ngbs[j];
      const struct Cell* cj = &cells[lj];
      int64_t y_begin = cj->Body[0];
      int64_t m = cj->Body[1] - y_begin;
      gen_matrix(eval, n, m, &bodies[x_begin * 3], &bodies[y_begin * 3], D[offsetD + j].A, D[offsetD + j].LDA);
    }
  }
}

void evalS(const EvalDouble& eval, struct Matrix* S, const struct Base* basis, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0;
  content_length(NULL, NULL, &ibegin, comm);
  int64_t seg = basis->dimS * 3;

#pragma omp parallel for
  for (int64_t x = 0; x < rels->N; x++) {
    int64_t n = basis->DimsLr[x + ibegin];

    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t m = basis->DimsLr[y];
      gen_matrix(eval, n, m, &basis->M_cpu[(x + ibegin) * seg], &basis->M_cpu[y * seg], S[yx].A, S[yx].LDA);
      mul_AS(&basis->R[x + ibegin], &basis->R[y], &S[yx]);
    }
  }
}

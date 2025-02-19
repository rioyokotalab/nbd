
#include <build_tree.hpp>
#include <linalg.hpp>
#include <comm.hpp>
#include <basis.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

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

void buildTree(int64_t* ncells, Cell* cells, double* bodies, int64_t nbodies, int64_t levels) {
  Cell* root = &cells[0];
  root->Body[0] = 0;
  root->Body[1] = nbodies;
  root->Level = 0;
  get_bounds(bodies, nbodies, root->R, root->C);

  int64_t len = 1;
  int64_t i = 0;
  while (i < len) {
    Cell* ci = &cells[i];
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

      Cell* c0 = &cells[len];
      Cell* c1 = &cells[len + 1];
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

void getList(char NoF, int64_t* len, int64_t rels[], int64_t ncells, const Cell cells[], int64_t i, int64_t j, double theta) {
  const Cell* Ci = &cells[i];
  const Cell* Cj = &cells[j];
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

void traverse(char NoF, CSR* rels, int64_t ncells, const Cell* cells, double theta) {
  rels->M = ncells;
  rels->N = ncells;
  std::vector<int64_t> rel_arr(ncells * ncells);
  int64_t len = 0;
  getList(NoF, &len, &rel_arr[0], ncells, cells, 0, 0, theta);
  std::sort(rel_arr.begin(), rel_arr.begin() + len);

  rels->RowIndex.resize(ncells + 1);
  rels->ColIndex.resize(len);

  int64_t loc = -1;
  for (int64_t i = 0; i < len; i++) {
    int64_t r = rel_arr[i];
    int64_t x = r / ncells;
    int64_t y = r - x * ncells;
    rels->ColIndex[i] = y;
    while (x > loc)
      rels->RowIndex[++loc] = i;
  }
  for (int64_t i = loc + 1; i <= ncells; i++)
    rels->RowIndex[i] = len;
}

void countMaxIJ(int64_t* max_i, int64_t* max_j, const CSR* rels) {
  std::vector<int64_t> countx(rels->N, 0), county(rels->M, 0);
  for (int64_t x = 0; x < rels->N; x++)
    for (int64_t yx = rels->RowIndex[x]; yx < rels->RowIndex[x + 1]; yx++) {
      int64_t y = rels->ColIndex[yx];
      countx[x] = countx[x] + 1;
      county[y] = county[y] + 1;
    }
  if (max_i)
    *max_i = *std::max_element(county.begin(), county.end());
  if (max_j)
    *max_j = *std::max_element(countx.begin(), countx.end());
}

void loadX(double* X, int64_t seg, const double Xbodies[], int64_t Xbegin, int64_t ncells, const Cell cells[]) {
  for (int64_t i = 0; i < ncells; i++) {
    int64_t b0 = cells[i].Body[0] - Xbegin;
    int64_t lenB = cells[i].Body[1] - cells[i].Body[0];
    for (int64_t j = 0; j < lenB; j++)
      X[i * seg + j] = Xbodies[j + b0];
  }
}

void evalD(const EvalDouble& eval, Matrix* D, const CSR* rels, const Cell* cells, const double* bodies, const CellComm* comm) {
  int64_t ibegin = 0, nodes = 0;
  content_length(&nodes, NULL, &ibegin, comm);
  ibegin = comm->iGlobal(ibegin);

  for (int64_t i = 0; i < nodes; i++) {
    int64_t lc = ibegin + i;
    const Cell* ci = &cells[lc];
    int64_t nbegin = rels->RowIndex[lc];
    int64_t nlen = rels->RowIndex[lc + 1] - nbegin;
    const int64_t* ngbs = &rels->ColIndex[nbegin];
    int64_t x_begin = ci->Body[0];
    int64_t n = ci->Body[1] - x_begin;
    int64_t offsetD = nbegin - rels->RowIndex[ibegin];

    for (int64_t j = 0; j < nlen; j++) {
      int64_t lj = ngbs[j];
      const Cell* cj = &cells[lj];
      int64_t y_begin = cj->Body[0];
      int64_t m = cj->Body[1] - y_begin;
      gen_matrix(eval, n, m, &bodies[x_begin * 3], &bodies[y_begin * 3], D[offsetD + j].A, D[offsetD + j].LDA);
    }
  }
}

void evalS(const EvalDouble& eval, Matrix* S, const Base* basis, const CSR* rels, const CellComm* comm) {
  int64_t ibegin = 0;
  content_length(NULL, NULL, &ibegin, comm);
  int64_t seg = basis->dimS * 3;

  for (int64_t x = 0; x < rels->N; x++) {
    int64_t n = basis->DimsLr[x + ibegin];

    for (int64_t yx = rels->RowIndex[x]; yx < rels->RowIndex[x + 1]; yx++) {
      int64_t y = rels->ColIndex[yx];
      int64_t m = basis->DimsLr[y];
      gen_matrix(eval, n, m, &basis->M[(x + ibegin) * seg], &basis->M[y * seg], S[yx].A, S[yx].LDA);
      mul_AS(&basis->R[x + ibegin], &basis->R[y], &S[yx]);
    }
  }
}

#pragma once

#include <vector>
#include <cstdint>

class Cell {
public:
  int64_t Child[2], Body[2], Level;
  double R[3], C[3];
};

class CSR {
public:
  int64_t M, N;
  std::vector<int64_t> RowIndex;
  std::vector<int64_t> ColIndex;

  CSR() : M(0), N(0), RowIndex(), ColIndex() {}
  CSR(const CSR& A, const CSR& B);

  int64_t lookupIJ(int64_t i, int64_t j) const {
    if (j < 0 || j >= N)
    { return -1; }
    const int64_t* row = &ColIndex[0];
    int64_t jbegin = RowIndex[j];
    int64_t jend = RowIndex[j + 1];
    const int64_t* row_iter = &row[jbegin];
    while (row_iter != &row[jend] && *row_iter != i)
      row_iter = row_iter + 1;
    int64_t k = row_iter - row;
    return (k < jend) ? k : -1;
  }
};

class EvalDouble;
class Matrix;
class Base;
class ColCommMPI;

void buildTree(int64_t* ncells, Cell* cells, double* bodies, int64_t nbodies, int64_t levels);

void traverse(char NoF, CSR* rels, int64_t ncells, const Cell* cells, double theta);

void countMaxIJ(int64_t* max_i, int64_t* max_j, const CSR* rels);

void loadX(double* X, int64_t seg, const double Xbodies[], int64_t Xbegin, int64_t ncells, const Cell cells[]);

void evalD(const EvalDouble& eval, Matrix* D, const CSR* rels, const Cell* cells, const double* bodies, const ColCommMPI* comm);

void evalS(const EvalDouble& eval, Matrix* S, const Base* basis, const CSR* rels, const ColCommMPI* comm);

void relations(CSR rels[], const CSR* cellRel, int64_t levels, const ColCommMPI* comm);

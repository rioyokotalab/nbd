
#pragma once

#include <cstdint>
#include <vector>

class CSR {
public:
  int64_t M, N;
  std::vector<int64_t> RowIndex;
  std::vector<int64_t> ColIndex;

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

#pragma once

#include <vector>
#include <array>

class Cell;

class CSR {
public:
  long long N;
  long long DiagCol;
  std::vector<long long> RowIndex;
  std::vector<long long> ColIndex;
  std::vector<long long> DiagIndex;

  CSR() : DiagCol(0) {}
  CSR(char NoF, const std::vector<Cell>& ci, const std::vector<Cell>& cj);
  CSR(const CSR& global, long long lbegin, long long lend, const std::vector<std::array<long long, 3>>& boxes);
};

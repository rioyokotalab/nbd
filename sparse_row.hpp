
#pragma once

#include <cstdint>
#include <vector>

class CSR {
public:
  int64_t M, N;
  std::vector<int64_t> RowIndex;
  std::vector<int64_t> ColIndex;
};

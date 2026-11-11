
#include <csr.hpp>
#include <build_tree.hpp>

#include <algorithm>
#include <numeric>
#include <map>

void getList(std::vector<std::pair<long long, long long>>& rels, const Cell ci[], long long i, const Cell cj[], long long j) {
  std::array<double, 3> dc({ ci[i].C[0] - cj[j].C[0], ci[i].C[1] - cj[j].C[1], ci[i].C[2] - cj[j].C[2] });
  bool admis = 0. < std::inner_product(dc.begin(), dc.end(), dc.begin(), 0.);
  if (admis) { rels.emplace_back(i, j); }
  
  if (!admis && ci[i].Child[0] >= 0 && cj[j].Child[0] >= 0)
    for (long long k = ci[i].Child[0]; k < ci[i].Child[1]; k++)
      for (long long l = cj[j].Child[0]; l < cj[j].Child[1]; l++)
        getList(rels, ci, k, cj, l);
}

CSR::CSR(const std::vector<Cell>& ci, const std::vector<Cell>& cj) : N(ci.size()) {
  std::vector<std::pair<long long, long long>> LIL;
  getList(LIL, &ci[0], 0, &cj[0], 0);
  std::sort(LIL.begin(), LIL.end());

  long long len = LIL.size();
  RowIndex.resize(N + 1);
  ColIndex.resize(len);

  RowIndex[0] = 0;
  std::transform(LIL.begin(), LIL.end(), ColIndex.begin(), [](const std::pair<long long, long long>& i) { return i.second; });
  std::for_each(LIL.begin(), LIL.end(), [&](const std::pair<long long, long long>& i) { ++RowIndex[i.first + 1]; });
  std::inclusive_scan(RowIndex.begin() + 1, RowIndex.end(), RowIndex.begin() + 1);
}

CSR::CSR(const CSR& global, long long lbegin, long long lend, const std::vector<std::array<long long, 3>>& boxes) : N(boxes.back()[2]) {
  std::map<long long, long long> local_map;
  for (const std::array<long long, 3>& box : boxes)
    for (long long i = 0; i < box[1]; ++i)
      local_map.emplace(box[0] + i, box[2] - box[1] + i);

  long long offset = global.RowIndex[lbegin];
  RowIndex.resize(lend - lbegin + 1);
  ColIndex.resize(global.RowIndex[lend] - global.RowIndex[lbegin]);

  std::transform(&(global.RowIndex)[lbegin], &(global.RowIndex)[lend + 1], RowIndex.begin(), [=](long long i) { return i - offset; });
  std::transform(&(global.ColIndex)[global.RowIndex[lbegin]], &(global.ColIndex)[global.RowIndex[lend]], ColIndex.begin(), [&](long long j) { return local_map[j]; });
}

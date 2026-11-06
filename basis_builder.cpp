
#include <basis_builder.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <csr.hpp>
#include <dense.hpp>
#include <svd_basis.hpp>
#include <lapack_f2c.hpp>

#include <algorithm>
#include <numeric>

inline void copy2d(int M, int N, const std::complex<double>* A, int lda, std::complex<double>* B, int ldb)
{ char uplo = 'A'; zlacpy_(&uplo, &M, &N, A, &lda, B, &ldb); }
inline void nngemm(int M, int N, int K, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, std::complex<double>* C, int ldc)
{ char transA = 'N'; std::complex<double> one(1., 0.), zero(0., 0.); zgemm_(&transA, &transA, &M, &N, &K, &one, A, &lda, B, &ldb, &zero, C, &ldc); }

BasisBuilder::BasisBuilder(long long lbegin, long long len, const Cell cells[], const CSR& Far) : lbegin(lbegin), lend(lbegin + len), Ncur(len, 0ll), N(len, 0ll), M(len), Ranks(len, 0ll), fcells(len), fmatrix(len), basis(len), basis_full(len) {
  for (long long y = lbegin; y < lend; ++y) {
    long long i = y - lbegin;
    M[i] = cells[y].Body[1] - cells[y].Body[0];
    
    for (long long yx = Far.RowIndex[y]; yx < Far.RowIndex[y + 1]; yx++) {
      long long x = Far.ColIndex[yx];
      long long n = cells[x].Body[1] - cells[x].Body[0];
      Ncur[i] += n;
      N[i] += n;
      if (0 < fcells[i].size() && fcells[i].back().second == cells[x].Body[0])
        fcells[i].back().second = cells[x].Body[1];
      else
        fcells[i].emplace_back(cells[x].Body[0], cells[x].Body[1]);
    }
  }
}

BasisBuilder::BasisBuilder(long long lbegin, long long len, const Cell cells[], const CSR& Far, const BasisBuilder& upper) : BasisBuilder(lbegin, len, cells, Far) {
  for (long long i = upper.lbegin; i < upper.lend; ++i)
    for (long long y = std::max(lbegin, cells[i].Child[0]); y < std::min(lend, cells[i].Child[1]); ++y) {
      N[y - lbegin] += upper.N[i - upper.lbegin];
      fcells[y - lbegin].insert(fcells[y - lbegin].end(), upper.fcells[i - upper.lbegin].begin(), upper.fcells[i - upper.lbegin].end());
    }
}

void BasisBuilder::evalLeafFMatrix(double epi, const DenseMatrix& matA, const Cell cells[]) {
  for (long long y = lbegin; y < lend; ++y) {
    long long i = y - lbegin;
    long long mA = M[i];
    if (0 < mA && 0 < N[i]) {
      long long iA = cells[y].Body[0];
      fmatrix[i].resize(mA * N[i]);
      basis[i].resize(mA * mA);

      long long j = 0;
      for (std::pair<long long, long long> x : fcells[i]) {
        long long jA = x.first;
        long long nA = x.second - jA;
        matA.Aij_block(mA, nA, iA, jA, &(fmatrix[i])[mA * j], mA);
        j += nA;
      }

      Ranks[i] = computeBasis(epi, Ranks[i], 'E', mA, N[i], 0, basis[i].data(), fmatrix[i].data());
      fmatrix[i].resize(Ranks[i] * N[i]); fmatrix[i].shrink_to_fit();
      basis_full[i].resize(mA * Ranks[i]);
      basis[i].resize(basis_full[i].size()); basis[i].shrink_to_fit();
      std::copy_n(basis[i].begin(), basis_full[i].size(), basis_full[i].begin());
    }
  }
}

void BasisBuilder::evalInnerFMatrix(double epi, const BasisBuilder& lower, const Cell cells[], const H2MatrixComm& comm) {
  for (long long y = lbegin; y < lend; ++y) {
    long long c_start = std::max(lower.lbegin, cells[y].Child[0]);
    long long c_end = std::min(lower.lend, cells[y].Child[1]);
    std::vector<long long> ranksOffsets(cells[y].Child[1] - cells[y].Child[0] + 1, 0ll);
    std::copy_n(lower.Ranks.begin() + (c_start - lower.lbegin), c_end - c_start, ranksOffsets.begin() + (c_start - cells[y].Child[0] + 1));
    ranksOffsets[0] = 0;
    
    if (lbegin + 1 == lend)
      comm.level_merge(&ranksOffsets[1], cells[y].Child[1] - cells[y].Child[0]);
    std::partial_sum(ranksOffsets.begin() + 1, ranksOffsets.end(), ranksOffsets.begin() + 1);
    long long i = y - lbegin;
    long long mi = M[i] = ranksOffsets.back();

    if (0 < mi && 0 < N[i]) {
      fmatrix[i].resize(mi * N[i]);
      basis[i].resize(mi * mi);
      std::fill(fmatrix[i].begin(), fmatrix[i].end(), std::complex<double>(0., 0.));

      for (long long c = c_start; c < c_end; ++c) {
        long long ci = c - lower.lbegin;
        long long m = lower.Ranks[ci];
        const std::complex<double>* matB = lower.fmatrix[ci].data() + m * lower.Ncur[ci];
        std::complex<double>* matC = fmatrix[i].data() + ranksOffsets[c - cells[y].Child[0]];
        copy2d(m, N[i], matB, m, matC, mi);
      }

      if (lbegin + 1 == lend)
        comm.level_merge(fmatrix[0].data(), fmatrix[0].size());
      Ranks[i] = computeBasis(epi, Ranks[i], 'E', mi, N[i], 0, basis[i].data(), fmatrix[i].data());

      long long size_full = cells[y].Body[1] - cells[y].Body[0];
      fmatrix[i].resize(Ranks[i] * N[i]); fmatrix[i].shrink_to_fit();
      basis[i].resize(mi * Ranks[i]); basis[i].shrink_to_fit();
      basis_full[i].resize(size_full * Ranks[i]);
      std::fill(basis_full[i].begin(), basis_full[i].end(), std::complex<double>(0., 0.));
      
      for (long long c = c_start; c < c_end; ++c) {
        long long ci = c - lower.lbegin;
        long long m = cells[c].Body[1] - cells[c].Body[0];
        long long n = Ranks[i];
        long long k = lower.Ranks[ci];
        const std::complex<double>* matA = lower.basis_full[ci].data();
        const std::complex<double>* matB = basis[i].data() + ranksOffsets[c - cells[y].Child[0]];
        std::complex<double>* matC = basis_full[i].data() + (cells[c].Body[0] - cells[y].Body[0]);
        nngemm(m, n, k, matA, m, matB, mi, matC, size_full);
      }
      
      if (lbegin + 1 == lend)
        comm.level_merge(basis_full[0].data(), basis_full[0].size());
    }
  }
}


#include <hss.hpp>
#include <dense.hpp>

#include <algorithm>
#include <numeric>
#include <map>
#include <set>

CSR::CSR(const std::vector<Cell>& cells) : N(cells.size()) {
  std::vector<std::pair<long long, long long>> rels;
  auto getList = [&](long long i, long long j, auto&& self) -> void {
    bool admis = i != j;
    if (admis) { rels.emplace_back(i, j); }
    if (!admis && cells[i].Child[0] >= 0 && cells[j].Child[0] >= 0)
      for (long long k = cells[i].Child[0]; k < cells[i].Child[1]; ++k)
        for (long long l = cells[j].Child[0]; l < cells[j].Child[1]; ++l)
          self(k, l, self);
  };
  getList(0, 0, getList);
  std::sort(rels.begin(), rels.end());

  long long len = rels.size();
  RowIndex.resize(N + 1);
  ColIndex.resize(len);

  RowIndex[0] = 0;
  std::transform(rels.begin(), rels.end(), ColIndex.begin(), [](const std::pair<long long, long long>& i) { return i.second; });
  std::for_each(rels.begin(), rels.end(), [&](const std::pair<long long, long long>& i) { ++RowIndex[i.first + 1]; });
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

HierarchicalComm::HierarchicalComm(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long CRows[], const long long CCols[], MPI_Comm world) {
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(world, &mpi_rank); MPI_Comm_size(world, &mpi_size);

  long long pbegin = Mapping[mpi_rank].first;
  long long pend = Mapping[mpi_rank].second;
  long long p = std::distance(&Mapping[0], std::find(&Mapping[0], &Mapping[mpi_rank], Mapping[mpi_rank]));
  long long lenp = std::distance(&Mapping[p], 
    std::find_if_not(&Mapping[p], &Mapping[mpi_size], [&](std::pair<long long, long long> i) { return i == Mapping[mpi_rank]; }));

  std::map<long long, long long> col_to_mpi_rank;
  std::for_each(Mapping, &Mapping[mpi_size], [&](std::pair<long long, long long>& i) {
    for (long long col = i.first; col < i.second; ++col) col_to_mpi_rank.emplace(col, (long long)std::distance(Mapping, &i)); });

  std::set<long long> cols; cols.insert(col_to_mpi_rank[pbegin]);
  std::for_each(&CCols[CRows[pbegin]], &CCols[CRows[pend]], [&](long long col) { cols.insert(col_to_mpi_rank[col]); });

  Proc = std::distance(cols.begin(), std::find(cols.begin(), cols.end(), p));
  NeighborComms = (p == mpi_rank) ? (cols.size() - 1ll) : 0ll;
  Boxes.reserve(cols.size()); long long lenBoxes = 0; for (long long i : cols)
  { long long ibegin = Mapping[i].first, iend = Mapping[i].second; Boxes.emplace_back(std::array<long long, 3>({ ibegin, iend - ibegin, lenBoxes += iend - ibegin })); }

  long long i = 0;
  std::vector<std::pair<long long, long long>> MappingNext(mpi_size, std::make_pair(-1, -1));
  while (i < (long long)mpi_size) {
    long long lenP = std::distance(&Mapping[i], std::find_if_not(&Mapping[i], &Mapping[mpi_size], 
      [&](std::pair<long long, long long> a) { return a == Mapping[i]; }));
    long long pbegin = Mapping[i].first, pend = Mapping[i].second;
    long long child = Tree[pbegin].first, lenC = Tree[pend - 1].second - child;
    if (child >= 0 && lenC > 0)
      for (long long j = 0; j < lenP; j++) {
        long long c0 = j * lenC / lenP;
        long long c1 = (j + 1) * lenC / lenP;
        c1 = std::max(c1, c0 + 1);
        MappingNext[i + j] = std::make_pair(child + c0, child + c1);
      }
    i += lenP;
  }
  std::copy(MappingNext.begin(), MappingNext.end(), Mapping);

  LocalChild = Mapping[mpi_rank].first - Tree[pbegin].first;
  MPI_Comm_split(world, mpi_rank == p ? 1 : MPI_UNDEFINED, mpi_rank, &RepComm);
  MPI_Comm_split(world, lenp > 1 ? p : MPI_UNDEFINED, mpi_rank, &DupComm);
  if (RepComm != MPI_COMM_NULL) { int rep_rank; MPI_Comm_rank(RepComm, &rep_rank); NeighborLoc = (long long)rep_rank - Proc; } else { NeighborLoc = -1ll; }
}

long long HierarchicalComm::oLocal() const { return Boxes[Proc][2] - Boxes[Proc][1]; }
long long HierarchicalComm::oGlobal() const { return Boxes[Proc][0]; }
long long HierarchicalComm::lenLocal() const { return Boxes[Proc][1]; }
long long HierarchicalComm::lenNeighbors() const { return Boxes.back()[2]; }
long long HierarchicalComm::oLocalChild(const HierarchicalComm& lower) const { return lower.oLocal() - LocalChild; }

void HierarchicalComm::bcastNeighbor(void* data, const long long noffsets[], size_t size) const {
  std::vector<MPI_Request> requests(NeighborComms * 2);
  record_mpi();

  long long send_loc = noffsets[Boxes[Proc][2] - Boxes[Proc][1]];
  long long send_len = (noffsets[Boxes[Proc][2]] - send_loc) * size;
  unsigned char* send_start = &reinterpret_cast<unsigned char*>(data)[send_loc * size];

  for (long long p = 0; p < NeighborComms; ++p) {
    long long dst = p + (long long)(Proc <= p);
    long long recv_loc = noffsets[Boxes[dst][2] - Boxes[dst][1]];
    long long recv_len = (noffsets[Boxes[dst][2]] - recv_loc) * size;
    unsigned char* recv_start = &reinterpret_cast<unsigned char*>(data)[recv_loc * size];

    long long dst_rank = NeighborLoc + dst;
    MPI_Isend(send_start, send_len, MPI_BYTE, dst_rank, 0, RepComm, &requests[p << 1]);
    MPI_Irecv(recv_start, recv_len, MPI_BYTE, dst_rank, 0, RepComm, &requests[(p << 1) | 1]);
  }

  if (0 < requests.size())
    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
  if (DupComm != MPI_COMM_NULL)
    MPI_Bcast(data, noffsets[Boxes.back()[2]] * size, MPI_BYTE, 0, DupComm);
  record_mpi();
}

void HierarchicalComm::bcastNeighbor(long long data[], const long long noffsets[]) const
{ bcastNeighbor(data, noffsets, sizeof(long long)); }

void HierarchicalComm::bcastNeighbor(std::complex<double> data[], const long long noffsets[]) const
{ bcastNeighbor(data, noffsets, sizeof(std::complex<double>)); }

std::pair<double, double> timer = std::make_pair(0., 0.);

double HierarchicalComm::get_comm_time(MPI_Comm world) {
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);
  double lapse = timer.first;
  MPI_Allreduce(MPI_IN_PLACE, &lapse, 1, MPI_DOUBLE, MPI_SUM, world);
  timer = std::make_pair(0., 0.);
  return lapse / mpi_size;
}

void HierarchicalComm::record_mpi() {
  if (timer.second == 0.)
    timer.second = MPI_Wtime();
  else {
    timer.first = timer.first + (MPI_Wtime() - timer.second);
    timer.second = 0.;
  }
}

void HSSLevel::eval(double epi, DenseMatrix& matA, const long long localM[], const long long localN[], const CSR& rels, const HierarchicalComm& comm, long long localNoffset) {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long xlen = comm.lenNeighbors();

  URowPrefixes.resize(xlen + 1); UColPrefixes.resize(xlen + 1);
  VRowPrefixes.resize(xlen + 1); VColPrefixes.resize(xlen + 1);
  UPrefixes.resize(nodes + 1); VPrefixes.resize(nodes + 1);
  CPrefixes.resize(rels.RowIndex[nodes] + 1);

  if (0ll <= localNoffset) {
    APrefixes.resize(nodes + 1);
    std::transform(localM, &localM[nodes], localN, APrefixes.begin() + 1, std::multiplies<long long>());
    std::partial_sum(APrefixes.begin() + 1, APrefixes.end(), APrefixes.begin() + 1);
    APrefixes[0] = 0ll; A.resize(APrefixes.back());

    long long localMoffset = matA.rowOffsets[matA.prime_rank];
    for (long long i = 0ll; i < nodes; ++i) {
      long long M = localM[i], N = localN[i];
      matA.AijExtractBlock(M, N, localMoffset, localNoffset, &A[APrefixes[i]], M);
      localMoffset += M; localNoffset += N;
    }
  }

  if (rels.RowIndex[nodes]) {
    long long colOffset = 0ll;
    URowPrefixes[0] = UColPrefixes[0] = VRowPrefixes[0] = VColPrefixes[0] = UPrefixes[0] = VPrefixes[0] = CPrefixes[0] = 0ll;
    std::vector<long long> DimsMLr(&localM[0], &localM[nodes]), DimsNLr(&localN[0], &localN[nodes]);
    matA.compress(epi, DimsMLr, U);
    do {
      DenseMatrix matAH(std::reduce(&localN[0], &localN[nodes]), matA.rowOffsets.back(), matA.prime, matA.sub);
      matA.adjointA(matAH); matAH.compress(epi, DimsNLr, V);
      matA = DenseMatrix(std::reduce(DimsMLr.begin(), DimsMLr.end()), matAH.rowOffsets.back(), matAH.prime, matAH.sub);
      matAH.adjointA(matA); colOffset = matAH.rowOffsets[matAH.prime_rank];
    } while(0);

    std::transform(&localM[0], &localM[nodes], &DimsMLr[0], UPrefixes.begin() + 1, std::multiplies<long long>());
    std::transform(&localN[0], &localN[nodes], &DimsNLr[0], VPrefixes.begin() + 1, std::multiplies<long long>());
    std::inclusive_scan(UPrefixes.begin() + 1, UPrefixes.end(), UPrefixes.begin() + 1);
    std::inclusive_scan(VPrefixes.begin() + 1, VPrefixes.end(), VPrefixes.begin() + 1);

    std::vector<long long> seq(xlen + 1); std::vector<std::array<long long, 4>> dims(xlen + 1);
    for (long long i = 0ll; i < nodes; ++i) { dims[i + ibegin] = std::array<long long, 4>{ localM[i], localN[i], DimsMLr[i], DimsNLr[i] }; }
    for (long long i = 0ll; i <= xlen; ++i) { seq[i] = i * 4ll; }
    comm.bcastNeighbor((long long*)dims.data(), seq.data());
    std::inclusive_scan(dims.begin(), dims.end(), dims.begin(),
      [](const std::array<long long, 4>& a, const std::array<long long, 4>& b) { return std::array<long long, 4>{ a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3] }; });
    
    for (long long i = 0ll; i < xlen; ++i) {
      long long i1 = i + 1ll; const auto& [m, n, k, l] = dims[i];
      URowPrefixes[i1] = m; VRowPrefixes[i1] = n; UColPrefixes[i1] = k; VColPrefixes[i1] = l;
    }

    for (long long i = 0ll; i < nodes; ++i)
      std::transform(rels.ColIndex.begin() + rels.RowIndex[i], rels.ColIndex.begin() + rels.RowIndex[i + 1], CPrefixes.begin() + rels.RowIndex[i] + 1,
        [&](long long col) { return (UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin]) * (VColPrefixes[col + 1] - VColPrefixes[col]); });
    std::inclusive_scan(CPrefixes.begin() + 1, CPrefixes.end(), CPrefixes.begin() + 1);
    C.resize(CPrefixes.back());

    long long rowOffset = matA.rowOffsets[matA.prime_rank] - UColPrefixes[ibegin]; colOffset -= VColPrefixes[ibegin];
    for (long long i = 0; i < nodes; i++) {
      long long mC = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
      long long y = rowOffset + UColPrefixes[ibegin + i];
      for (long long ij = rels.RowIndex[i]; ij < rels.RowIndex[i + 1] && 0 < mC; ij++) {
        long long j = rels.ColIndex[ij];
        long long nC = VColPrefixes[j + 1] - VColPrefixes[j];
        long long x = colOffset + VColPrefixes[j];
        matA.AijExtractBlock(mC, nC, y, x, &C[CPrefixes[ij]], mC);
      }
    }
  } else {
    std::vector<long long> seq(xlen + 1); std::vector<std::array<long long, 2>> dims(xlen + 1);
    for (long long i = 0ll; i < nodes; ++i) { dims[i + ibegin] = std::array<long long, 2>{ localM[i], localN[i] }; }
    for (long long i = 0ll; i <= xlen; ++i) { seq[i] = i * 2ll; }
    comm.bcastNeighbor((long long*)dims.data(), seq.data());
    std::inclusive_scan(dims.begin(), dims.end(), dims.begin(),
      [](const std::array<long long, 2>& a, const std::array<long long, 2>& b) { return std::array<long long, 2>{ a[0] + b[0], a[1] + b[1] }; });
    
    URowPrefixes[0] = VRowPrefixes[0] = CPrefixes[0] = 0ll;
    for (long long i = 0ll; i < xlen; ++i) {
      long long i1 = i + 1ll; const auto& [m, n] = dims[i];
      URowPrefixes[i1] = m; VRowPrefixes[i1] = n;
    }

    std::fill(UColPrefixes.begin(), UColPrefixes.end(), 0ll);
    std::fill(VColPrefixes.begin(), VColPrefixes.end(), 0ll);
    std::fill(UPrefixes.begin(), UPrefixes.end(), 0ll);
    std::fill(VPrefixes.begin(), VPrefixes.end(), 0ll);
  }
}

extern "C" void zgemv_(const char*, const int*, const int*, const std::complex<double>*, const std::complex<double>*, const int*, const std::complex<double>*, const int*, const std::complex<double>*, std::complex<double>*, const int*) noexcept;

void HSSLevel::matVecUpwardPass(const std::complex<double>* X_in, std::complex<double>* Z, const HierarchicalComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  auto gemv = [](int m, int n, const std::complex<double>* a, const std::complex<double>* x, std::complex<double>* b) {
    char transA = 'C'; int incx = 1; std::complex<double> one(1., 0.), zero(0., 0.); zgemv_(&transA, &n, &m, &one, a, &n, x, &incx, &zero, b, &incx);
  };

  for (long long i = 0; i < nodes; i++) {
    long long M = VColPrefixes[i + ibegin + 1] - VColPrefixes[i + ibegin];
    long long N = VRowPrefixes[i + ibegin + 1] - VRowPrefixes[i + ibegin];
    if (0 < N)
    { gemv(M, N, &V[VPrefixes[i]], &X_in[VRowPrefixes[i + ibegin] - VRowPrefixes[ibegin]], &Z[VColPrefixes[i + ibegin]]); }
  }
  comm.bcastNeighbor(Z, VColPrefixes.data());
}

void HSSLevel::matVecHorizontalandDownwardPass(std::complex<double>* Y_out, const std::complex<double>* Z, std::complex<double>* Y, const CSR& rels, const HierarchicalComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long lenX = URowPrefixes[ibegin + nodes] - URowPrefixes[ibegin];
  std::fill(Y_out, Y_out + lenX, std::complex<double>(0., 0.));
  auto gemv_b1 = [](int m, int n, const std::complex<double>* a, const std::complex<double>* x, std::complex<double>* b) {
    char transA = 'N'; int incx = 1; std::complex<double> one(1., 0.); zgemv_(&transA, &m, &n, &one, a, &m, x, &incx, &one, b, &incx);
  };
  auto gemv_b0 = [](int m, int n, const std::complex<double>* a, const std::complex<double>* x, std::complex<double>* b) {
    char transA = 'N'; int incx = 1; std::complex<double> one(1., 0.), zero(0., 0.); zgemv_(&transA, &m, &n, &one, a, &m, x, &incx, &zero, b, &incx);
  };

  for (long long i = 0; i < nodes; i++) {
    long long M = URowPrefixes[i + ibegin + 1] - URowPrefixes[i + ibegin];
    long long K = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
    if (0 < K) {
      std::complex<double>* y = &Y[UColPrefixes[i + ibegin]];
      for (long long ij = rels.RowIndex[i]; ij < rels.RowIndex[i + 1]; ij++) {
        long long j = rels.ColIndex[ij];
        long long N = VColPrefixes[j + 1] - VColPrefixes[j];
        gemv_b1(K, N, &C[CPrefixes[ij]], &Z[VColPrefixes[j]], y);
      }
      gemv_b0(M, K, &U[UPrefixes[i]], y, &Y_out[URowPrefixes[i + ibegin] - URowPrefixes[ibegin]]);
    }
  }
}

void HSSLevel::matVecLeafHorizontalPass(std::complex<double>* X_io, const std::complex<double>* Z, std::complex<double>* Y, const CSR& rels, const HierarchicalComm& comm) const {
  long long ibegin = comm.oLocal();
  long long nodes = comm.lenLocal();
  long long lenX = VRowPrefixes[ibegin + nodes] - VRowPrefixes[ibegin];

  std::vector<std::complex<double>> X(VRowPrefixes.back(), std::complex<double>(0., 0.));
  std::copy(&X_io[0], &X_io[lenX], X.begin() + VRowPrefixes[ibegin]);
  comm.bcastNeighbor(X.data(), VRowPrefixes.data());
  std::fill(&X_io[0], &X_io[lenX], std::complex<double>(0., 0.));
  auto gemv = [](int m, int n, const std::complex<double>* a, const std::complex<double>* x, std::complex<double>* b) {
    char transA = 'N'; int incx = 1; std::complex<double> one(1., 0.); zgemv_(&transA, &m, &n, &one, a, &m, x, &incx, &one, b, &incx);
  };

  for (long long i = 0; i < nodes; i++) {
    long long M = URowPrefixes[i + ibegin + 1] - URowPrefixes[i + ibegin];
    long long K = UColPrefixes[i + ibegin + 1] - UColPrefixes[i + ibegin];
    std::complex<double>* w = &X_io[URowPrefixes[i + ibegin] - URowPrefixes[ibegin]];
    
    if (0 < K) {
      std::complex<double>* y = &Y[UColPrefixes[i + ibegin]];
      for (long long ij = rels.RowIndex[i]; ij < rels.RowIndex[i + 1]; ij++) {
        long long j = rels.ColIndex[ij];
        long long N = VColPrefixes[j + 1] - VColPrefixes[j];
        gemv(K, N, &C[CPrefixes[ij]], &Z[VColPrefixes[j]], y);
      }
      gemv(M, K, &U[UPrefixes[i]], y, w);
    }

    long long N = VRowPrefixes[i + ibegin + 1] - VRowPrefixes[i + ibegin];
    gemv(M, N, &A[APrefixes[i]], &X[VRowPrefixes[i + ibegin]], w);
  }
}

HSS::HSS(const std::vector<Cell>& cells, long long levels, MPI_Comm world) : 
  levels(levels), world(world), A(levels + 1) {
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);

  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  CSR Far(cells);
  for (long long i = 0; i <= levels; ++i) {
    comm.emplace_back(&tree[0], &mapping[0], Far.RowIndex.data(), Far.ColIndex.data(), world);
    Far_local.emplace_back(Far, comm[i].oGlobal(), comm[i].oGlobal() + comm[i].lenLocal(), comm[i].Boxes);
  }

  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].N[0], cells[gbegin + llen - 1].N[1]);
}

void HSS::evalMatVecA(DenseMatrix& matA, double epi, const Cell cells[]) {
  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  std::vector<long long> localM, localN(llen);
  std::transform(&cells[gbegin], &cells[gbegin + llen], localN.begin(), [](const Cell& c) { return c.N[1] - c.N[0]; });
  A[levels].eval(epi, matA, &localN[0], &localN[0], Far_local[levels], comm[levels], cells[gbegin].N[0]);

  for (long long l = levels - 1; l >= 0; l--) {
    do {
      DenseMatrix matAl(A[l + 1].UColPrefixes.back(), matA.N, comm[l].RepComm, comm[l].DupComm);
      matAl.copyFromA(matA); matA = std::move(matAl);
    } while(0);
    llen = comm[l].lenLocal(); gbegin = comm[l].oGlobal(); localM.resize(llen); localN.resize(llen);
    for (long long i = 0; i < llen; ++i) {
      long long cbegin = cells[gbegin + i].Child[0] - cells[gbegin].Child[0], cend = cells[gbegin + i].Child[1] - cells[gbegin].Child[0];
      localM[i] = A[l + 1].UColPrefixes[cend] - A[l + 1].UColPrefixes[cbegin];
      localN[i] = A[l + 1].VColPrefixes[cend] - A[l + 1].VColPrefixes[cbegin];
    }
    A[l].eval(epi, matA, &localM[0], &localN[0], Far_local[l], comm[l]);
  }
}

void HSS::allocRHSwork(std::vector<long long>& iwork, std::vector<std::complex<double>>& cwork) const {
  iwork.resize((levels * 4ll) + 4ll);
  long long* lenX = &iwork[0], *lenY = &iwork[levels + 2ll], *locV = &iwork[(levels * 2ll) + 4ll], *locU = &iwork[(levels * 3ll) + 4ll];
  lenX[0] = lenY[0] = 0;
  for (long long l = 0; l < levels; l++) {
    lenX[l + 1] = 0ll < l ? A[l].VRowPrefixes.back() : 0ll;
    lenY[l + 1] = 0ll < l ? A[l].URowPrefixes.back() : 0ll;
    locV[l] = A[l + 1].VColPrefixes[comm[l].oLocalChild(comm[l + 1])];
    locU[l] = A[l + 1].UColPrefixes[comm[l].oLocalChild(comm[l + 1])];
  }

  lenX[levels + 1] = A[levels].VRowPrefixes.back();
  lenY[levels + 1] = A[levels].URowPrefixes.back();
  std::partial_sum(&lenX[1], locV, &lenX[1]);
  cwork.resize(lenY[levels + 1]);
}

void HSS::matVecMul(std::complex<double> X[]) const {
  std::vector<long long> iworkA;
  std::vector<std::complex<double>> cworkA;
  allocRHSwork(iworkA, cworkA);
  matVecMul(X, iworkA.data(), cworkA.data());
}

void HSS::matVecMul(std::complex<double> X[], const long long iwork[], std::complex<double> cwork[]) const {
  const long long* lenX = iwork, *lenY = &iwork[levels + 2ll], *locV = &iwork[(levels * 2ll) + 4ll], *locU = &iwork[(levels * 3ll) + 4ll];
  if (0 < levels)
    A[levels].matVecUpwardPass(X, &cwork[lenX[levels]], comm[levels]);
  for (long long l = levels - 1; l >= 0; l--)
    A[l].matVecUpwardPass(&cwork[lenX[l + 1] + locV[l]], &cwork[lenX[l]], comm[l]);

  for (long long l = 0; l < levels; l++)
    A[l].matVecHorizontalandDownwardPass(&cwork[lenY[l + 1] + locU[l]], &cwork[lenX[l]], &cwork[lenY[l]], Far_local[l], comm[l]);
  A[levels].matVecLeafHorizontalPass(X, &cwork[lenX[levels]], &cwork[lenY[levels]], Far_local[levels], comm[levels]);
}

double HSS::solveRelErr(const std::complex<double> X[], const std::complex<double> ref[]) {
  long long lenX = local_bodies.second - local_bodies.first;
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    err[0] += std::norm(X[i] - ref[i]);
    err[1] += std::norm(ref[i]);
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, world);
  return std::sqrt(err[0] / err[1]);
}

void HSS::free_all_comms() {
  for (HierarchicalComm& com : comm) {
    if (com.RepComm != MPI_COMM_NULL)
      MPI_Comm_free(&com.RepComm);
    if (com.DupComm != MPI_COMM_NULL)
      MPI_Comm_free(&com.DupComm);
  }
}

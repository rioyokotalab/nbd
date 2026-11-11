
#include <comm-mpi.hpp>

#include <algorithm>
#include <set>
#include <map>

void getNextLevelMapping(std::pair<long long, long long> Mapping[], const std::pair<long long, long long> Tree[], long long mpi_size) {
  long long p = 0;
  std::vector<std::pair<long long, long long>> MappingNext(mpi_size, std::make_pair(-1, -1));

  while (p < mpi_size) {
    long long lenP = std::distance(&Mapping[p], std::find_if_not(&Mapping[p], &Mapping[mpi_size], 
      [&](std::pair<long long, long long> a) { return a == Mapping[p]; }));
    long long pbegin = Mapping[p].first;
    long long pend = Mapping[p].second;
    long long child = Tree[pbegin].first;
    long long lenC = Tree[pend - 1].second - child;

    if (child >= 0 && lenC > 0)
      for (long long j = 0; j < lenP; j++) {
        long long c0 = j * lenC / lenP;
        long long c1 = (j + 1) * lenC / lenP;
        c1 = std::max(c1, c0 + 1);
        MappingNext[p + j] = std::make_pair(child + c0, child + c1);
      }
    p += lenP;
  }
  std::copy(MappingNext.begin(), MappingNext.end(), Mapping);
}

H2MatrixComm::H2MatrixComm(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long CRows[], const long long CCols[], MPI_Comm world) : worldComm(world) {
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(world, &mpi_rank);
  MPI_Comm_size(world, &mpi_size);

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

  NeighborRanks.resize(cols.size());
  std::copy(cols.begin(), cols.end(), NeighborRanks.begin());
  Proc = std::distance(NeighborRanks.begin(), std::find(NeighborRanks.begin(), NeighborRanks.end(), p));
  Boxes.reserve(NeighborRanks.size());
  NeighborComms = (p == mpi_rank) ? (NeighborRanks.size() - 1ll) : 0ll;

  long long lenBoxes = 0;
  for (long long i = 0; i < (long long)NeighborRanks.size(); i++) {
    long long ibegin = Mapping[NeighborRanks[i]].first, iend = Mapping[NeighborRanks[i]].second;
    Boxes.emplace_back(std::array<long long, 3>({ ibegin, iend - ibegin, lenBoxes += iend - ibegin }));
  }

  getNextLevelMapping(&Mapping[0], Tree, mpi_size);
  LocalChild = std::make_pair(Mapping[mpi_rank].first - Tree[pbegin].first, Tree[pend - 1].second - Mapping[mpi_rank].first);
  MPI_Comm_split(world, mpi_rank == p ? 1 : MPI_UNDEFINED, mpi_rank, &RepComm);
  MPI_Comm_split(world, lenp > 1 ? p : MPI_UNDEFINED, mpi_rank, &DupComm);
}

long long H2MatrixComm::oLocal() const {
  return Boxes[Proc][2] - Boxes[Proc][1];
}

long long H2MatrixComm::oGlobal() const {
  return Boxes[Proc][0];
}

long long H2MatrixComm::lenLocal() const {
  return Boxes[Proc][1];
}

long long H2MatrixComm::lenNeighbors() const {
  return Boxes.back()[2]; 
}

std::pair<long long, long long> H2MatrixComm::oLocalChild(const H2MatrixComm& lower) const {
  return std::make_pair(lower.oLocal() - LocalChild.first, lower.oLocal() + LocalChild.second);
}

void H2MatrixComm::neighbor_bcast(void* data, const long long noffsets[], size_t size) const {
  std::vector<MPI_Request> requests(NeighborComms * 2);
  record_mpi();

  long long src_rank = NeighborRanks[Proc];
  long long send_loc = noffsets[Boxes[Proc][2] - Boxes[Proc][1]];
  long long send_len = (noffsets[Boxes[Proc][2]] - send_loc) * size;
  unsigned char* send_start = &reinterpret_cast<unsigned char*>(data)[send_loc * size];

  for (long long p = 0; p < NeighborComms; ++p) {
    long long dst = p + (long long)(Proc <= p);
    long long dst_rank = NeighborRanks[dst];
    long long recv_loc = noffsets[Boxes[dst][2] - Boxes[dst][1]];
    long long recv_len = (noffsets[Boxes[dst][2]] - recv_loc) * size;
    unsigned char* recv_start = &reinterpret_cast<unsigned char*>(data)[recv_loc * size];

    MPI_Isend(send_start, send_len, MPI_BYTE, dst_rank, src_rank, worldComm, &requests[p << 1]);
    MPI_Irecv(recv_start, recv_len, MPI_BYTE, dst_rank, dst_rank, worldComm, &requests[(p << 1) | 1]);
  }

  if (0 < requests.size())
    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
  if (DupComm != MPI_COMM_NULL)
    MPI_Bcast(data, noffsets[Boxes.back()[2]] * size, MPI_BYTE, 0, DupComm);
  record_mpi();
}

void H2MatrixComm::neighbor_bcast(long long data[], const long long noffsets[]) const {
  neighbor_bcast(data, noffsets, sizeof(long long));
}

void H2MatrixComm::neighbor_bcast(std::complex<double> data[], const long long noffsets[]) const {
  neighbor_bcast(data, noffsets, sizeof(std::complex<double>));
}

std::pair<double, double> timer = std::make_pair(0., 0.);

double H2MatrixComm::get_comm_time(MPI_Comm world) {
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);
  double lapse = timer.first;
  MPI_Allreduce(MPI_IN_PLACE, &lapse, 1, MPI_DOUBLE, MPI_SUM, world);
  timer = std::make_pair(0., 0.);
  return lapse / mpi_size;
}

void H2MatrixComm::record_mpi() {
  if (timer.second == 0.)
    timer.second = MPI_Wtime();
  else {
    timer.first = timer.first + (MPI_Wtime() - timer.second);
    timer.second = 0.;
  }
}



#pragma once

#include <mpi.h>
#include <vector>
#include <complex>
#include <array>

class H2MatrixComm {
public:
  long long Proc;
  std::pair<long long, long long> LocalChild;
  std::vector<std::array<long long, 3>> Boxes;
  
  long long NeighborComms;
  std::vector<long long> NeighborRanks;
  MPI_Comm MergeComm;
  MPI_Comm DupComm;
  MPI_Comm worldComm;

  H2MatrixComm(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long ARows[], const long long ACols[], const long long CRows[], const long long CCols[], MPI_Comm world = MPI_COMM_WORLD);
  
  long long oLocal() const;
  long long oGlobal() const;
  long long lenLocal() const;
  long long lenNeighbors() const;
  std::pair<long long, long long> oLocalChild(const H2MatrixComm& lower) const;

  void level_merge(void* data, long long len, size_t size) const;
  void neighbor_bcast(void* data, const long long noffsets[], size_t size) const;

  void level_merge(long long* data, long long len) const;
  void level_merge(std::complex<double>* data, long long len) const;
  void neighbor_bcast(long long data[], const long long noffsets[]) const;
  void neighbor_bcast(std::complex<double> data[], const long long noffsets[]) const;

  static double get_comm_time(MPI_Comm world);
  static void record_mpi();
};


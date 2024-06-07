
#pragma once

#include "mpi.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cstddef>

class CommTimer {
public:
  std::vector<std::pair<double, double>> timings;

  void record_mpi(double t1, double t2)
  { timings.emplace_back(t1, t2); }

  double get_comm_timing() {
    double sum = 0.;
    for (auto& i : timings)
      sum = sum + i.second - i.first;
    timings.clear();
    return sum;
  }
};

class CellComm {
public:
  int64_t Proc;
  std::vector<std::pair<int64_t, int64_t>> ProcBoxes;
  std::vector<std::pair<int64_t, int64_t>> LocalChild, LocalParent;
  
  std::vector<std::pair<int, MPI_Comm>> Comm_box;
  MPI_Comm Comm_share, Comm_merge;
  
  CommTimer* timer;

  int64_t iLocal(int64_t iglobal) const;
  int64_t iGlobal(int64_t ilocal) const;
};

class Cell;
class CSR;

void buildComm(CellComm* comms, int64_t ncells, const Cell* cells, const CSR* cellFar, const CSR* cellNear, int64_t levels);

void cellComm_free(CellComm* comms, int64_t levels);

void relations(CSR rels[], const CSR* cellRel, int64_t levels, const CellComm* comm);

void content_length(int64_t* local, int64_t* neighbors, int64_t* local_off, const CellComm* comm);

int64_t neighbor_bcast_sizes_cpu(int64_t* data, const CellComm* comm);

void neighbor_bcast_cpu(double* data, int64_t seg, const CellComm* comm);

void neighbor_reduce_cpu(double* data, int64_t seg, const CellComm* comm);

void level_merge_cpu(double* data, int64_t len, const CellComm* comm);

void dup_bcast_cpu(double* data, int64_t len, const CellComm* comm);


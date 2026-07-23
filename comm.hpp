
#pragma once

#include "mpi.h"
#include "cuda_runtime_api.h"
#include "nccl.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cstddef>

class CellComm {
public:
  int64_t Proc;
  std::vector<std::pair<int64_t, int64_t>> ProcBoxes;
  std::vector<std::pair<int64_t, int64_t>> LocalChild, LocalParent;
  
  std::vector<std::pair<int, MPI_Comm>> Comm_box;
  MPI_Comm Comm_share, Comm_merge;

  cudaStream_t stream;
  std::vector<std::pair<int, ncclComm_t>> NCCL_box;
  ncclComm_t NCCL_merge, NCCL_share;
  
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

void neighbor_bcast_gpu(double* data, int64_t seg, const CellComm* comm);

void neighbor_reduce_gpu(double* data, int64_t seg, const CellComm* comm);

void level_merge_gpu(double* data, int64_t len, const CellComm* comm);

void dup_bcast_gpu(double* data, int64_t len, const CellComm* comm);


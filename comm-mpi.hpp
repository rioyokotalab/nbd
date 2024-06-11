
#pragma once

#include <mpi.h>
#include <vector>
#include <complex>

class ColCommMPI {
protected:
  long long Proc;
  std::vector<std::pair<long long, long long>> Boxes;
  
  std::pair<int, MPI_Comm> MergeComm;
  std::vector<std::pair<int, MPI_Comm>> NeighborComm;
  MPI_Comm AllReduceComm;
  MPI_Comm DupComm;
  std::vector<MPI_Comm> allocedComm;

  template<typename T> inline void level_merge(T* data, long long len) const;
  template<typename T> inline void level_sum(T* data, long long len) const;
  template<typename T> inline void neighbor_bcast(T* data, const long long box_dims[]) const;
  template<typename T> inline void neighbor_reduce(T* data, const long long box_dims[]) const;
  template<typename T> inline void dup_bcast(T* data, long long len) const;

public:
  std::pair<double, double>* timer;

  ColCommMPI() : Proc(-1), Boxes(), MergeComm(0, MPI_COMM_NULL), NeighborComm(), AllReduceComm(MPI_COMM_NULL), DupComm(MPI_COMM_NULL), allocedComm(), timer(nullptr) {};
  ColCommMPI(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long Rows[], const long long Cols[], MPI_Comm world = MPI_COMM_WORLD);
  
  long long iLocal(long long iglobal) const;
  long long iGlobal(long long ilocal) const;
  long long oLocal() const;
  long long oGlobal() const;
  long long lenLocal() const;
  long long lenNeighbors() const;

  void level_merge(double* data, long long len) const;
  void level_merge(std::complex<double>* data, long long len) const;

  void level_sum(std::complex<double>* data, long long len) const;

  void neighbor_bcast(long long* data, const long long box_dims[]) const;
  void neighbor_bcast(double* data, const long long box_dims[]) const;
  void neighbor_bcast(std::complex<double>* data, const long long box_dims[]) const;

  void neighbor_reduce(long long* data, const long long box_dims[]) const;
  void neighbor_reduce(double* data, const long long box_dims[]) const;
  void neighbor_reduce(std::complex<double>* data, const long long box_dims[]) const;

  void dup_bcast(double* data, long long len) const;

  void record_mpi() const;

  void free_all_comms();
};

class Cell;
class CSR;

void buildComm(ColCommMPI* comms, int64_t ncells, const Cell* cells, const CSR* cellFar, const CSR* cellNear, int64_t levels);

void cellComm_free(ColCommMPI* comms, int64_t levels);

void content_length(int64_t* local, int64_t* neighbors, int64_t* local_off, const ColCommMPI* comm);

void neighbor_bcast_sizes_cpu(int64_t* data, const ColCommMPI* comm);

void neighbor_bcast_cpu(double* data, int64_t seg, const ColCommMPI* comm);

void neighbor_reduce_cpu(double* data, int64_t seg, const ColCommMPI* comm);

void level_merge_cpu(double* data, int64_t len, const ColCommMPI* comm);

void dup_bcast_cpu(double* data, int64_t len, const ColCommMPI* comm);



#include <comm-mpi.hpp>

#include <cuda_runtime_api.h>
#include <nccl.h>

class ColCommNCCL : public ColCommMPI {
private:
  ncclComm_t MergeNCCL;
  std::vector<ncclComm_t> NeighborNCCL;
  ncclComm_t AllReduceNCCL;
  ncclComm_t DupNCCL;
  std::vector<ncclComm_t> allocedNCCL;

  template<typename T> inline void level_merge(T* data, long long len, cudaStream_t stream) const;
  template<typename T> inline void level_sum(T* data, long long len, cudaStream_t stream) const;
  template<typename T> inline void neighbor_bcast(T* data, const long long box_dims[], cudaStream_t stream) const;
  template<typename T> inline void neighbor_reduce(T* data, const long long box_dims[], cudaStream_t stream) const;

public:
  ColCommNCCL() : ColCommMPI(), MergeNCCL(nullptr), NeighborNCCL(), AllReduceNCCL(nullptr), DupNCCL(nullptr), allocedNCCL() {};
  ColCommNCCL(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long Rows[], const long long Cols[], MPI_Comm world = MPI_COMM_WORLD);

  void free_all_comms();
  static int set_device(MPI_Comm world = MPI_COMM_WORLD);
};


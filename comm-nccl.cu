
#include <comm-nccl.hpp>

#include <algorithm>
#include <numeric>

template<typename T> inline ncclDataType_t get_nccl_datatype() {
  if (typeid(T) == typeid(long long))
    return ncclInt64;
  if (typeid(T) == typeid(double))
    return ncclDouble;
  if (typeid(T) == typeid(std::complex<double>))
    return ncclDouble;
  return ncclChar;
}

template<typename T> inline int get_data_multiplier() {
  if (typeid(T) == typeid(long long))
    return 1;
  if (typeid(T) == typeid(double))
    return 1;
  if (typeid(T) == typeid(std::complex<double>))
    return 2;
  return sizeof(T);
}

template<typename T> inline void ColCommNCCL::level_merge(T* data, long long len, cudaStream_t stream) const {
  if (MergeNCCL != nullptr) {
    long long mlen = len * get_data_multiplier<T>();
    ncclReduce((const void*)data, data, mlen, get_nccl_datatype<T>(), ncclSum, 0, MergeNCCL, stream);
  }
}

template<typename T> inline void ColCommNCCL::level_sum(T* data, long long len, cudaStream_t stream) const {
  long long mlen = len * get_data_multiplier<T>();
  if (AllReduceNCCL != nullptr)
    ncclAllReduce((const void*)data, data, mlen, get_nccl_datatype<T>(), ncclSum, AllReduceNCCL, stream);
  if (DupNCCL != nullptr)
    ncclBroadcast((const void*)data, data, mlen, get_nccl_datatype<T>(), 0, DupNCCL, stream);
}

template<typename T> inline void ColCommNCCL::neighbor_bcast(T* data, const long long box_dims[], cudaStream_t stream) const {
  std::vector<long long> offsets(Boxes.size() + 1, 0);
  for (long long p = 0; p < (long long)Boxes.size(); p++) {
    long long end = Boxes[p].second;
    offsets[p + 1] = std::reduce(box_dims, &box_dims[end], offsets[p]);
    box_dims = &box_dims[end];
  }

  for (long long p = 0; p < (long long)NeighborNCCL.size(); p++) {
    long long llen = (offsets[p + 1] - offsets[p]) * get_data_multiplier<T>();
    void* loc = &data[offsets[p]];
    ncclBroadcast((const void*)loc, loc, llen, get_nccl_datatype<T>(), NeighborComm[p].first, NeighborNCCL[p], stream);
  }
  if (DupNCCL != nullptr) {
    long long mlen = offsets.back() * get_data_multiplier<T>();
    ncclBroadcast((const void*)data, data, mlen, get_nccl_datatype<T>(), 0, DupNCCL, stream);
  }
}

template<typename T> inline void ColCommNCCL::neighbor_reduce(T* data, const long long box_dims[], cudaStream_t stream) const {
  std::vector<long long> offsets(Boxes.size() + 1, 0);
  for (long long p = 0; p < (long long)Boxes.size(); p++) {
    long long end = Boxes[p].second;
    offsets[p + 1] = std::reduce(box_dims, &box_dims[end], offsets[p]);
    box_dims = &box_dims[end];
  }

  for (long long p = 0; p < (long long)NeighborNCCL.size(); p++) {
    long long llen = (offsets[p + 1] - offsets[p]) * get_data_multiplier<T>();
    void* loc = &data[offsets[p]];
    ncclReduce((const void*)loc, loc, llen, get_nccl_datatype<T>(), ncclSum, NeighborComm[p].first, NeighborNCCL[p], stream);
  }
  if (DupNCCL != nullptr) {
    long long mlen = offsets.back() * get_data_multiplier<T>();
    ncclBroadcast((const void*)data, data, mlen, get_nccl_datatype<T>(), 0, DupNCCL, stream);
  }
}

ColCommNCCL::ColCommNCCL(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long Rows[], const long long Cols[], MPI_Comm world) : 
  ColCommMPI(Tree, Mapping, Rows, Cols, world) {

  long long len = allocedComm.size();
  std::vector<ncclUniqueId> ids(len);
  allocedNCCL = std::vector<ncclComm_t>(len);

  ncclGroupStart();
  for (long long i = 0; i < len; i++) {
    int rank, size;
    MPI_Comm_rank(allocedComm[i], &rank);
    MPI_Comm_size(allocedComm[i], &size);
    if (rank == 0)
      ncclGetUniqueId(&ids[i]);
    MPI_Bcast((void*)&ids[i], sizeof(ncclUniqueId), MPI_BYTE, 0, allocedComm[i]);
    ncclCommInitRank(&allocedNCCL[i], size, ids[i], rank);
  }
  ncclGroupEnd();

  auto find_nccl = [&](const MPI_Comm& c) {
    const std::vector<MPI_Comm>::iterator i = std::find(allocedComm.begin(), allocedComm.end(), c);
    return i == allocedComm.end() ? nullptr : *(allocedNCCL.begin() + std::distance(allocedComm.begin(), i));
  };

  NeighborNCCL = std::vector<ncclComm_t>(NeighborComm.size(), nullptr);
  MergeNCCL = find_nccl(MergeComm.second);
  for (long long i = 0; i < (long long)NeighborComm.size(); i++)
    NeighborNCCL[i] = find_nccl(NeighborComm[i].second);
  AllReduceNCCL = find_nccl(AllReduceComm);
  DupNCCL = find_nccl(DupComm);
}

void ColCommNCCL::free_all_comms() {
  MergeComm = std::make_pair(0, MPI_COMM_NULL);
  NeighborComm.clear();
  AllReduceComm = MPI_COMM_NULL;
  DupComm = MPI_COMM_NULL;
  
  for (MPI_Comm& c : allocedComm)
    MPI_Comm_free(&c);
  allocedComm.clear();

  MergeNCCL = nullptr;
  NeighborNCCL.clear();
  AllReduceNCCL = nullptr;
  DupNCCL = nullptr;
  
  for (ncclComm_t& c : allocedNCCL)
    ncclCommDestroy(c);
  allocedNCCL.clear();
}

int ColCommNCCL::set_device(MPI_Comm world) {
  int mpi_rank;
  MPI_Comm_rank(world, &mpi_rank);
  int num_device;
  int gpu_avail = (cudaGetDeviceCount(&num_device) == cudaSuccess);
  int device = mpi_rank % num_device;
  if (gpu_avail)
    return cudaSetDevice(device) == cudaSuccess;
  else
    return 0;
}

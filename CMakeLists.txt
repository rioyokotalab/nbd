
cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -m64 -Wall -Wextra")

project ("shimo" LANGUAGES CXX CUDA)

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_lp64)
set(MKL_THREADING sequential)
set(MKL_MPI intelmpi)

find_package(MKL REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(lorasp lorasp.cpp linalg.cpp basis.cpp comm.cpp build_tree.cpp umv.cpp gpu_linalg.cu)
target_include_directories(lorasp PRIVATE ./)
target_link_libraries(lorasp MPI::MPI_CXX OpenMP::OpenMP_CXX MKL::MKL -L$ENV{CUDADIR}/lib64 -lcublas -lcudart -lcusolver -lnccl)
set_target_properties(lorasp PROPERTIES CUDA_ARCHITECTURES "80")

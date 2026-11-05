cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project ("nbd" LANGUAGES CXX)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(main.app lorasp.cpp)

target_include_directories(main.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(main.app BLAS::BLAS LAPACK::LAPACK -llapacke MPI::MPI_CXX CUDA::cudart CUDA::cublas CUDA::cusolver -lnccl)

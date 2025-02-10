cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project ("shimo" LANGUAGES CXX CUDA)

find_package(BLAS REQUIRED)
find_package(MPI REQUIRED)

file(GLOB sources "*.cpp" "*.cu")
file(GLOB headers "*.hpp")

add_executable(main.app ${sources} ${headers})

target_include_directories(main.app PRIVATE ./)
target_compile_definitions(main.app PRIVATE MKL_ILP64 MKL_Complex16=std::complex<double>)
target_link_libraries(main.app MPI::MPI_CXX BLAS::BLAS -llapacke -L$ENV{CUDADIR}/lib64 -lcublas -lcudart -lcusolver -lnccl)
set_target_properties(main.app PROPERTIES CUDA_ARCHITECTURES "80")

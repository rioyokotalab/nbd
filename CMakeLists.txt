cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project ("shimo" LANGUAGES CXX)

set(BLA_VENDOR OpenBLAS)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MAGMA REQUIRED magma)

file(GLOB sources "*.cpp")
file(GLOB headers "*.hpp")

add_executable(main.app ${sources} ${headers})

target_compile_definitions(main.app PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
target_include_directories(main.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${MAGMA_INCLUDE_DIRS})
target_link_libraries(main.app BLAS::BLAS LAPACK::LAPACK -llapacke MPI::MPI_CXX Eigen3::Eigen ${MAGMA_LINK_LIBRARIES} CUDA::cusolver -lnccl)

cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_BUILD_TYPE Release)

project ("nbd" LANGUAGES CXX)
message(STATUS "C++ Compiler is: " ${CMAKE_CXX_COMPILER_ID})

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Hyacin REQUIRED PATHS "~/cmake_install/lib/cmake")

add_executable(real.app lorasp.cpp)
add_executable(complex.app basis_builder.cpp build_tree.cpp comm-mpi.cpp csr.cpp dense.cpp h2matrix.cpp h2matrix.cpp main.cpp solver.cpp svd_basis.cpp)

target_link_libraries(real.app BLAS::BLAS LAPACK::LAPACK -llapacke MPI::MPI_CXX CUDA::cudart CUDA::cublas CUDA::cusolver -lnccl)
target_link_libraries(complex.app BLAS::BLAS LAPACK::LAPACK MPI::MPI_CXX Hyacin::hyacin)
target_include_directories(complex.app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(real.app PRIVATE "-Wall" "-Wextra" "-fexceptions" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>" "$<$<CONFIG:RELEASE>:-O3>")
  target_compile_options(complex.app PRIVATE "-Wall" "-Wextra" "-fexceptions" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>" "$<$<CONFIG:RELEASE>:-O3>")
endif()


cmake_minimum_required (VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project ("shimo" LANGUAGES CXX)

set(MKL_ARCH intel64)
set(MKL_LINK dynamic)
set(MKL_INTERFACE_FULL intel_ilp64)
set(MKL_THREADING sequential)

find_package(MKL REQUIRED)
find_package(MPI REQUIRED)

file(GLOB sources "*.cpp")
file(GLOB headers "*.hpp")

add_executable(main.app ${sources} ${headers})

target_include_directories(main.app PRIVATE ./)
target_compile_definitions(main.app PRIVATE MKL_ILP64 MKL_Complex16=std::complex<double>)
target_link_libraries(main.app MPI::MPI_CXX MKL::MKL)

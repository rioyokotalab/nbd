

#include "solver.hxx"
#include "dist.hxx"
#include "h2mv.hxx"
#include "minblas.h"

#include <random>
#include <cstdio>
#include <cmath>
#include <iostream>

#include "mpi.h"

using namespace nbd;

int main(int argc, char* argv[]) {

  initComm(&argc, &argv);

  int64_t Nbody = atol(argv[1]);
  int64_t Ncrit = atol(argv[2]);
  int64_t theta = atol(argv[3]);
  int64_t dim = atol(argv[4]);
  
  EvalFunc ef = dim == 2 ? l2d() : l3d();

  std::srand(100);
  std::vector<double> R(1 << 16);
  for (int64_t i = 0; i < R.size(); i++)
    R[i] = -1. + 2. * ((double)std::rand() / RAND_MAX);

  std::vector<double> my_min(dim + 1, 0.);
  std::vector<double> my_max(dim + 1, 1.);

  Bodies body(Nbody);
  randomBodies(body, Nbody, &my_min[0], &my_max[0], dim, 1234);
  Cells cell;
  int64_t levels = buildTree(cell, body, Ncrit, &my_min[0], &my_max[0], dim);
  traverse(cell, levels, dim, theta);
  const Cell* lcleaf = &cell[0];
  lcleaf = findLocalAtLevel(lcleaf, levels);

  std::vector<CSC> rels(levels + 1);
  relationsNear(&rels[0], cell);

  Matrices A(rels[levels].NNZ);
  evaluateLeafNear(A, ef, &cell[0], dim, rels[levels]);

  SpDense sp;
  allocSpDense(sp, &rels[0], levels);
  MPI_Barrier();
  double start_time = std::chrono::system_clock::now();
  factorSpDense(sp, lcleaf, A, 200, &R[0], R.size());
  MPI_Barrier();
  double stop_time = std::chrono::system_clock::now();
  double factor_time_process = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);

  double total_time;
  MPI_Reduce(&factor_time_process, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  int mpi_size;
  MPI_Comm_Size(MPI_COMM_WORLD, &mpi_size);
  total_time = total_time / mpi_size; 

  Vectors X, Xref;
  loadX(X, lcleaf, levels);
  loadX(Xref, lcleaf, levels);

  RHSS rhs(levels + 1);

  MPI_Barrier();
  double start_solve = std::chrono::system_clock::now();
  solveSpDense(&rhs[0], sp, X);
  MPI_Barrier();
  double stop_solve = std::chrono::system_clock::now();
  double solve_time_process = std::chrono::duration_cast<std::chrono::milliseconds>(stop_solve - start_solve);
  double total_solve_time = 0;
  MPI_Reduce(&factor_time_process, &total_solve_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  total_solve_time = total_solve_time / mpi_size;

  DistributeVectorsList(rhs[levels].X, levels);
  for (int64_t i = 0; i < X.size(); i++)
    zeroVector(X[i]);
  closeQuarter(X, rhs[levels].X, ef, lcleaf, dim, levels);

  int64_t mpi_rank;
  commRank(&mpi_rank, NULL, NULL);
  double err;
  solveRelErr(&err, X, Xref, levels);
  printf("%lld ERR: %e\n", mpi_rank, err);

  int64_t* flops = getFLOPS();
  double gf = flops[0] * 1.e-9;
  printf("%lld GFLOPS: %f\n", mpi_rank, gf);

  if (mpi_rank == 0) {
    std::cout << Nbody << "," << Ncrit << "," << theta << "," << dim
	    << "," << mpi_size << "," << total_factor_size << ","
	    << "," << total_solve_size << std::endl;
	    
  }
  closeComm();
  return 0;
}

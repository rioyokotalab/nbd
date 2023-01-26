
#include "nbd.h"
#include "profile.h"

#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  init_batch_lib();

  double prog_time = MPI_Wtime();

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1e0;
  int64_t leaf_size = argc > 3 ? atol(argv[3]) : 256;
  double epi = argc > 4 ? atof(argv[4]) : 1e-10;
  int64_t rank_max = argc > 5 ? atol(argv[5]) : 100;
  int64_t sp_pts = argc > 6 ? atol(argv[6]) : 2000;
  const char* fname = argc > 7 ? argv[7] : NULL;

  int64_t levels = (int64_t)log2((double)Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  int64_t ncells = Nleaf + Nleaf - 1;
  
  //void(*ef)(double*) = laplace3d;
  void(*ef)(double*) = yukawa3d;
  set_kernel_constants(1.e-3 / Nbody, 1.);
  
  double* body = (double*)malloc(sizeof(double) * Nbody * 3);
  double* Xbody = (double*)malloc(sizeof(double) * Nbody);
  struct Cell* cell = (struct Cell*)malloc(sizeof(struct Cell) * ncells);
  struct CellBasis* cell_basis = (struct CellBasis*)malloc(sizeof(struct CellBasis) * ncells);
  struct CSC cellNear, cellFar;
  struct CSC* rels_far = (struct CSC*)malloc(sizeof(struct CSC) * (levels + 1));
  struct CSC* rels_near = (struct CSC*)malloc(sizeof(struct CSC) * (levels + 1));
  struct CellComm* cell_comm = (struct CellComm*)malloc(sizeof(struct CellComm) * (levels + 1));
  struct Base* basis = (struct Base*)malloc(sizeof(struct Base) * (levels + 1));
  struct Node* nodes = (struct Node*)malloc(sizeof(struct Node) * (levels + 1));
  struct RightHandSides* rhs = (struct RightHandSides*)malloc(sizeof(struct RightHandSides) * (levels + 1));

  if (fname == NULL) {
    mesh_unit_sphere(body, Nbody);
    //mesh_unit_cube(body, Nbody);
    //uniform_unit_cube(body, Nbody, 3, 1234);
    buildTree(&ncells, cell, body, Nbody, levels);
  }
  else {
    int64_t* buckets = (int64_t*)malloc(sizeof(int64_t) * Nleaf);
    read_sorted_bodies(&Nbody, Nleaf, body, buckets, fname);
    buildTreeBuckets(cell, body, buckets, levels);
    free(buckets);
  }
  body_neutral_charge(Xbody, Nbody, 1., 999);

  int64_t body_local[2];
  local_bodies(body_local, ncells, cell, levels);
  int64_t lenX = body_local[1] - body_local[0];
  double* X1 = (double*)malloc(sizeof(double) * lenX);
  double* X2 = (double*)malloc(sizeof(double) * lenX);

  traverse('N', &cellNear, ncells, cell, theta);
  traverse('F', &cellFar, ncells, cell, theta);

  buildComm(cell_comm, ncells, cell, &cellFar, &cellNear, levels);
  relations(rels_near, ncells, cell, &cellNear, levels, cell_comm);
  relations(rels_far, ncells, cell, &cellFar, levels, cell_comm);

  double construct_time, construct_comm_time;
  startTimer(&construct_time, &construct_comm_time);
  buildCellBasis(epi, rank_max, sp_pts, ef, cell_basis, ncells, cell, Nbody, body, &cellNear, levels, cell_comm);
  stopTimer(&construct_time, &construct_comm_time);

  double* Workspace = NULL;
  int64_t Lwork = 0;
  buildBasis(8, basis, ncells, cell, cell_basis, levels, cell_comm);
  allocNodes(nodes, &Workspace, &Lwork, basis, rels_near, rels_far, cell_comm, levels);

  evalD(ef, nodes[levels].A, ncells, cell, body, &cellNear, levels);
  for (int64_t i = 0; i <= levels; i++)
    evalS(ef, nodes[i].S, &basis[i], body, &rels_far[i], &cell_comm[i]);

  if (Nbody > 10000) {
    loadX(X1, body_local, Xbody);
    allocRightHandSides('M', rhs, basis, levels);
    matVecA(rhs, nodes, basis, rels_near, rels_far, X1, cell_comm, levels);
    for (int64_t i = 0; i <= levels; i++)
      rightHandSides_free(&rhs[i]);
  }
  else 
    mat_vec_reference(ef, body_local[0], body_local[1], X1, Nbody, body, Xbody);
  
  factorA_mov_mem('S', nodes, basis, levels);
  double factor_time, factor_comm_time;
  startTimer(&factor_time, &factor_comm_time);
  factorA(nodes, basis, cell_comm, levels);
  stopTimer(&factor_time, &factor_comm_time);
  factorA_mov_mem('G', nodes, basis, levels);

  int64_t factor_flops[3];
  get_factor_flops(factor_flops);
  int64_t sum_flops = factor_flops[0] + factor_flops[1] + factor_flops[2];
  double percent[3];
  for (int i = 0; i < 3; i++)
    percent[i] = (double)factor_flops[i] / (double)sum_flops * (double)100;

  allocRightHandSides('S', rhs, basis, levels);

  double solve_time, solve_comm_time;
  startTimer(&solve_time, &solve_comm_time);
  solveA(rhs, nodes, basis, rels_near, X1, cell_comm, levels);
  stopTimer(&solve_time, &solve_comm_time);

  loadX(X2, body_local, Xbody);
  double err;
  solveRelErr(&err, X1, X2, lenX);

  int64_t mem_basis, mem_A, mem_X;
  basis_mem(&mem_basis, basis, levels);
  node_mem(&mem_A, nodes, levels);
  rightHandSides_mem(&mem_X, rhs, levels);

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  prog_time = MPI_Wtime() - prog_time;
  double cm_time;
  getCommTime(&cm_time);

  if (mpi_rank == 0)
    printf("LORASP: %d,%d,%lf,%d,%d,%d\nConstruct: %lf s. COMM: %lf s.\n"
      "Factorize: %lf s. COMM: %lf s.\n"
      "Solution: %lf s. COMM: %lf s.\n"
      "Factorization GFLOPS: %lf GFLOPS/s.\n"
      "GEMM: %lf%%, POTRF: %lf%%, TRSM: %lf%%\n"
      "Basis Memory: %lf GiB.\n"
      "Matrix Memory: %lf GiB.\n"
      "Vector Memory: %lf GiB.\n"
      "Err: %e\n"
      "Program: %lf s. COMM: %lf s.\n",
      (int)Nbody, (int)(Nbody / Nleaf), theta, 3, (int)mpi_size, omp_get_max_threads(),
      construct_time, construct_comm_time, factor_time, factor_comm_time, solve_time, solve_comm_time, (double)sum_flops * 1.e-9 / factor_time,
      percent[0], percent[1], percent[2], (double)mem_basis * 1.e-9, (double)mem_A * 1.e-9, (double)mem_X * 1.e-9, err, prog_time, cm_time);

  for (int64_t i = 0; i <= levels; i++) {
    csc_free(&rels_far[i]);
    csc_free(&rels_near[i]);
    basis_free(&basis[i]);
    node_free(&nodes[i]);
    rightHandSides_free(&rhs[i]);
    cellComm_free(&cell_comm[i]);
  }
  for (int64_t i = 0; i < ncells; i++)
    cellBasis_free(&cell_basis[i]);
  csc_free(&cellFar);
  csc_free(&cellNear);
  
  free(body);
  free(Xbody);
  free(cell);
  free(cell_basis);
  free(rels_far);
  free(rels_near);
  free(cell_comm);
  free(basis);
  free(nodes);
  free(rhs);
  free(X1);
  free(X2);
  set_work_size(0, &Workspace, &Lwork);

  finalize_batch_lib();
  MPI_Finalize();
  return 0;
}

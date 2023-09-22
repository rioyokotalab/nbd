
#include "geometry.hxx"
#include "kernel.hxx"
#include "nbd.hxx"
#include "profile.hxx"

#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]) {
  cudaStream_t stream = init_libs(&argc, &argv);

  double prog_time = MPI_Wtime();

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1e0;
  int64_t leaf_size = argc > 3 ? atol(argv[3]) : 256;
  double epi = argc > 4 ? atof(argv[4]) : 1e-10;
  int64_t rank_max = argc > 5 ? atol(argv[5]) : 100;
  int64_t sp_pts = argc > 6 ? atol(argv[6]) : 2000;
  const char* fname = argc > 7 ? argv[7] : NULL;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  int64_t levels = (int64_t)log2((double)Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  int64_t ncells = Nleaf + Nleaf - 1;
  
  Laplace3D eval(1.e-6);
  //Yukawa3D eval(1.e-6, 1.);
  //Gaussian eval(0.2);
  
  double* body = (double*)malloc(sizeof(double) * Nbody * 3);
  double* Xbody = (double*)malloc(sizeof(double) * Nbody);
  struct Cell* cell = (struct Cell*)malloc(sizeof(struct Cell) * ncells);
  struct CSC cellNear, cellFar;
  struct CSC* rels_far = (struct CSC*)malloc(sizeof(struct CSC) * (levels + 1));
  struct CSC* rels_near = (struct CSC*)malloc(sizeof(struct CSC) * (levels + 1));
  struct CellComm* cell_comm = (struct CellComm*)calloc(levels + 1, sizeof(struct CellComm));
  struct Base* basis = (struct Base*)calloc(levels + 1, sizeof(struct Base));
  struct Node* nodes = (struct Node*)malloc(sizeof(struct Node) * (levels + 1));

  if (fname == NULL) {
    mesh_unit_sphere(body, Nbody);
    //mesh_unit_cube(body, Nbody);
    //uniform_unit_cube(body, Nbody, 1);
    double c[3] = { 0, 0, 0 };
    double r[3] = { 1, 1, 1 };
    magnify_reloc(body, Nbody, c, c, r, sqrt(Nbody));
    buildTree(&ncells, cell, body, Nbody, levels);
  }
  else {
    int64_t* buckets = (int64_t*)malloc(sizeof(int64_t) * Nleaf);
    read_sorted_bodies(&Nbody, Nleaf, body, buckets, fname);
    //buildTreeBuckets(cell, body, buckets, levels);
    buildTree(&ncells, cell, body, Nbody, levels);
    free(buckets);
  }
  body_neutral_charge(Xbody, Nbody, 1., 999);

  traverse('N', &cellNear, ncells, cell, theta);
  traverse('F', &cellFar, ncells, cell, theta);

  struct CommTimer timer;
  buildComm(cell_comm, ncells, cell, &cellFar, &cellNear, levels);
  for (int64_t i = 0; i <= levels; i++) {
    cell_comm[i].stream = stream;
    cell_comm[i].timer = &timer;
  }
  relations(rels_near, &cellNear, levels, cell_comm);
  relations(rels_far, &cellFar, levels, cell_comm);

  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &cell_comm[levels]);
  int64_t gbegin = lbegin;
  i_global(&gbegin, &cell_comm[levels]);

  MPI_Barrier(MPI_COMM_WORLD);
  double construct_time = MPI_Wtime(), construct_comm_time;
  buildBasis(eval, basis, cell, &cellNear, levels, cell_comm, body, Nbody, epi, rank_max, sp_pts, 4);

  MPI_Barrier(MPI_COMM_WORLD);
  construct_time = MPI_Wtime() - construct_time;
  construct_comm_time = timer.get_comm_timing();

  double* Workspace = NULL;
  int64_t Lwork = 0;
  allocNodes(nodes, &Workspace, &Lwork, basis, rels_near, rels_far, cell_comm, levels);

  evalD(eval, nodes[levels].A, &cellNear, cell, body, &cell_comm[levels]);
  for (int64_t i = 0; i <= levels; i++)
    evalS(eval, nodes[i].S, &basis[i], &rels_far[i], &cell_comm[i]);

  int64_t lenX = rels_near[levels].N * basis[levels].dimN;
  double* X1 = (double*)calloc(lenX, sizeof(double));
  double* X2 = (double*)calloc(lenX, sizeof(double));

  loadX(X1, basis[levels].dimN, Xbody, 0, llen, &cell[gbegin]);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  matVecA(nodes, basis, rels_near, X1, cell_comm, levels);

  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = timer.get_comm_timing();

  double cerr = 0.;
  if (Nbody < 20000) {
    int64_t body_local[2] = { cell[gbegin].Body[0], cell[gbegin + llen - 1].Body[1] };
    std::vector<double> X3(lenX, 0);
    mat_vec_reference(eval, body_local[0], body_local[1], &X3[0], Nbody, body, Xbody);
    loadX(X2, basis[levels].dimN, &X3[0], body_local[0], llen, &cell[gbegin]);

    solveRelErr(&cerr, X1, X2, lenX);
    std::iter_swap(&X1, &X2);
  }
  
  factorA_mov_mem('S', nodes, basis, levels);
  MPI_Barrier(MPI_COMM_WORLD);
  double factor_time = MPI_Wtime(), factor_comm_time;

  for (int64_t i = levels; i > 0; i--)
    batchCholeskyFactor(&nodes[i].params, &cell_comm[i]);
  chol_decomp(&nodes[0].params, &cell_comm[0]);

  cudaStreamSynchronize(stream);
  MPI_Barrier(MPI_COMM_WORLD);

  factor_time = MPI_Wtime() - factor_time;
  factor_comm_time = timer.get_comm_timing();

  Profile profile;
  for (int64_t i = 1; i <= levels; i++)
    profile.record_factor(basis[i].dimR, basis[i].dimN, nodes[i].params.L_nnz, nodes[i].params.L_diag, nodes[i].params.L_rows);
  profile.record_factor(basis[0].dimR, basis[0].dimN, 1, 1, 1);
  
  int64_t factor_flops[4], mem_A[3];
  profile.get_profile(factor_flops, mem_A);
  MPI_Allreduce(MPI_IN_PLACE, factor_flops, 4, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  
  int64_t sum_flops = factor_flops[0] + factor_flops[1] + factor_flops[2];
  double percent[3];
  for (int i = 0; i < 3; i++)
    percent[i] = (double)factor_flops[i] / (double)sum_flops * (double)100;

  cudaMemcpy(&nodes[levels].X_ptr[lbegin * basis[levels].dimN], X1, lenX * sizeof(double), cudaMemcpyHostToDevice);

  MPI_Barrier(MPI_COMM_WORLD);
  double solve_time = MPI_Wtime(), solve_comm_time;

  for (int64_t i = levels; i > 0; i--)
    batchForwardULV(&nodes[i].params, &cell_comm[i]);
  chol_solve(&nodes[0].params, &cell_comm[0]);
  for (int64_t i = 1; i <= levels; i++)
    batchBackwardULV(&nodes[i].params, &cell_comm[i]);

  cudaStreamSynchronize(stream);
  MPI_Barrier(MPI_COMM_WORLD);

  solve_time = MPI_Wtime() - solve_time;
  solve_comm_time = timer.get_comm_timing();

  cudaMemcpy(X1, &nodes[levels].X_ptr[lbegin * basis[levels].dimN], lenX * sizeof(double), cudaMemcpyDeviceToHost);

  loadX(X2, basis[levels].dimN, Xbody, 0, llen, &cell[gbegin]);
  double err;
  solveRelErr(&err, X1, X2, lenX);

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  prog_time = MPI_Wtime() - prog_time;

  if (mpi_rank == 0)
    printf("LORASP: %d,%d,%lf,%d,%d,%d\n"
      "Construct: %lf s. COMM: %lf s.\n"
      "Mat-Vec: %lf s. COMM: %lf s.\n"
      "Factorize: %lf s. COMM: %lf s.\n"
      "Solution: %lf s. COMM: %lf s.\n"
      "Factorization GFLOPS: %lf GFLOPS/s.\n"
      "GEMM: %lf%%, POTRF: %lf%%, TRSM: %lf%%\n"
      "Pre-fac vs. Actual: %lf%%, %lf%% \n"
      "Matrix Memory: %lf GiB.\n"
      "Basis Memory: %lf GiB.\n"
      "Vector Memory: %lf GiB.\n"
      "Err: Compress %e, Factor %e\n"
      "Program: %lf s.\n",
      (int)Nbody, (int)(Nbody / Nleaf), theta, 3, (int)mpi_size, omp_get_max_threads(),
      construct_time, construct_comm_time, matvec_time, matvec_comm_time, factor_time, factor_comm_time, 
      solve_time, solve_comm_time, (double)sum_flops * 1.e-9 / factor_time, percent[0], percent[1], percent[2], 
      (double)100 * factor_flops[3] / (sum_flops + factor_flops[3]), (double)100 * sum_flops / (sum_flops + factor_flops[3]),
      (double)mem_A[0] * 1.e-9, (double)mem_A[1] * 1.e-9, (double)mem_A[2] * 1.e-9, cerr, err, prog_time);

  for (int64_t i = 0; i <= levels; i++) {
    csc_free(&rels_far[i]);
    csc_free(&rels_near[i]);
    basis_free(&basis[i]);
    node_free(&nodes[i]);
  }
  cellComm_free(cell_comm, levels);
  csc_free(&cellFar);
  csc_free(&cellNear);
  
  free(body);
  free(Xbody);
  free(cell);
  free(rels_far);
  free(rels_near);
  free(cell_comm);
  free(basis);
  free(nodes);
  free(X1);
  free(X2);
  set_work_size(0, &Workspace, &Lwork);

  fin_libs();
  return 0;
}

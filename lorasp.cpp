
#include <basis.hpp>
#include <build_tree.hpp>
#include <comm.hpp>
#include <gpu_linalg.hpp>
#include <linalg.hpp>
#include <umv.hpp>
#include <geometry.hpp>
#include <profile.hpp>

#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]) {
  cudaStream_t stream = (cudaStream_t)init_libs(&argc, &argv);

  double prog_time = MPI_Wtime();

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1e0;
  int64_t leaf_size = argc > 3 ? atol(argv[3]) : 256;
  int64_t rank_max = argc > 4 ? atol(argv[4]) : 100;
  int64_t sp_pts = argc > 5 ? atol(argv[5]) : Nbody;
  const char* fname = argc > 6 ? argv[6] : NULL;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  int64_t levels = (int64_t)log2((double)Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  int64_t ncells = Nleaf + Nleaf - 1;
  
  double* body = (double*)malloc(sizeof(double) * Nbody * 3);
  Cell* cell = (Cell*)calloc(ncells, sizeof(Cell));
  CSR cellNear, cellFar;
  CSR* rels_far = (CSR*)calloc(levels + 1, sizeof(CSR));
  CSR* rels_near = (CSR*)calloc(levels + 1, sizeof(CSR));
  CellComm* cell_comm = (CellComm*)calloc(levels + 1, sizeof(CellComm));
  Base* basis = (Base*)calloc(levels + 1, sizeof(Base));
  Node* nodes = (Node*)malloc(sizeof(Node) * (levels + 1));

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
    buildTree(&ncells, cell, body, Nbody, levels);
    free(buckets);
  }

  Eigen::VectorXd Xbody(Nbody);
  body_neutral_charge(Xbody.data(), Nbody, 1., 999);
  Eigen::VectorXcd vecX = Xbody;

  traverse('N', &cellNear, ncells, cell, theta);
  traverse('F', &cellFar, ncells, cell, theta);

  Laplace3D eval(1.e-1);
  Eigen::MatrixXd dataA(Nbody, Nbody);
  gen_matrix(eval, Nbody, Nbody, body, body, dataA.data(), Nbody);

  DenseZMat denseA(Nbody, Nbody);
  Eigen::Map<Eigen::MatrixXcd>(denseA.A, Nbody, Nbody) = dataA;

  CommTimer timer;
  buildComm(cell_comm, ncells, cell, &cellFar, &cellNear, levels);
  for (int64_t i = 0; i <= levels; i++) {
    cell_comm[i].stream = stream;
    cell_comm[i].timer = &timer;
  }
  relations(rels_near, &cellNear, levels, cell_comm);
  relations(rels_far, &cellFar, levels, cell_comm);

  std::vector<Hmatrix> hA;
  hA.reserve(levels + 1);

  for (int64_t i = 0; i <= levels; i++) {
    int64_t lbegin = 0, llen = 0;
    content_length(&llen, NULL, &lbegin, &cell_comm[i]);
    int64_t gbegin = cell_comm[i].iGlobal(lbegin);
    hA.emplace_back(1.e-10, denseA, rank_max, rank_max * 2, 2, gbegin, llen, cell, cellFar);
  }

  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &cell_comm[levels]);
  int64_t gbegin = cell_comm[levels].iGlobal(lbegin);
  int64_t body_local[2] = { cell[gbegin].Body[0], cell[gbegin + llen - 1].Body[1] };

  Hmatrix blA(denseA, gbegin, llen, cell, cellNear);

  MPI_Barrier(MPI_COMM_WORLD);
  double construct_time = MPI_Wtime(), construct_comm_time;
  buildBasis(eval, basis, cell, &cellNear, levels, cell_comm, body, Nbody, rank_max, sp_pts, 4);

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
  Eigen::VectorXd X1(lenX), X2(lenX);
  Eigen::VectorXcd X3 = Eigen::VectorXcd::Zero(lenX);

  double matvec_time = MPI_Wtime(), matvec_comm_time;
  for (auto& h : hA)
    h.matVecMul(body_local[1] - body_local[0], Nbody, 1, body_local[0], 0, vecX.data(), Nbody, X3.data(), lenX);
  blA.matVecMul(body_local[1] - body_local[0], Nbody, 1, body_local[0], 0, vecX.data(), Nbody, X3.data(), lenX);

  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = timer.get_comm_timing();
  X2 = X3.real();
  X3.setZero();

  double cerr = 0.;
  denseA.op_Aij_mulB('N', body_local[1] - body_local[0], Nbody, 1, body_local[0], 0, vecX.data(), Nbody, X3.data(), lenX);
  X1 = X3.real();
  solveRelErr(&cerr, X1.data(), X2.data(), lenX);
  
  factorA_mov_mem(nodes, levels);
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

  cudaMemcpy(&nodes[levels].X_ptr[lbegin * basis[levels].dimN], X1.data(), lenX * sizeof(double), cudaMemcpyHostToDevice);

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

  cudaMemcpy(X1.data(), &nodes[levels].X_ptr[lbegin * basis[levels].dimN], lenX * sizeof(double), cudaMemcpyDeviceToHost);

  loadX(X2.data(), basis[levels].dimN, Xbody.data(), 0, llen, &cell[gbegin]);
  double err;
  solveRelErr(&err, X1.data(), X2.data(), lenX);

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  prog_time = MPI_Wtime() - prog_time;

  if (mpi_rank == 0)
    printf("LORASP: %d,%d,%lf,%d,%d\n"
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
      (int)Nbody, (int)(Nbody / Nleaf), theta, 3, (int)mpi_size,
      construct_time, construct_comm_time, matvec_time, matvec_comm_time, factor_time, factor_comm_time, 
      solve_time, solve_comm_time, (double)sum_flops * 1.e-9 / factor_time, percent[0], percent[1], percent[2], 
      (double)100 * factor_flops[3] / (sum_flops + factor_flops[3]), (double)100 * sum_flops / (sum_flops + factor_flops[3]),
      (double)mem_A[0] * 1.e-9, (double)mem_A[1] * 1.e-9, (double)mem_A[2] * 1.e-9, cerr, err, prog_time);

  for (int64_t i = 0; i <= levels; i++) {
    basis_free(&basis[i]);
    node_free(&nodes[i]);
  }
  cellComm_free(cell_comm, levels);
  
  free(body);
  free(cell);
  free(rels_far);
  free(rels_near);
  free(cell_comm);
  free(basis);
  free(nodes);
  set_work_size(0, &Workspace, &Lwork);

  fin_libs();
  return 0;
}

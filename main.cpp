
#include <basis_builder.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <csr.hpp>
#include <dense.hpp>
#include <h2matrix.hpp>
#include <solver.hpp>
#include <svd_basis.hpp>

#include <test_funcs.hpp>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  svd_lib_init();

  long long Nbody = argc > 1 ? std::atoll(argv[1]) : 2048;
  long long leaf_size = argc > 2 ? std::atoll(argv[2]) : 256;
  double epi = argc > 3 ? std::atof(argv[3]) : 1e-10;

  leaf_size = Nbody < leaf_size ? Nbody : leaf_size;
  long long levels = (long long)std::log2((double)Nbody / leaf_size);
  long long Nleaf = (long long)1 << levels;
  long long ncells = Nleaf + Nleaf - 1;

  int mpi_rank = 0;
  MPI_Comm world = MPI_COMM_NULL;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_split(MPI_COMM_WORLD, mpi_rank < Nleaf ? 1 : MPI_UNDEFINED, mpi_rank, &world);

  if (world == MPI_COMM_NULL) 
  { MPI_Finalize(); return 0; }
  
  //Laplace3D eval(1.);
  //Yukawa3D eval(1, 1.);
  //Gaussian eval(0.005);
  Helmholtz3D eval(4., 1.E0);
  
  std::vector<double> body(Nbody * 3);
  std::vector<std::complex<double>> Xbody(Nbody);
  std::vector<Cell> cell(ncells);

  mesh_sphere(&body[0], Nbody);
  //uniform_unit_cube_rnd(&body[0], Nbody, 3, 999);
  //uniform_unit_cube(&body[0], Nbody, 3);
  buildBinaryTree(&cell[0], &body[0], Nbody, levels);

  std::mt19937_64 gen;
  std::uniform_real_distribution<double> uniform_dist(0., 1.);
  std::generate(Xbody.begin(), Xbody.end(), 
    [&]() { return std::complex<double>(uniform_dist(gen), 0.); });

  /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
  cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
  ncells = Nleaf + 1;
  levels = 1;*/

  CSR Near('N', cell, cell), Far('F', cell, cell);
  H2MatrixSolver matA(cell, Near, Far, levels, world);
  long long lenX = matA.local_bodies.second - matA.local_bodies.first;
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));

  DenseMatrix denseA(matA.local_bodies.first, matA.local_bodies.second, Nbody, world);
  gen_matrix(eval, lenX, Nbody, &body[3 * matA.local_bodies.first], &body[0], denseA.A.data());

  MPI_Barrier(world);
  double h2_construct_time = MPI_Wtime(), h2_construct_comm_time;
  matA.evalMatVecA(denseA, epi, cell.data(), Far);

  MPI_Barrier(world);
  h2_construct_time = MPI_Wtime() - h2_construct_time;
  h2_construct_comm_time = H2MatrixComm::get_comm_time(world);

  std::copy(&Xbody[matA.local_bodies.first], &Xbody[matA.local_bodies.second], &X2[0]);
  double refmatvec_time = MPI_Wtime();
  denseA.matVecA(&X2[0]);
  refmatvec_time = MPI_Wtime() - refmatvec_time;

  std::copy(&Xbody[matA.local_bodies.first], &Xbody[matA.local_bodies.second], &X1[0]);
  MPI_Barrier(world);
  double matvec_time = MPI_Wtime(), matvec_comm_time;
  matA.matVecMul(&X1[0]);

  MPI_Barrier(world);
  matvec_time = MPI_Wtime() - matvec_time;
  matvec_comm_time = H2MatrixComm::get_comm_time(world);
  double cerr = matA.solveRelErr(&X1[0], &X2[0]);

  if (mpi_rank == 0) {
    int mpi_size = 1;
    MPI_Comm_size(world, &mpi_size);
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Active MPI processes: " << mpi_size << std::endl;
    std::cout << "Matrix Order: " << Nbody << " Leaf-size: " << leaf_size << std::endl;
    std::cout << "HSS-Matvec Target accuracy: " << epi << std::endl;
    std::cout << "Construct Err: " << cerr << std::endl;
    std::cout << "HSS-Matrix Construct Time: (tot) " << h2_construct_time << " (comm) " << h2_construct_comm_time << std::endl;
    std::cout << "HSS-Matvec Time: (tot) " << matvec_time << " (comm) " << matvec_comm_time << std::endl;
    std::cout << "Dense Matvec Time: (tot) " << refmatvec_time << std::endl;
    //system("free");
  }

  matA.free_all_comms();
  MPI_Comm_free(&world);
  MPI_Finalize();
  svd_lib_finalize();
  return 0;
}


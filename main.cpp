
#include <dense.hpp>
#include <hss.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class MatrixAccessor {
public:
  virtual std::complex<double> operator()(double d) const = 0;
};

class Laplace3D : public MatrixAccessor {
public:
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::complex<double>(-1. / d, 0.);
  }
};

class Yukawa3D : public MatrixAccessor {
public:
  double singularity, alpha;
  Yukawa3D (double s, double a) : singularity(1. / s), alpha(a) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::complex<double>(std::exp(-alpha * d) / d, 0.);
  }
};

class Gaussian : public MatrixAccessor {
public:
  double alpha;
  Gaussian (double a) : alpha(a) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(1.0001, 0.);
    return std::complex<double>(std::exp(- alpha * d * d), 0.);
  }
};

class Helmholtz3D : public MatrixAccessor {
public:
  double k;
  double singularity;
  Helmholtz3D(double wave_number, double s) : k(wave_number), singularity(1. / s) {}
  std::complex<double> operator()(double d) const override {
    if (d == 0.)
      return std::complex<double>(singularity, 0.);
    else
      return std::complex<double>(-std::cos(k * d), std::sin(k * d)) / d;  
  }
};

void uniform_unit_cube(double* bodies, long long nbodies, long long dim) {
  long long side = std::ceil(std::pow(nbodies, 1. / dim));
  long long lens[3] = { dim > 0 ? side : 1, dim > 1 ? side : 1, dim > 2 ? side : 1 };
  double step = 1. / side;

  for (long long i = 0; i < lens[0]; ++i)
    for (long long j = 0; j < lens[1]; ++j)
       for (long long k = 0; k < lens[2]; ++k) {
    long long x = k + lens[2] * (j + lens[1] * i);
    if (x < nbodies) {
      bodies[x * 3] = i * step;
      bodies[x * 3 + 1] = j * step;
      bodies[x * 3 + 2] = k * step;
    }
  }
}

void uniform_unit_cube_rnd(double* bodies, long long nbodies, long long dim, unsigned long long seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> uniform_dist(0., 1.);

  std::array<double, 3>* b3 = reinterpret_cast<std::array<double, 3>*>(bodies);
  std::array<double, 3>* b3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[3 * nbodies]);
  std::for_each(b3, b3_end, [&](std::array<double, 3>& body) {
    for (int i = 0; i < 3; i++)
      body[i] = i < dim ? uniform_dist(gen) : 0.;
  });
}

void mesh_sphere(double* bodies, long long nbodies) {
  const double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians

  for (long long i = 0; i < nbodies; ++i) {
    const double y = 1. - 2. * ((double)i / (double)(nbodies - 1));  // y goes from r to -r

    // Note: setting constant radius = 1 will produce a cylindrical shape
    const double radius = std::sqrt(1. - y * y);  // radius at y
    const double theta = (double)i * phi;

    const double x = radius * std::cos(theta);
    const double z = radius * std::sin(theta);
    bodies[i * 3] = x;
    bodies[i * 3 + 1] = y;
    bodies[i * 3 + 2] = z;
  }
}

void gen_matrix(const MatrixAccessor& eval, long long m, long long n, const double* bi, const double* bj, std::complex<double> Aij[]) {
  const std::array<double, 3>* bi3 = reinterpret_cast<const std::array<double, 3>*>(bi);
  const std::array<double, 3>* bi3_end = reinterpret_cast<const std::array<double, 3>*>(&bi[3 * m]);
  const std::array<double, 3>* bj3 = reinterpret_cast<const std::array<double, 3>*>(bj);
  const std::array<double, 3>* bj3_end = reinterpret_cast<const std::array<double, 3>*>(&bj[3 * n]);

  std::for_each(bj3, bj3_end, [&](const std::array<double, 3>& j) -> void {
    long long ix = std::distance(bj3, &j);
    std::for_each(bi3, bi3_end, [&](const std::array<double, 3>& i) -> void {
      long long iy = std::distance(bi3, &i);
      std::array<double, 3> d({ i[0] - j[0], i[1] - j[1], i[2] - j[2] });
      double d_nrm = std::sqrt(std::inner_product(d.begin(), d.end(), d.begin(), 0.));
      Aij[iy + ix * m] = eval(d_nrm);
    });
  });
}

void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels) {
  cells[0].N[0] = 0;
  cells[0].N[1] = nbodies;
  auto get_bounds = [=](double* bodies, long long nbodies, double R[], double C[]) {
    const std::array<double, 3>* bodies3 = (const std::array<double, 3>*)(&bodies[0]);
    const std::array<double, 3>* bodies3_end = (const std::array<double, 3>*)(&bodies[nbodies * 3]);
    double Xmin[3], Xmax[3];
    for (int i = 0; i < 3; i++) {
      auto minmax = std::minmax_element(bodies3, bodies3_end, 
        [=](const std::array<double, 3>& x, const std::array<double, 3>& y) { return x[i] < y[i]; });
      Xmin[i] = (*minmax.first)[i]; Xmax[i] = (*minmax.second)[i];
    }
    std::transform(Xmin, &Xmin[3], Xmax, C, [](double min, double max) { return (min + max) * 0.5; });
    std::transform(Xmin, &Xmin[3], Xmax, R, [](double min, double max) { return (min == max && min == 0.) ? 0. : ((max - min) * 0.5 + 1.e-8); });
  };
  get_bounds(bodies, nbodies, cells[0].R.data(), cells[0].C.data());

  long long nleaf = 1ll << levels;
  for (long long i = 0; i < nleaf - 1; i++) {
    Cell& ci = cells[i];
    long long sdim = std::distance(ci.R.begin(), std::max_element(ci.R.begin(), ci.R.end()));
    long long i_begin = ci.N[0];
    long long i_end = ci.N[1];

    std::array<double, 3>* bodies3 = reinterpret_cast<std::array<double, 3>*>(&bodies[i_begin * 3]);
    std::array<double, 3>* bodies3_end = reinterpret_cast<std::array<double, 3>*>(&bodies[i_end * 3]);
    std::sort(bodies3, bodies3_end, 
      [=](std::array<double, 3>& i, std::array<double, 3>& j) { return i[sdim] < j[sdim]; });

    long long len = (i << 1) + 1;
    Cell& c0 = cells[len];
    Cell& c1 = cells[len + 1];
    ci.Child[0] = len;
    ci.Child[1] = len + 2;

    long long loc = i_begin + (i_end - i_begin) / 2;
    c0.N[0] = i_begin;
    c0.N[1] = loc;
    c1.N[0] = loc;
    c1.N[1] = i_end;

    get_bounds(&bodies[i_begin * 3], loc - i_begin, c0.R.data(), c0.C.data());
    get_bounds(&bodies[loc * 3], i_end - loc, c1.R.data(), c1.C.data());
  }
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

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
  std::generate(Xbody.begin(), Xbody.end(), [&]() { return std::complex<double>(uniform_dist(gen), uniform_dist(gen)); });

  /*cell.erase(cell.begin() + 1, cell.begin() + Nleaf - 1);
  cell[0].Child[0] = 1; cell[0].Child[1] = Nleaf + 1;
  ncells = Nleaf + 1;
  levels = 1;*/

  HSS matA(cell, levels, world);
  long long lenX = matA.local_bodies.second - matA.local_bodies.first;
  std::vector<std::complex<double>> X1(lenX, std::complex<double>(0., 0.));
  std::vector<std::complex<double>> X2(lenX, std::complex<double>(0., 0.));

  DenseMatrix denseA(lenX, Nbody, world, MPI_COMM_NULL);
  for (long long i = matA.local_bodies.first; i < matA.local_bodies.second; i += 512ll)
    for (long long j = 0; j < Nbody; j += 512ll) {
      long long rows = std::min(matA.local_bodies.second - i, 512ll);
      long long cols = std::min(Nbody - j, 512ll);
      std::vector<std::complex<double>> A(rows * cols);
      gen_matrix(eval, rows, cols, &body[3ll * i], &body[3 * j], &A[0]);
      denseA.AijSetBlock(rows, cols, i, j, &A[0], rows);
  }

  DenseMatrix denseX(lenX, 1, world, MPI_COMM_NULL);
  denseX.AijSetBlock(lenX, 1, matA.local_bodies.first, 0, &Xbody[matA.local_bodies.first], Nbody);
  double refmatvec_time = MPI_Wtime();
  denseX.matVecA(denseA, &X2[0]);
  refmatvec_time = MPI_Wtime() - refmatvec_time;

  MPI_Barrier(world);
  double h2_construct_time = MPI_Wtime();
  matA.evalMatVecA(denseA, epi, cell.data());

  MPI_Barrier(world);
  h2_construct_time = MPI_Wtime() - h2_construct_time;

  std::copy(&Xbody[matA.local_bodies.first], &Xbody[matA.local_bodies.second], &X1[0]);
  MPI_Barrier(world);
  double matvec_time = MPI_Wtime();
  matA.matVecMul(&X1[0]);

  MPI_Barrier(world);
  matvec_time = MPI_Wtime() - matvec_time;
  double cerr = matA.solveRelErr(&X1[0], &X2[0]);

  if (mpi_rank == 0) {
    int mpi_size = 1;
    MPI_Comm_size(world, &mpi_size);
    std::cout << "Active MPI processes: " << mpi_size << std::endl;
    std::cout << "Matrix Order: " << Nbody << " Leaf-size: " << leaf_size << std::endl;
    std::cout << " HSS  Matvec Target accuracy: " << epi << std::endl;
    std::cout << "Construct Err: " << cerr << std::endl;
    std::cout << " HSS  Construct Time: " << h2_construct_time << std::endl;
    std::cout << " HSS  Matvec Time: " << matvec_time << std::endl;
    std::cout << "Dense Matvec Time: " << refmatvec_time << std::endl;
  }

  matA.free_all_comms();
  MPI_Comm_free(&world);
  MPI_Finalize();
  return 0;
}


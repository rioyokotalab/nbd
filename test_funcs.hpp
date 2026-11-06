
#include <complex>
#include <cmath>
#include <random>
#include <array>
#include <algorithm>
#include <iostream>
#include <fstream>

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

void read_sorted_bodies(long long* nbodies, long long lbuckets, double* bodies, long long buckets[], const char* fname) {
  std::ifstream file(fname);

  long long curr = 1, cbegin = 0, iter = 0, len = *nbodies;
  while (iter < len && !file.eof()) {
    long long b = 0;
    double x = 0., y = 0., z = 0.;
    file >> x >> y >> z >> b;

    if (lbuckets < b)
      len = iter;
    else if (!file.eof()) {
      bodies[iter * 3] = x;
      bodies[iter * 3 + 1] = y;
      bodies[iter * 3 + 2] = z;
      while (curr < b && curr <= lbuckets) {
        buckets[curr - 1] = iter - cbegin;
        cbegin = iter;
        curr++;
      }
      iter++;
    }
  }
  while (curr <= lbuckets) {
    buckets[curr - 1] = iter - cbegin;
    cbegin = iter;
    curr++;
  }
  *nbodies = iter;
}

void write_to_csv(const char* fname, int mpi_size, long long N, double theta, long long leaf_size, long long rank, double epi, const char* mode, 
  double h2cerr, double h2ctime, double h2ctime_comm, double h2mvtime, double h2mvtime_comm, double dense_mvtime,
  double mctime, double mctime_comm, double mcerr, double factor_time, double factor_time_comm, double sub_time, double sub_time_comm, double sub_err,
  double gmres_err, double gmres_iters, double gmres_time, double gmres_time_comm, const double* iter_err) {
  
  std::ofstream file(fname, std::ios_base::app);
  if (!file.bad())
  {
    file << mpi_size << ',' << N << ',' << theta << ',' << leaf_size << ',' << rank << ',' << epi << ',' << mode << ','; // 0 - 6
    file << h2cerr << ',' << h2ctime << ',' << h2ctime_comm << ',' << h2mvtime << ',' << h2mvtime_comm << ',' << dense_mvtime << ','; // 7 - 12
    file << mctime << ',' << mctime_comm << ',' << mcerr << ',' << factor_time << ',' << factor_time_comm << ',' << sub_time << ',' << sub_time_comm << ',' << sub_err << ','; // 13 - 20
    file << gmres_err << ',' << gmres_iters << ',' << gmres_time << ',' << gmres_time_comm; // 21 - 24
    for (long long i = 0; i <= gmres_iters; i++)
      file << ',' << iter_err[i];
    file << std::endl;
    file.close();
  }
}

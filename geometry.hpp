
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void uniform_unit_cube(double* bodies, int64_t nbodies, int64_t dim) {
  int64_t side = ceil(pow(nbodies, 1. / dim));
  int64_t lens[3] = { dim > 0 ? side : 1, dim > 1 ? side : 1, dim > 2 ? side : 1 };
  double step = 1. / side;

  for (int64_t i = 0; i < lens[0]; ++i)
    for (int64_t j = 0; j < lens[1]; ++j)
       for (int64_t k = 0; k < lens[2]; ++k) {
    int64_t x = k + lens[2] * (j + lens[1] * i);
    if (x < nbodies) {
      bodies[x * 3] = i * step;
      bodies[x * 3 + 1] = j * step;
      bodies[x * 3 + 2] = k * step;
    }
  }
}

void uniform_unit_cube_rnd(double* bodies, int64_t nbodies, int64_t dim, unsigned int seed) {
  if (seed > 0)
    srand(seed);

  for (int64_t i = 0; i < nbodies; i++) {
    double r0 = dim > 0 ? ((double)rand() / RAND_MAX) : 0.;
    double r1 = dim > 1 ? ((double)rand() / RAND_MAX) : 0.;
    double r2 = dim > 2 ? ((double)rand() / RAND_MAX) : 0.;
    bodies[i * 3] = r0;
    bodies[i * 3 + 1] = r1;
    bodies[i * 3 + 2] = r2;
  }
}

void mesh_unit_cube(double* bodies, int64_t nbodies) {
  if (nbodies < 8) {
    std::cerr << "Error cubic mesh size (GT/EQ. 8 required): %" << nbodies << "." << std::endl;
    return;
  }

  // compute splits: solution to 6x^2 + 12x + 8 = nbodies.
  int64_t x_lower_bound = (int64_t)floor(sqrt(6 * nbodies - 12) / 6 - 1);
  int64_t x_splits[3] = { x_lower_bound, x_lower_bound, x_lower_bound };
  
  for (int64_t i = 0; i < 3; i++) {
    int64_t x = x_splits[0];
    int64_t y = x_splits[1];
    int64_t z = x_splits[2];
    int64_t mesh_points = 8 + 4 * x + 4 * y + 4 * z + 2 * x * y + 2 * x * z + 2 * y * z;
    if (mesh_points < nbodies)
      x_splits[i] = x_splits[i] + 1;
  }

  int64_t lens[7] = { 8, 4 * x_splits[0], 4 * x_splits[1], 4 * x_splits[2],
    2 * x_splits[0] * x_splits[1], 2 * x_splits[0] * x_splits[2], 2 * x_splits[1] * x_splits[2] };

  double segment_x = 2. / (1. + x_splits[0]);
  double segment_y = 2. / (1. + x_splits[1]);
  double segment_z = 2. / (1. + x_splits[2]);

  for (int64_t i = 0; i < nbodies; i++) {
    int64_t region = 0;
    int64_t ri = i;
    while (region < 6 && ri >= lens[region]) {
      ri = ri - lens[region];
      region = region + 1;
    }

    switch (region) {
    case 0: { // Vertex
      bodies[i * 3] = (double)(1 - 2 * ((ri & 4) >> 2));
      bodies[i * 3 + 1] = (double)(1 - 2 * ((ri & 2) >> 1));
      bodies[i * 3 + 2] = (double)(1 - 2 * (ri & 1));
      break;
    }
    case 1: { // edges parallel to X-axis
      bodies[i * 3] = -1 + ((ri >> 2) + 1) * segment_x;
      bodies[i * 3 + 1] = (double)(1 - 2 * ((ri & 2) >> 1));
      bodies[i * 3 + 2] = (double)(1 - 2 * (ri & 1));
      break;
    }
    case 2: { // edges parallel to Y-axis
      bodies[i * 3] = (double)(1 - 2 * ((ri & 2) >> 1));
      bodies[i * 3 + 1] = -1 + ((ri >> 2) + 1) * segment_y;
      bodies[i * 3 + 2] = (double)(1 - 2 * (ri & 1));
      break;
    }
    case 3: { // edges parallel to Z-axis
      bodies[i * 3] = (double)(1 - 2 * ((ri & 2) >> 1));
      bodies[i * 3 + 1] = (double)(1 - 2 * (ri & 1));
      bodies[i * 3 + 2] = -1 + ((ri >> 2) + 1) * segment_z;
      break;
    }
    case 4: { // surface parallel to X-Y plane
      int64_t x = (ri >> 1) / x_splits[1];
      int64_t y = (ri >> 1) - x * x_splits[1];
      bodies[i * 3] = -1 + (x + 1) * segment_x;
      bodies[i * 3 + 1] = -1 + (y + 1) * segment_y;
      bodies[i * 3 + 2] = (double)(1 - 2 * (ri & 1));
      break;
    }
    case 5: { // surface parallel to X-Z plane
      int64_t x = (ri >> 1) / x_splits[2];
      int64_t z = (ri >> 1) - x * x_splits[2];
      bodies[i * 3] = -1 + (x + 1) * segment_x;
      bodies[i * 3 + 1] = (double)(1 - 2 * (ri & 1));
      bodies[i * 3 + 2] = -1 + (z + 1) * segment_z;
      break;
    }
    case 6: { // surface parallel to Y-Z plane
      int64_t y = (ri >> 1) / x_splits[2];
      int64_t z = (ri >> 1) - y * x_splits[2];
      bodies[i * 3] = (double)(1 - 2 * (ri & 1));
      bodies[i * 3 + 1] = -1 + (y + 1) * segment_y;
      bodies[i * 3 + 2] = -1 + (z + 1) * segment_z;
      break;
    }
    default:
      break;
    }
  }
}

void mesh_unit_sphere(double* bodies, int64_t nbodies) {
  const double phi = M_PI * (3. - std::sqrt(5.));  // golden angle in radians
  for (int64_t i = 0; i < nbodies; i++) {
    const double y = 1. - ((double)i / ((double)nbodies - 1)) * 2.;  // y goes from 1 to -1

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

void magnify_reloc(double* bodies, int64_t nbodies, const double Ccur[], const double Cnew[], const double R[], double alpha) {
  double Ra[3];
  for (int64_t i = 0; i < 3; i++)
    Ra[i] = R[i] * alpha;
  for (int64_t i = 0; i < nbodies; i++) {
    double* x_bi = &bodies[i * 3];
    x_bi[0] = Cnew[0] + Ra[0] * (x_bi[0] - Ccur[0]);
    x_bi[1] = Cnew[1] + Ra[1] * (x_bi[1] - Ccur[1]);
    x_bi[2] = Cnew[2] + Ra[2] * (x_bi[2] - Ccur[2]);
  }
}

void body_neutral_charge(double X[], int64_t nbodies, double cmax, unsigned int seed) {
  if (seed > 0)
    srand(seed);

  double cmax2 = cmax * 2;
  for (int64_t i = 0; i < nbodies; i++)
    X[i] = ((double)rand() / RAND_MAX) * cmax2 - cmax;
}

void read_sorted_bodies(int64_t* nbodies, int64_t lbuckets, double* bodies, int64_t buckets[], const char* fname) {
  std::ifstream file(fname);

  int64_t curr = 1, cbegin = 0, iter = 0, len = *nbodies;
  while (iter < len && !file.eof()) {
    int64_t b = 0;
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

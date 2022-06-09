
#include "kernel.h"

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "inttypes.h"

double _singularity = 1.e-8;
double _alpha = 1.;

void laplace3d(double* r2) {
  double _r2 = *r2;
  double r = sqrt(_r2) + _singularity;
  *r2 = 1. / r;
}

void yukawa3d(double* r2) {
  double _r2 = *r2;
  double r = sqrt(_r2) + _singularity;
  *r2 = exp(_alpha * -r) / r;
}

void set_kernel_constants(double singularity, double alpha) {
  _singularity = singularity;
  _alpha = alpha;
}

void gen_matrix(KerFunc_t ef, int64_t m, int64_t n, const Bodies bi, const Bodies bj, double Aij[], const int64_t sel_i[], const int64_t sel_j[]) {
  for (int64_t i = 0; i < m * n; i++) {
    int64_t x = i / m;
    int64_t bx = sel_j == NULL ? x : sel_j[x];
    int64_t y = i - x * m;
    int64_t by = sel_i == NULL ? y : sel_i[y];

    const struct Body* bii = bi + by;
    const struct Body* bjj = bj + bx;

    double dX = bii->X[0] - bjj->X[0];
    double dY = bii->X[1] - bjj->X[1];
    double dZ = bii->X[2] - bjj->X[2];

    double r2 = dX * dX + dY * dY + dZ * dZ;
    ef(&r2);
    Aij[x * m + y] = r2;
  }
}

void uniform_unit_cube(Bodies bodies, int64_t nbodies, int64_t dim, unsigned int seed) {
  if (seed > 0)
    srand(seed);

  for (int64_t i = 0; i < nbodies; i++) {
    double r0 = dim > 0 ? ((double)rand() / RAND_MAX) : 0.;
    double r1 = dim > 1 ? ((double)rand() / RAND_MAX) : 0.;
    double r2 = dim > 2 ? ((double)rand() / RAND_MAX) : 0.;
    bodies[i].X[0] = r0;
    bodies[i].X[1] = r1;
    bodies[i].X[2] = r2;
  }
}

void mesh_unit_sphere(Bodies bodies, int64_t nbodies) {
  int64_t mlen = nbodies - 2;
  if (mlen < 0) {
    fprintf(stderr, "Error spherical mesh size (GT/EQ. 2 required): %" PRId64 ".\n", nbodies);
    return;
  }

  double alen = sqrt(mlen);
  int64_t m = (int64_t)ceil(alen);
  int64_t n = (int64_t)ceil((double)mlen / m);

  double pi = M_PI;
  double seg_theta = pi / (m + 1);
  double seg_phi = 2. * pi / n;

  for (int64_t i = 0; i < mlen; i++) {
    int64_t x = i / m;
    int64_t y = 1 + i - x * m;
    int64_t x2 = !(y & 1);

    double theta = y * seg_theta;
    double phi = (0.5 * x2 + x) * seg_phi;

    double cost = cos(theta);
    double sint = sin(theta);
    double cosp = cos(phi);
    double sinp = sin(phi);

    double* x_bi = bodies[i + 1].X;
    x_bi[0] = sint * cosp;
    x_bi[1] = sint * sinp;
    x_bi[2] = cost;
  }

  bodies[0].X[0] = 0.;
  bodies[0].X[1] = 0.;
  bodies[0].X[2] = 1.;

  bodies[nbodies - 1].X[0] = 0.;
  bodies[nbodies - 1].X[1] = 0.;
  bodies[nbodies - 1].X[2] = -1.;
}

void mesh_unit_cube(Bodies bodies, int64_t nbodies) {
  if (nbodies < 0) {
    fprintf(stderr, "Error cubic mesh size (GT/EQ. 0 required): %" PRId64 ".\n", nbodies);
    return;
  }

  int64_t mlen = (int64_t)ceil((double)nbodies / 6.);
  double alen = sqrt(mlen);
  int64_t m = (int64_t)ceil(alen);
  int64_t n = (int64_t)ceil((double)mlen / m);

  double seg_fv = 1. / (m - 1);
  double seg_fu = 1. / n;
  double seg_sv = 1. / (m + 1);
  double seg_su = 1. / (n + 1);

  for (int64_t i = 0; i < nbodies; i++) {
    int64_t face = i / mlen;
    int64_t ii = i - face * mlen;
    int64_t x = ii / m;
    int64_t y = ii - x * m;
    int64_t x2 = y & 1;

    double u, v;
    double* x_bi = bodies[i].X;

    switch (face) {
      case 0: // POSITIVE X
        v = y * seg_fv;
        u = (0.5 * x2 + x) * seg_fu;
        x_bi[0] = 1.;
        x_bi[1] = 2. * v - 1.;
        x_bi[2] = -2. * u + 1.;
        break;
      case 1: // NEGATIVE X
        v = y * seg_fv;
        u = (0.5 * x2 + x) * seg_fu;
        x_bi[0] = -1.;
        x_bi[1] = 2. * v - 1.;
        x_bi[2] = 2. * u - 1.;
        break;
      case 2: // POSITIVE Y
        v = (y + 1) * seg_sv;
        u = (0.5 * x2 + x + 1) * seg_su;
        x_bi[0] = 2. * u - 1.;
        x_bi[1] = 1.;
        x_bi[2] = -2. * v + 1.;
        break;
      case 3: // NEGATIVE Y
        v = (y + 1) * seg_sv;
        u = (0.5 * x2 + x + 1) * seg_su;
        x_bi[0] = 2. * u - 1.;
        x_bi[1] = -1.;
        x_bi[2] = 2. * v - 1.;
        break;
      case 4: // POSITIVE Z
        v = y * seg_fv;
        u = (0.5 * x2 + x) * seg_fu;
        x_bi[0] = 2. * u - 1.;
        x_bi[1] = 2. * v - 1.;
        x_bi[2] = 1.;
        break;
      case 5: // NEGATIVE Z
        v = y * seg_fv;
        u = (0.5 * x2 + x) * seg_fu;
        x_bi[0] = -2. * u + 1.;
        x_bi[1] = 2. * v - 1.;
        x_bi[2] = -1.;
        break;
    }
  }
}

void magnify_reloc(Bodies bodies, int64_t nbodies, const double Ccur[], const double Cnew[], const double R[]) {
  for (int64_t i = 0; i < nbodies; i++) {
    double* x_bi = bodies[i].X;
    double v0 = x_bi[0] - Ccur[0];
    double v1 = x_bi[1] - Ccur[1];
    double v2 = x_bi[2] - Ccur[2];
    x_bi[0] = Cnew[0] + R[0] * v0;
    x_bi[1] = Cnew[1] + R[1] * v1;
    x_bi[2] = Cnew[2] + R[2] * v2;
  }
}

void body_neutral_charge(Bodies bodies, int64_t nbodies, double cmax, unsigned int seed) {
  if (seed > 0)
    srand(seed);

  double avg = 0.;
  double cmax2 = cmax * 2;
  for (int64_t i = 0; i < nbodies; i++) {
    double c = ((double)rand() / RAND_MAX) * cmax2 - cmax;
    bodies[i].B = c;
    avg = avg + c;
  }
  avg = avg / nbodies;

  if (avg != 0.)
    for (int64_t i = 0; i < nbodies; i++)
      bodies[i].B = bodies[i].B - avg;
}

void get_bounds(const Bodies bodies, int64_t nbodies, double R[], double C[]) {
  double Xmin[DIM_MAX], Xmax[DIM_MAX];
  Xmin[0] = Xmax[0] = bodies[0].X[0];
  Xmin[1] = Xmax[1] = bodies[0].X[1];
  Xmin[2] = Xmax[2] = bodies[0].X[2];

  for (int64_t i = 1; i < nbodies; i++) {
    const double* x_bi = bodies[i].X;
    Xmin[0] = fmin(x_bi[0], Xmin[0]);
    Xmin[1] = fmin(x_bi[1], Xmin[1]);
    Xmin[2] = fmin(x_bi[2], Xmin[2]);

    Xmax[0] = fmax(x_bi[0], Xmax[0]);
    Xmax[1] = fmax(x_bi[1], Xmax[1]);
    Xmax[2] = fmax(x_bi[2], Xmax[2]);
  }

  C[0] = (Xmin[0] + Xmax[0]) / 2.;
  C[1] = (Xmin[1] + Xmax[1]) / 2.;
  C[2] = (Xmin[2] + Xmax[2]) / 2.;

  double d0 = Xmax[0] - Xmin[0];
  double d1 = Xmax[1] - Xmin[1];
  double d2 = Xmax[2] - Xmin[2];

  R[0] = (d0 == 0. && Xmin[0] == 0.) ? 0. : (1.e-8 +  d0 / 2.);
  R[1] = (d1 == 0. && Xmin[1] == 0.) ? 0. : (1.e-8 +  d1 / 2.);
  R[2] = (d2 == 0. && Xmin[2] == 0.) ? 0. : (1.e-8 +  d2 / 2.);
}

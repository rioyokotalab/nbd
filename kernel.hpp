#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>

struct EvalDouble {
  virtual double operator()(double d) const = 0;
};

struct Laplace3D : public EvalDouble {
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  double operator()(double d) const override {
    return d == 0. ? singularity : (1. / d);
  }
};

struct Yukawa3D : public EvalDouble {
  double singularity, alpha;
  Yukawa3D (double s, double a) : singularity(1. / s), alpha(a) {}
  double operator()(double d) const override {
    return d == 0. ? singularity : (std::exp(-alpha * d) / d);
  }
};

struct Gaussian : public EvalDouble {
  double alpha;
  Gaussian (double a) : alpha(1. / (a * a)) {}
  double operator()(double d) const override {
    return std::exp(- alpha * d * d);
  }
};


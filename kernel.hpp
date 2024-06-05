#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>

class EvalDouble {
public:
  virtual double operator()(double d) const = 0;
};

class Laplace3D : public EvalDouble {
public:
  double singularity;
  Laplace3D (double s) : singularity(1. / s) {}
  double operator()(double d) const override {
    return d == 0. ? singularity : (1. / d);
  }
};

class Yukawa3D : public EvalDouble {
public:
  double singularity, alpha;
  Yukawa3D (double s, double a) : singularity(1. / s), alpha(a) {}
  double operator()(double d) const override {
    return d == 0. ? singularity : (std::exp(-alpha * d) / d);
  }
};

class Gaussian : public EvalDouble {
public:
  double alpha;
  Gaussian (double a) : alpha(1. / (a * a)) {}
  double operator()(double d) const override {
    return std::exp(- alpha * d * d);
  }
};


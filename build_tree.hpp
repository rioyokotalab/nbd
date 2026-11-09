
#pragma once

#include <array>

class Cell {
public:
  std::array<long long, 2> Child;
  std::array<long long, 2> M;
  std::array<long long, 2> N;
  std::array<double, 3> R;
  std::array<double, 3> C;

  Cell() : Child(std::array<long long, 2>{ -1, -1 }), M(std::array<long long, 2>{ -1, -1 }), N(std::array<long long, 2>{ -1, -1 }), R(std::array<double, 3>{ 0., 0., 0. }), C(std::array<double, 3>{ 0., 0., 0. }) {}
};

void buildBinaryTree(Cell* cells, double* bodies, long long nbodies, long long levels);


#pragma once

#include <cstdint>
#include <cstddef>

class Node;
class Base;
class CSR;
class ColCommMPI;

void matVecA(const Node A[], const Base basis[], const CSR rels_near[], double* X, const ColCommMPI comm[], int64_t levels);

void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX);



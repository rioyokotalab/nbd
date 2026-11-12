#pragma once

#include <complex>
#include <vector>
#include <array>
#include <mpi.h>

class DenseMatrix;

class Cell {
public:
  std::array<long long, 2> Child, N;
  std::array<double, 3> R, C;

  Cell() : Child(std::array<long long, 2>{ -1, -1 }), N(std::array<long long, 2>{ -1, -1 }), R(std::array<double, 3>{ 0., 0., 0. }), C(std::array<double, 3>{ 0., 0., 0. }) {}
};

class CSR {
public:
  long long N;
  std::vector<long long> RowIndex, ColIndex;

  CSR(const std::vector<Cell>& cells);
  CSR(const CSR& global, long long lbegin, long long lend, const std::vector<std::array<long long, 3>>& boxes);
};

class HierarchicalComm {
public:
  long long Proc, LocalChild;
  std::vector<std::array<long long, 3>> Boxes;
  
  long long NeighborComms, NeighborLoc;
  MPI_Comm DupComm, RepComm;

  HierarchicalComm(const std::pair<long long, long long> Tree[], std::pair<long long, long long> Mapping[], const long long CRows[], const long long CCols[], MPI_Comm world = MPI_COMM_WORLD);
  
  long long oLocal() const;
  long long oGlobal() const;
  long long lenLocal() const;
  long long lenNeighbors() const;
  long long oLocalChild(const HierarchicalComm& lower) const;

  void bcastNeighbor(void* data, const long long noffsets[], size_t size) const;
  void bcastNeighbor(long long data[], const long long noffsets[]) const;
  void bcastNeighbor(std::complex<double> data[], const long long noffsets[]) const;

  static double get_comm_time(MPI_Comm world);
  static void record_mpi();
};

class HSSLevel {
public:
  std::vector<long long> URowPrefixes;
  std::vector<long long> UColPrefixes;
  std::vector<long long> VRowPrefixes;
  std::vector<long long> VColPrefixes;

  std::vector<long long> UPrefixes;
  std::vector<long long> VPrefixes;
  std::vector<long long> APrefixes;
  std::vector<long long> CPrefixes;

  std::vector<std::complex<double>> U;
  std::vector<std::complex<double>> V;
  std::vector<std::complex<double>> A;
  std::vector<std::complex<double>> C;
  
  void eval(double epi, DenseMatrix& matA, const long long localM[], const long long localN[], const CSR& rels, const HierarchicalComm& comm, long long localNoffset = -1ll);
  void matVecUpwardPass(const std::complex<double>* X_in, std::complex<double>* Z, const HierarchicalComm& comm) const;
  void matVecHorizontalandDownwardPass(std::complex<double>* Y_out, const std::complex<double>* Z, std::complex<double>* Y, const CSR& rels, const HierarchicalComm& comm) const;
  void matVecLeafHorizontalPass(std::complex<double>* X_io, const std::complex<double>* Z, std::complex<double>* Y, const CSR& rels, const HierarchicalComm& comm) const;
};

class HSS {
public:
  long long levels;
  std::vector<HierarchicalComm> comm;
  std::vector<CSR> Far_local;
  MPI_Comm world;

  std::vector<HSSLevel> A;
  std::pair<long long, long long> local_bodies;

  HSS(const std::vector<Cell>& cells, long long levels, MPI_Comm world);
  void evalMatVecA(DenseMatrix& matA, double epi, const Cell cells[]);
  void allocRHSwork(std::vector<long long>& iwork, std::vector<std::complex<double>>& cwork) const;

  void matVecMul(std::complex<double> X[]) const;
  void matVecMul(std::complex<double> X[], const long long iwork[], std::complex<double> cwork[]) const;
  double solveRelErr(const std::complex<double> X[], const std::complex<double> ref[]);

  void free_all_comms();
};

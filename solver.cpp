
#include <solver.hpp>
#include <basis_builder.hpp>
#include <build_tree.hpp>
#include <comm-mpi.hpp>
#include <csr.hpp>
#include <dense.hpp>
#include <h2matrix.hpp>

#include <algorithm>
#include <numeric>

H2MatrixSolver::H2MatrixSolver(const std::vector<Cell>& cells, const CSR& Far, long long levels, MPI_Comm world) : 
  levels(levels), world(world), A(levels + 1) {
  int mpi_size = 1;
  MPI_Comm_size(world, &mpi_size);

  std::vector<std::pair<long long, long long>> mapping(mpi_size, std::make_pair(0, 1));
  std::vector<std::pair<long long, long long>> tree(cells.size());
  std::transform(cells.begin(), cells.end(), tree.begin(), [](const Cell& c) { return std::make_pair(c.Child[0], c.Child[1]); });
  
  for (long long i = 0; i <= levels; ++i) {
    comm.emplace_back(&tree[0], &mapping[0], Far.RowIndex.data(), Far.ColIndex.data(), world);
    Far_local.emplace_back(Far, comm[i].oGlobal(), comm[i].oGlobal() + comm[i].lenLocal(), comm[i].Boxes);
  }

  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  local_bodies = std::make_pair(cells[gbegin].Body[0], cells[gbegin + llen - 1].Body[1]);
}

void H2MatrixSolver::evalMatVecA(const DenseMatrix& matA, double epi, const Cell cells[], const CSR& Far) {
  std::vector<BasisBuilder> row_side, col_side;
  row_side.reserve(levels + 1);
  col_side.reserve(levels + 1);
  row_side.emplace_back(0, 1, cells, Far);
  col_side.emplace_back(0, 1, cells, Far);

  for (long long l = 1; l <= levels; l++) {
    row_side.emplace_back(comm[l].oGlobal(), comm[l].lenLocal(), cells, Far, row_side[l - 1]);
    col_side.emplace_back(comm[l].oGlobal(), comm[l].lenLocal(), cells, Far, col_side[l - 1]);
  }

  DenseMatrix AH(matA.rowOffsets[matA.mpi_rank + 1] - matA.rowOffsets[matA.mpi_rank], matA.N, world);
  matA.adjointA(AH);

  row_side[levels].evalLeafFMatrix(epi, matA, cells);
  std::copy(row_side[levels].Ranks.begin(), row_side[levels].Ranks.end(), col_side[levels].Ranks.begin());
  col_side[levels].evalLeafFMatrix(epi, AH, cells);

  A[levels].eval(cells, Far, row_side[levels], col_side[levels], Far_local[levels], comm[levels]);
  A[levels].evalLeafA(matA, cells, comm[levels]);
  AH.A.clear(); AH.A.shrink_to_fit();

  for (long long l = levels - 1; l >= 0; l--) {
    row_side[l].evalInnerFMatrix(epi, row_side[l + 1], cells, comm[l]);
    std::copy(row_side[l].Ranks.begin(), row_side[l].Ranks.end(), col_side[l].Ranks.begin());
    col_side[l].evalInnerFMatrix(epi, col_side[l + 1], cells, comm[l]);
    A[l].eval(cells, Far, row_side[l], col_side[l], Far_local[l], comm[l]);
  }
}

void H2MatrixSolver::allocRHSwork(std::vector<long long>& iwork, std::vector<std::complex<double>>& cwork) const {
  iwork.resize(levels + levels + 2);
  for (long long l = 0; l < levels; l++) {
    iwork[l + 1] = A[l].DimsOffsets.back();
    iwork[l + levels + 2] = A[l + 1].DimsLrOffsets[comm[l].oLocalChild(comm[l + 1]).first];
  }

  iwork[0] = 0;
  iwork[levels + 1] = A[levels].DimsOffsets.back();
  std::partial_sum(&iwork[1], &iwork[levels + 2], &iwork[1]);
  cwork.resize(iwork[levels + 1] * 2);
}

void H2MatrixSolver::matVecMul(std::complex<double> X[]) const {
  std::vector<long long> iworkA;
  std::vector<std::complex<double>> cworkA;
  allocRHSwork(iworkA, cworkA);
  matVecMul(X, iworkA.data(), cworkA.data());
}

void H2MatrixSolver::matVecMul(std::complex<double> X[], const long long iwork[], std::complex<double> cwork[]) const {
  const long long* xSizes = iwork, *LowerZ = &iwork[levels + 2];
  std::complex<double>* x = cwork, *y = &cwork[xSizes[levels + 1]];

  if (0 < levels)
    A[levels].matVecUpwardPass(X, &x[xSizes[levels]], comm[levels]);
  for (long long l = levels - 1; l >= 0; l--)
    A[l].matVecUpwardPass(&x[xSizes[l + 1] + LowerZ[l]], &x[xSizes[l]], comm[l]);

  for (long long l = 0; l < levels; l++)
    A[l].matVecHorizontalandDownwardPass(&y[xSizes[l + 1] + LowerZ[l]], &x[xSizes[l]], &y[xSizes[l]], Far_local[l]);
  A[levels].matVecLeafHorizontalPass(X, &x[xSizes[levels]], &y[xSizes[levels]], Far_local[levels], comm[levels]);
}

double H2MatrixSolver::solveRelErr(const std::complex<double> X[], const std::complex<double> ref[]) {
  long long lenX = local_bodies.second - local_bodies.first;
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    std::complex<double> diff = X[i] - ref[i];
    err[0] += std::norm(diff);
    err[1] += std::norm(ref[i]);
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, world);
  return std::sqrt(err[0] / err[1]);
}

void H2MatrixSolver::free_all_comms() {
  for (H2MatrixComm& com : comm) {
    if (com.MergeComm != MPI_COMM_NULL)
      MPI_Comm_free(&com.MergeComm);
    if (com.DupComm != MPI_COMM_NULL)
      MPI_Comm_free(&com.DupComm);
  }
}

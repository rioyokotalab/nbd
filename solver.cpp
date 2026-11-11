
#include <solver.hpp>
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
  local_bodies = std::make_pair(cells[gbegin].N[0], cells[gbegin + llen - 1].N[1]);
}

void H2MatrixSolver::evalMatVecA(DenseMatrix& matA, double epi, Cell cells[]) {
  long long llen = comm[levels].lenLocal();
  long long gbegin = comm[levels].oGlobal();
  std::vector<long long> localM, localN(llen);
  std::transform(&cells[gbegin], &cells[gbegin + llen], localN.begin(), [](const Cell& c) { return c.N[1] - c.N[0]; });
  A[levels].eval(epi, matA, &localN[0], &localN[0], Far_local[levels], comm[levels], cells[gbegin].N[0]);

  for (long long l = levels - 1; l >= 0; l--) {
    do {
      DenseMatrix matAl(A[l + 1].UColPrefixes.back(), matA.N, comm[l].RepComm, comm[l].DupComm);
      matAl.copyFromA(matA); matA = std::move(matAl);
    } while(0);
    llen = comm[l].lenLocal(); gbegin = comm[l].oGlobal(); localM.resize(llen); localN.resize(llen);
    for (long long i = 0; i < llen; ++i) {
      long long cbegin = cells[gbegin + i].Child[0] - cells[gbegin].Child[0], cend = cells[gbegin + i].Child[1] - cells[gbegin].Child[0];
      localM[i] = A[l + 1].UColPrefixes[cend] - A[l + 1].UColPrefixes[cbegin];
      localN[i] = A[l + 1].VColPrefixes[cend] - A[l + 1].VColPrefixes[cbegin];
    }
    A[l].eval(epi, matA, &localM[0], &localN[0], Far_local[l], comm[l]);
  }
}

void H2MatrixSolver::allocRHSwork(std::vector<long long>& iwork, std::vector<std::complex<double>>& cwork) const {
  iwork.resize((levels * 4ll) + 4ll);
  long long* lenX = &iwork[0], *lenY = &iwork[levels + 2ll], *locV = &iwork[(levels * 2ll) + 4ll], *locU = &iwork[(levels * 3ll) + 4ll];
  lenX[0] = lenY[0] = 0;
  for (long long l = 0; l < levels; l++) {
    lenX[l + 1] = 0ll < l ? A[l].VRowPrefixes.back() : 0ll;
    lenY[l + 1] = 0ll < l ? A[l].URowPrefixes.back() : 0ll;
    locV[l] = A[l + 1].VColPrefixes[comm[l].oLocalChild(comm[l + 1]).first];
    locU[l] = A[l + 1].UColPrefixes[comm[l].oLocalChild(comm[l + 1]).first];
  }

  lenX[levels + 1] = A[levels].VRowPrefixes.back();
  lenY[levels + 1] = A[levels].URowPrefixes.back();
  std::partial_sum(&lenX[1], locV, &lenX[1]);
  cwork.resize(lenY[levels + 1]);
}

void H2MatrixSolver::matVecMul(std::complex<double> X[]) const {
  std::vector<long long> iworkA;
  std::vector<std::complex<double>> cworkA;
  allocRHSwork(iworkA, cworkA);
  matVecMul(X, iworkA.data(), cworkA.data());
}

void H2MatrixSolver::matVecMul(std::complex<double> X[], const long long iwork[], std::complex<double> cwork[]) const {
  const long long* lenX = iwork, *lenY = &iwork[levels + 2ll], *locV = &iwork[(levels * 2ll) + 4ll], *locU = &iwork[(levels * 3ll) + 4ll];
  if (0 < levels)
    A[levels].matVecUpwardPass(X, &cwork[lenX[levels]], comm[levels]);
  for (long long l = levels - 1; l >= 0; l--)
    A[l].matVecUpwardPass(&cwork[lenX[l + 1] + locV[l]], &cwork[lenX[l]], comm[l]);

  for (long long l = 0; l < levels; l++)
    A[l].matVecHorizontalandDownwardPass(&cwork[lenY[l + 1] + locU[l]], &cwork[lenX[l]], &cwork[lenY[l]], Far_local[l], comm[l]);
  A[levels].matVecLeafHorizontalPass(X, &cwork[lenX[levels]], &cwork[lenY[levels]], Far_local[levels], comm[levels]);
}

double H2MatrixSolver::solveRelErr(const std::complex<double> X[], const std::complex<double> ref[]) {
  long long lenX = local_bodies.second - local_bodies.first;
  double err[2] = { 0., 0. };
  for (long long i = 0; i < lenX; i++) {
    err[0] += std::norm(X[i] - ref[i]);
    err[1] += std::norm(ref[i]);
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, world);
  return std::sqrt(err[0] / err[1]);
}

void H2MatrixSolver::free_all_comms() {
  for (H2MatrixComm& com : comm) {
    if (com.RepComm != MPI_COMM_NULL)
      MPI_Comm_free(&com.RepComm);
    if (com.DupComm != MPI_COMM_NULL)
      MPI_Comm_free(&com.DupComm);
  }
}

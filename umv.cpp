
#include <nbd.hpp>

#include <cstdio>
#include <cstring>
#include <cmath>

#include <cstdlib>
#include <algorithm>

void allocNodes(struct Node A[], double** Workspace, int64_t* Lwork, const struct Base basis[], const CSR rels_near[], const CSR rels_far[], const struct CellComm comm[], int64_t levels) {
  int64_t work_size = 0;

  for (int64_t i = levels; i >= 0; i--) {
    int64_t n_i = 0, ulen = 0, nloc = 0;
    content_length(&n_i, &ulen, &nloc, &comm[i]);
    int64_t nnz = rels_near[i].RowIndex[n_i];
    int64_t nnz_f = rels_far[i].RowIndex[n_i];

    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * (nnz + nnz_f));
    A[i].A = arr_m;
    A[i].S = &arr_m[nnz];
    A[i].lenA = nnz;
    A[i].lenS = nnz_f;

    int64_t dimn = basis[i].dimR + basis[i].dimS;
    int64_t dimn_up = i > 0 ? basis[i - 1].dimN : 0;

    int64_t stride = dimn * dimn;
    allocBufferedList((void**)&A[i].A_ptr, (void**)&A[i].A_buf, sizeof(double), stride * nnz);
    allocBufferedList((void**)&A[i].X_ptr, (void**)&A[i].X_buf, sizeof(double), dimn * ulen);

    int64_t k1, k2;
    countMaxIJ(&k1, &k2, &rels_near[i]);
    int64_t acc_required = std::max(k1 * ulen, k2 * n_i);
    int64_t work_required = std::max(n_i * stride, (acc_required + n_i) * dimn);
    work_size = std::max(work_size, work_required);

    for (int64_t x = 0; x < n_i; x++) {
      for (int64_t yx = rels_near[i].RowIndex[x]; yx < rels_near[i].RowIndex[x + 1]; yx++)
        arr_m[yx] = (struct Matrix) { &A[i].A_buf[yx * stride], dimn, dimn, dimn }; // A

      for (int64_t yx = rels_far[i].RowIndex[x]; yx < rels_far[i].RowIndex[x + 1]; yx++)
        arr_m[yx + nnz] = (struct Matrix) { NULL, basis[i].dimS, basis[i].dimS, dimn_up }; // S
    }

    if (i < levels) {
      int64_t ploc = 0;
      content_length(NULL, NULL, &ploc, &comm[i + 1]);
      int64_t seg = basis[i + 1].dimS;

      for (int64_t j = 0; j < rels_near[i].N; j++) {
        int64_t x0 = std::get<0>(comm[i].LocalChild[j + nloc]) - ploc;
        int64_t lenx = std::get<1>(comm[i].LocalChild[j + nloc]);

        for (int64_t ij = rels_near[i].RowIndex[j]; ij < rels_near[i].RowIndex[j + 1]; ij++) {
          int64_t li = rels_near[i].ColIndex[ij];
          int64_t y0 = std::get<0>(comm[i].LocalChild[li]);
          int64_t leny = std::get<1>(comm[i].LocalChild[li]);
          
          for (int64_t x = 0; x < lenx; x++)
            if ((x + x0) >= 0 && (x + x0) < rels_far[i + 1].N)
              for (int64_t yx = rels_far[i + 1].RowIndex[x + x0]; yx < rels_far[i + 1].RowIndex[x + x0 + 1]; yx++)
                for (int64_t y = 0; y < leny; y++)
                  if (rels_far[i + 1].ColIndex[yx] == (y + y0))
                    A[i + 1].S[yx].A = &A[i].A[ij].A[(y * dimn + x) * seg];
        }
      }
    }
  }
  
  set_work_size(work_size, Workspace, Lwork);
  for (int64_t i = levels; i > 0; i--) {
    int64_t ibegin = 0, N_rows = 0, N_cols = 0;
    content_length(&N_cols, &N_rows, &ibegin, &comm[i]);
    int64_t nnz = A[i].lenA;
    int64_t dimc = basis[i].dimR;
    int64_t dimr = basis[i].dimS;

    int64_t n_next = basis[i - 1].dimR + basis[i - 1].dimS;
    int64_t ibegin_next = 0;
    content_length(NULL, NULL, &ibegin_next, &comm[i - 1]);

    std::vector<double*> A_next(nnz);
    for (int64_t x = 0; x < N_cols; x++)
      for (int64_t yx = rels_near[i].RowIndex[x]; yx < rels_near[i].RowIndex[x + 1]; yx++) {
        int64_t y = rels_near[i].ColIndex[yx];
        std::pair<int64_t, int64_t> px = comm[i].LocalParent[x + ibegin];
        std::pair<int64_t, int64_t> py = comm[i].LocalParent[y];
        int64_t ij = rels_near[i - 1].lookupIJ(std::get<0>(py), std::get<0>(px) - ibegin_next);
        A_next[yx] = &A[i - 1].A_ptr[(std::get<1>(py) * n_next + std::get<1>(px)) * basis[i].dimS + ij * n_next * n_next];
      }

    std::vector<double*> X_next(N_rows);
    for (int64_t x = 0; x < N_rows; x++) { 
      std::pair<int64_t, int64_t> p = comm[i].LocalParent[x];
      X_next[x] = &A[i - 1].X_ptr[std::get<1>(p) * basis[i].dimS + std::get<0>(p) * n_next];
    }

    batchParamsCreate(&A[i].params, dimc, dimr, basis[i].U_gpu, A[i].A_ptr, A[i].X_ptr, n_next, &A_next[0], &X_next[0],
      *Workspace, work_size, N_rows, N_cols, ibegin, &rels_near[i].ColIndex[0], &rels_near[i].RowIndex[0]);
  }

  int64_t child = std::get<0>(comm[0].LocalChild[0]);
  int64_t clen = std::get<1>(comm[0].LocalChild[0]);
  std::vector<int64_t> cdims(clen);
  if (child >= 0)
    for (int64_t i = 0; i < clen; i++)
      cdims[i] = basis[1].DimsLr[child + i];
  else
    cdims.emplace_back(basis[0].Dims[0]);
  int64_t low_s = clen > 0 ? basis[1].dimS : 0;
  lastParamsCreate(&A[0].params, A[0].A_ptr, A[0].X_ptr, basis[0].dimN, low_s, cdims.size(), &cdims[0]);
}

void node_free(struct Node* node) {
  freeBufferedList(node->A_ptr, node->A_buf);
  freeBufferedList(node->X_ptr, node->X_buf);
  free(node->A);
  batchParamsDestory(&node->params);
}

void factorA_mov_mem(char dir, struct Node A[], const struct Base basis[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t stride = basis[i].dimN * basis[i].dimN;
    flushBuffer(dir, A[i].A_ptr, A[i].A_buf, sizeof(double), stride * A[i].lenA);
  }
}

struct RightHandSides { struct Matrix *X, *Xc, *Xo, *B; };

void allocRightHandSidesMV(struct RightHandSides rhs[], const struct Base base[], const struct CellComm comm[], int64_t levels) {
  for (int64_t l = 0; l <= levels; l++) {
    int64_t len;
    content_length(NULL, &len, NULL, &comm[l]);
    int64_t len_arr = len * 4;
    struct Matrix* arr_m = (struct Matrix*)calloc(len_arr, sizeof(struct Matrix));
    rhs[l].X = arr_m;
    rhs[l].B = &arr_m[len];
    rhs[l].Xo = &arr_m[len * 2];
    rhs[l].Xc = &arr_m[len * 3];

    int64_t len_data = len * base[l].dimN * 2;
    double* data = (double*)calloc(len_data, sizeof(double));
    for (int64_t i = 0; i < len; i++) {
      std::pair<int64_t, int64_t> p = comm[l].LocalParent[i];
      arr_m[i] = (struct Matrix) { &data[i * base[l].dimN], base[l].dimN, 1, base[l].dimN }; // X
      arr_m[i + len] = (struct Matrix) { &data[len * base[l].dimN + i * base[l].dimN], base[l].dimN, 1, base[l].dimN }; // B

      double* x_next = (l == 0) ? NULL : &rhs[l - 1].X[0].A[std::get<1>(p) * base[l].dimS + std::get<0>(p) * base[l - 1].dimN];
      arr_m[i + len * 2] = (struct Matrix) { x_next, base[l].dimS, 1, base[l].dimS }; // Xo

      double* b_next = (l == 0) ? NULL : &rhs[l - 1].B[0].A[std::get<1>(p) * base[l].dimS + std::get<0>(p) * base[l - 1].dimN];
      arr_m[i + len * 3] = (struct Matrix) { b_next, base[l].dimS, 1, base[l].dimS }; // Xc
    }
  }
}

void rightHandSides_free(struct RightHandSides* rhs) {
  double* data = rhs->X[0].A;
  if (data)
    free(data);
  free(rhs->X);
}

void matVecA(const struct Node A[], const struct Base basis[], const CSR rels_near[], double* X, const struct CellComm comm[], int64_t levels) {
  int64_t lbegin = 0, llen = 0;
  content_length(&llen, NULL, &lbegin, &comm[levels]);

  std::vector<RightHandSides> rhs(levels + 1);
  allocRightHandSidesMV(&rhs[0], basis, comm, levels);
  memcpy(rhs[levels].X[lbegin].A, X, llen * basis[levels].dimN * sizeof(double));

  for (int64_t i = levels; i > 0; i--) {
    int64_t ibegin = 0, iboxes = 0, xlen = 0;
    content_length(&iboxes, &xlen, &ibegin, &comm[i]);

    level_merge_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);
    neighbor_bcast_cpu(rhs[i].X[0].A, basis[i].dimN, &comm[i]);
    dup_bcast_cpu(rhs[i].X[0].A, xlen * basis[i].dimN, &comm[i]);

    for (int64_t j = 0; j < iboxes; j++)
      mmult('T', 'N', &basis[i].Uo[j + ibegin], &rhs[i].X[j + ibegin], &rhs[i].Xo[j + ibegin], 1., 0.);
  }

  level_merge_cpu(rhs[0].X[0].A, basis[0].dimN, &comm[0]);
  dup_bcast_cpu(rhs[0].X[0].A, basis[0].dimN, &comm[0]);
  mmult('N', 'N', &A[0].A[0], &rhs[0].X[0], &rhs[0].B[0], 1., 0.);

  for (int64_t i = 1; i <= levels; i++) {
    int64_t ibegin = 0, iboxes = 0;
    content_length(&iboxes, NULL, &ibegin, &comm[i]);
    for (int64_t j = 0; j < iboxes; j++)
      mmult('N', 'N', &basis[i].Uo[j + ibegin], &rhs[i].Xc[j + ibegin], &rhs[i].B[j + ibegin], 1., 0.);
    for (int64_t y = 0; y < iboxes; y++)
      for (int64_t xy = rels_near[i].RowIndex[y]; xy < rels_near[i].RowIndex[y + 1]; xy++) {
        int64_t x = rels_near[i].ColIndex[xy];
        mmult('N', 'N', &A[i].A[xy], &rhs[i].X[x], &rhs[i].B[y + ibegin], 1., 1.);
      }
  }
  memcpy(X, rhs[levels].B[lbegin].A, llen * basis[levels].dimN * sizeof(double));
  for (int64_t i = 0; i <= levels; i++)
    rightHandSides_free(&rhs[i]);
}


void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX) {
  double err[2] = { 0., 0. };
  for (int64_t i = 0; i < lenX; i++) {
    double diff = X[i] - ref[i];
    err[0] = err[0] + diff * diff;
    err[1] = err[1] + ref[i] * ref[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = sqrt(err[0] / err[1]);
}


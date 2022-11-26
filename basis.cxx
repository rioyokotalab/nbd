
#include "nbd.h"
#include "profile.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include "stdlib.h"
#include "math.h"
#include "string.h"

struct SampleBodies 
{ int64_t LTlen, *FarLens, *FarAvails, **FarBodies, *CloseLens, *CloseAvails, **CloseBodies, *SkeLens, **Skeletons; };

int64_t gen_close(int64_t clen, int64_t close[], int64_t ngbs, const int64_t ngbs_body[], const int64_t ngbs_len[]) {
  int64_t avail = std::accumulate(ngbs_len, ngbs_len + ngbs, 0);
  clen = std::min(avail, clen);
  std::iota(close, close + clen, 0);
  std::transform(close, close + clen, close, [avail, clen](int64_t& i)->int64_t { return (double)(i * avail) / clen; });
  int64_t* begin = close;
  for (int64_t i = 0; i < ngbs; i++) {
    int64_t len = std::min(std::distance(begin, close + clen), ngbs_len[i]);
    int64_t slen = std::accumulate(ngbs_len, ngbs_len + i, 0);
    int64_t bound = ngbs_len[i] + slen;
    int64_t body = ngbs_body[i] - slen;
    int64_t* next = std::find_if(begin, begin + len, [bound](int64_t& i)->bool { return i >= bound; });
    std::transform(begin, next, begin, [body](int64_t& i)->int64_t { return i + body; });
    begin = next;
  }
  return clen;
}

int64_t gen_far(int64_t flen, int64_t far[], int64_t ngbs, const int64_t ngbs_body[], const int64_t ngbs_len[], int64_t nbody) {
  int64_t near = std::accumulate(ngbs_len, ngbs_len + ngbs, 0);
  int64_t avail = nbody - near;
  flen = std::min(avail, flen);
  std::iota(far, far + flen, 0);
  std::transform(far, far + flen, far, [avail, flen](int64_t& i)->int64_t { return (double)(i * avail) / flen; });
  int64_t* begin = far;
  for (int64_t i = 0; i < ngbs; i++) {
    int64_t slen = std::accumulate(ngbs_len, ngbs_len + i, 0);
    int64_t bound = ngbs_body[i] - slen;
    int64_t* next = std::find_if(begin, far + flen, [bound](int64_t& i)->bool { return i >= bound; });
    std::transform(begin, next, begin, [slen](int64_t& i)->int64_t { return i + slen; });
    begin = next;
  }
  std::transform(begin, far + flen, begin, [near](int64_t& i)->int64_t { return i + near; });
  return flen;
}

void buildSampleBodies(struct SampleBodies* sample, int64_t sp_max_far, int64_t sp_max_near, int64_t nbodies, int64_t ncells, const struct Cell* cells, 
const struct CSC* rels, const int64_t* lt_child, const struct Base* basis_lo, int64_t level) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  const int64_t LEN_CHILD = 2;
  
  int64_t jbegin = 0, jend = ncells;
  get_level(&jbegin, &jend, cells, level, -1);
  int64_t ibegin = jbegin, iend = jend;
  get_level(&ibegin, &iend, cells, level, mpi_rank);
  int64_t nodes = iend - ibegin;
  int64_t* arr_ctrl = (int64_t*)malloc(sizeof(int64_t) * nodes * 5);
  int64_t** arr_list = (int64_t**)malloc(sizeof(int64_t*) * nodes * 3);
  sample->LTlen = nodes;
  sample->FarLens = arr_ctrl;
  sample->CloseLens = &arr_ctrl[nodes];
  sample->SkeLens = &arr_ctrl[nodes * 2];
  sample->FarAvails = &arr_ctrl[nodes * 3];
  sample->CloseAvails = &arr_ctrl[nodes * 4];
  sample->FarBodies = arr_list;
  sample->CloseBodies = &arr_list[nodes];
  sample->Skeletons = &arr_list[nodes * 2];

  int64_t count_f = 0, count_c = 0, count_s = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t li = ibegin + i;
    int64_t nbegin = rels->ColIndex[li];
    int64_t nlen = rels->ColIndex[li + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t far_avail = nbodies;
    int64_t close_avail = 0;
    for (int64_t j = 0; j < nlen; j++) {
      int64_t lj = ngbs[j];
      const struct Cell* cj = &cells[lj];
      int64_t len = cj->Body[1] - cj->Body[0];
      far_avail = far_avail - len;
      if (lj != li)
        close_avail = close_avail + len;
    }

    int64_t lc = lt_child[i];
    int64_t ske_len = 0;
    if (basis_lo != NULL && lc >= 0)
      for (int64_t j = 0; j < LEN_CHILD; j++)
        ske_len = ske_len + basis_lo->DimsLr[lc + j];
    else
      ske_len = cells[li].Body[1] - cells[li].Body[0];

    int64_t far_len = sp_max_far < far_avail ? sp_max_far : far_avail;
    int64_t close_len = sp_max_near < close_avail ? sp_max_near : close_avail;
    arr_ctrl[i] = far_len;
    arr_ctrl[i + nodes] = close_len;
    arr_ctrl[i + nodes * 2] = ske_len;
    arr_ctrl[i + nodes * 3] = far_avail;
    arr_ctrl[i + nodes * 4] = close_avail;
    count_f = count_f + far_len;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }

  int64_t* arr_bodies = NULL;
  if ((count_f + count_c + count_s) > 0)
    arr_bodies = (int64_t*)malloc(sizeof(int64_t) * (count_f + count_c + count_s));
  count_s = count_f + count_c;
  count_c = count_f;
  count_f = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t* remote = &arr_bodies[count_f];
    int64_t* close = &arr_bodies[count_c];
    int64_t* skeleton = &arr_bodies[count_s];
    int64_t far_len = arr_ctrl[i];
    int64_t close_len = arr_ctrl[i + nodes];
    int64_t ske_len = arr_ctrl[i + nodes * 2];
    arr_list[i] = remote;
    arr_list[i + nodes] = close;
    arr_list[i + nodes * 2] = skeleton;
    count_f = count_f + far_len;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }

#pragma omp parallel for
  for (int64_t i = 0; i < nodes; i++) {
    int64_t li = ibegin + i;
    int64_t nbegin = rels->ColIndex[li];
    int64_t nlen = rels->ColIndex[li + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t* remote = arr_list[i];
    int64_t* close = arr_list[i + nodes];
    int64_t* skeleton = arr_list[i + nodes * 2];
    int64_t far_len = arr_ctrl[i];
    int64_t close_len = arr_ctrl[i + nodes];
    int64_t ske_len = arr_ctrl[i + nodes * 2];
    int64_t far_avail = arr_ctrl[i + nodes * 3];
    int64_t close_avail = arr_ctrl[i + nodes * 4];

    int64_t box_i = 0;
    int64_t s_lens = 0;
    int64_t ic = ngbs[box_i];
    int64_t offset_i = cells[ic].Body[0];
    int64_t len_i = cells[ic].Body[1] - offset_i;

    int64_t cpos = 0;
    while (cpos < nlen && ngbs[cpos] != li)
      cpos = cpos + 1;

    for (int64_t j = 0; j < far_len; j++) {
      int64_t loc = (int64_t)((double)(far_avail * j) / far_len);
      while (box_i < nlen && loc + s_lens >= offset_i) {
        s_lens = s_lens + len_i;
        box_i = box_i + 1;
        ic = box_i < nlen ? (ngbs[box_i]) : ic;
        offset_i = cells[ic].Body[0];
        len_i = cells[ic].Body[1] - offset_i;
      }
      remote[j] = loc + s_lens;
    }

    box_i = (int64_t)(cpos == 0);
    s_lens = 0;
    ic = box_i < nlen ? (ngbs[box_i]) : ic;
    offset_i = cells[ic].Body[0];
    len_i = cells[ic].Body[1] - offset_i;

    std::vector<int64_t> body, lens;
    for (int64_t j = 0; j < nlen; j++)
      if (ngbs[j] != li) {
        body.emplace_back(cells[ngbs[j]].Body[0]);
        lens.emplace_back(cells[ngbs[j]].Body[1] - cells[ngbs[j]].Body[0]);
      }
    gen_close(sp_max_near, close, body.size(), body.data(), lens.data());

    body.clear(); lens.clear();
    for (int64_t j = 0; j < nlen; j++) {
      body.emplace_back(cells[ngbs[j]].Body[0]);
      lens.emplace_back(cells[ngbs[j]].Body[1] - cells[ngbs[j]].Body[0]);
    }
    gen_far(sp_max_far, remote, body.size(), body.data(), lens.data(), nbodies);

    int64_t lc = lt_child[i];
    int64_t sbegin = cells[li].Body[0];
    if (basis_lo != NULL && lc >= 0) {
      memcpy(skeleton, basis_lo->Multipoles + basis_lo->dimS * lc, sizeof(int64_t) * basis_lo->DimsLr[lc]);
      memcpy(skeleton + basis_lo->DimsLr[lc], basis_lo->Multipoles + basis_lo->dimS * (lc + 1), sizeof(int64_t) * basis_lo->DimsLr[lc + 1]);
    }
    else
      for (int64_t j = 0; j < ske_len; j++)
        skeleton[j] = j + sbegin;
  }
}

void sampleBodies_free(struct SampleBodies* sample) {
  int64_t* data = sample->FarBodies[0];
  if (data)
    free(data);
  free(sample->FarLens);
  free(sample->FarBodies);
}

int64_t dist_int_64(int64_t arr[], int64_t blen, const struct CellComm* comm) {
  int64_t plen = comm->Proc[0] == comm->worldRank ? comm->lenTargets : 0;
  const int64_t* row = comm->ProcTargets;
  int64_t lbegin = 0;
  int64_t lmax = 0;
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t len = (comm->ProcBoxesEnd[p] - comm->ProcBoxes[p]) * blen;
    MPI_Bcast(&arr[lbegin], len, MPI_INT64_T, comm->ProcRootI[p], comm->Comm_box[p]);
    lbegin = lbegin + len;
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = xlen * blen;
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr, alen, MPI_INT64_T, 0, comm->Comm_share);

  for (int64_t i = 0; i < alen; i++)
    lmax = lmax < arr[i] ? arr[i] : lmax;
  MPI_Allreduce(MPI_IN_PLACE, &lmax, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);

#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
  return lmax;
}

void dist_double(double* arr[], const struct CellComm* comm) {
  int64_t plen = comm->Proc[0] == comm->worldRank ? comm->lenTargets : 0;
  const int64_t* row = comm->ProcTargets;
  double* data = arr[0];
  int64_t lbegin = 0;
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t llen = comm->ProcBoxesEnd[p] - comm->ProcBoxes[p];
    int64_t offset = arr[lbegin] - data;
    int64_t len = arr[lbegin + llen] - arr[lbegin];
    MPI_Bcast(&data[offset], len, MPI_DOUBLE, comm->ProcRootI[p], comm->Comm_box[p]);
    lbegin = lbegin + llen;
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = arr[xlen] - data;
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(data, alen, MPI_DOUBLE, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void buildBasis(void(*ef)(double*), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels, 
const struct CellComm* comm, const struct Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0;
    content_length(&xlen, &comm[l]);
    basis[l].Ulen = xlen;
    int64_t* arr_i = (int64_t*)malloc(sizeof(int64_t) * xlen * 3);
    basis[l].Lchild = arr_i;
    basis[l].Dims = &arr_i[xlen];
    basis[l].DimsLr = &arr_i[xlen * 2];

    struct Matrix* arr_m = (struct Matrix*)calloc(xlen * 3, sizeof(struct Matrix));
    basis[l].Uo = arr_m;
    basis[l].Uc = &arr_m[xlen];
    basis[l].R = &arr_m[xlen * 2];

    int64_t jbegin = 0, jend = ncells;
    int64_t ibegin = 0, iend = xlen;
    get_level(&jbegin, &jend, cells, l, -1);
    self_local_range(&ibegin, &iend, &comm[l]);
    int64_t nodes = iend - ibegin;

    for (int64_t i = 0; i < xlen; i++) {
      int64_t gi = i;
      i_global(&gi, &comm[l]);
      int64_t lc = cells[jbegin + gi].Child;
      int64_t ske = cells[jbegin + gi].Body[1] - cells[jbegin + gi].Body[0];

      if (lc >= 0) {
        lc = lc - jend;
        i_local(&lc, &comm[l + 1]);
        ske = basis[l + 1].DimsLr[lc] + basis[l + 1].DimsLr[lc + 1];
      }
      arr_i[i] = lc;
      arr_i[i + xlen] = ske;
    }

    basis[l].dimN = dist_int_64(basis[l].Dims, 1, &comm[l]);
    std::vector<int64_t> skeletons(basis[l].dimN * xlen);

    for (int64_t i = ibegin; i < iend; i++) {
      int64_t lc = arr_i[i];
      int64_t ske = arr_i[i + xlen];

      if (lc >= 0) {
        const int64_t* m1 = basis[l + 1].Multipoles + basis[l + 1].dimS * lc;
        const int64_t* m2 = basis[l + 1].Multipoles + basis[l + 1].dimS * (lc + 1);
        int64_t len1 = basis[l + 1].DimsLr[lc];
        int64_t len2 = basis[l + 1].DimsLr[lc + 1];
        std::copy(m1, m1 + len1, skeletons.begin() + i * basis[l].dimN);
        std::copy(m2, m2 + len2, skeletons.begin() + i * basis[l].dimN + len1);
      }
      else {
        int64_t gi = i;
        i_global(&gi, &comm[l]);
        std::iota(skeletons.begin() + i * basis[l].dimN, skeletons.begin() + i * basis[l].dimN + ske, cells[jbegin + gi].Body[0]);
      }
    }

    dist_int_64(skeletons.data(), basis[l].dimN, &comm[l]);

    struct SampleBodies samples;
    buildSampleBodies(&samples, sp_pts, sp_pts, nbodies, ncells, cells, rel_near, &basis[l].Lchild[ibegin], l == levels ? NULL : &basis[l + 1], l);

    int64_t count = 0;
    int64_t count_m = 0;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_m = samples.FarLens[i] < samples.CloseLens[i] ? samples.CloseLens[i] : samples.FarLens[i];
      len_m = len_m < ske_len ? ske_len : len_m;
      count = count + ske_len;
      count_m = count_m + ske_len * (ske_len + len_m + 2);
    }

    int32_t* ipiv_data = (int32_t*)malloc(sizeof(int32_t) * count);
    int32_t** ipiv_ptrs = (int32_t**)malloc(sizeof(int32_t*) * nodes);
    double* matrix_data = (double*)malloc(sizeof(double) * count_m);
    double** matrix_ptrs = (double**)malloc(sizeof(double*) * (xlen + 1));

    count = 0;
    count_m = 0;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_m = samples.FarLens[i] < samples.CloseLens[i] ? samples.CloseLens[i] : samples.FarLens[i];
      len_m = len_m < ske_len ? ske_len : len_m;
      ipiv_ptrs[i] = &ipiv_data[count];
      matrix_ptrs[i + ibegin] = &matrix_data[count_m];
      count = count + ske_len;
      count_m = count_m + ske_len * (ske_len + len_m + 2);
    }

#pragma omp parallel for
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_s = samples.FarLens[i] + (samples.CloseLens[i] > 0 ? ske_len : 0);
      double* mat = matrix_ptrs[i + ibegin];
      struct Matrix S = (struct Matrix){ mat, ske_len, len_s, ske_len };

      struct Matrix S_dn = (struct Matrix){ mat, ske_len, ske_len, ske_len };
      double nrm_dn = 0.;
      double nrm_lr = 0.;
      struct Matrix S_dn_work = (struct Matrix){ &mat[ske_len * ske_len], ske_len, samples.CloseLens[i], ske_len };
      gen_matrix(ef, ske_len, samples.CloseLens[i], bodies, bodies, S_dn_work.A, S_dn_work.LDA, &skeletons[(i + ibegin) * basis[l].dimN], samples.CloseBodies[i]);
      mmult('N', 'T', &S_dn_work, &S_dn_work, &S_dn, 1., 0.);
      nrm2_A(&S_dn, &nrm_dn);

      struct Matrix S_lr = (struct Matrix){ &mat[ske_len * ske_len], ske_len, samples.FarLens[i], ske_len };
      gen_matrix(ef, ske_len, samples.FarLens[i], bodies, bodies, S_lr.A, S_lr.LDA, &skeletons[(i + ibegin) * basis[l].dimN], samples.FarBodies[i]);
      nrm2_A(&S_lr, &nrm_lr);
      double scale = (nrm_dn == 0. || nrm_lr == 0.) ? 1. : nrm_lr / nrm_dn;
      scal_A(&S_dn, scale);

      int64_t rank = ske_len < len_s ? ske_len : len_s;
      rank = mrank > 0 ? (mrank < rank ? mrank : rank) : rank;
      double* Svec = &mat[ske_len * len_s];
      svd_U(&S, Svec);

      if (epi > 0.) {
        int64_t r = 0;
        double sepi = Svec[0] * epi;
        while(r < rank && Svec[r] > sepi)
          r += 1;
        rank = r;
      }
      basis[l].DimsLr[i + ibegin] = rank;
      
      int32_t* pa = ipiv_ptrs[i];
      struct Matrix Qo = (struct Matrix){ mat, ske_len, rank, ske_len };
      id_row(&Qo, pa, S_dn_work.A);

      for (int64_t j = 0; j < rank; j++) {
        int64_t piv = (int64_t)pa[j] - 1;
        if (piv != j) { 
          int64_t c = samples.Skeletons[i][piv];
          samples.Skeletons[i][piv] = samples.Skeletons[i][j];
          samples.Skeletons[i][j] = c;
        }
      }

      if (rank > 0) {
        struct Matrix Q = (struct Matrix){ mat, ske_len, ske_len, ske_len };
        struct Matrix R = (struct Matrix){ &mat[ske_len * ske_len], rank, rank, rank };
        int64_t lc = basis[l].Lchild[i + ibegin];
        if (lc >= 0)
          upper_tri_reflec_mult('L', 2, &(basis[l + 1].R)[lc], &Qo);
        qr_full(&Q, &R);
      }
    }

    basis[l].dimS = dist_int_64(basis[l].DimsLr, 1, &comm[l]);

    count_m = 0;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      count_m = count_m + m * m + n * n;
    }

    basis[l].Multipoles = NULL;
    if (basis[l].dimS > 0)
      basis[l].Multipoles = (int64_t*)malloc(sizeof(int64_t) * basis[l].dimS * xlen);
    for (int64_t i = 0; i < nodes; i++) {
      int64_t offset = basis[l].dimS * (i + ibegin);
      int64_t n = basis[l].DimsLr[i + ibegin];
      if (n > 0)
        memcpy(&basis[l].Multipoles[offset], samples.Skeletons[i], sizeof(int64_t) * n);
    }
    dist_int_64(basis[l].Multipoles, basis[l].dimS, &comm[l]);

    double* data_basis = NULL;
    if (count_m > 0)
      data_basis = (double*)malloc(sizeof(int64_t) * count_m);
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      int64_t size = m * m + n * n;
      if (ibegin <= i && i < iend && size > 0)
        memcpy(data_basis, matrix_ptrs[i], sizeof(double) * size);
      basis[l].Uo[i] = (struct Matrix){ data_basis, m, n, m };
      basis[l].Uc[i] = (struct Matrix){ &data_basis[m * n], m, m - n, m };
      basis[l].R[i] = (struct Matrix){ &data_basis[m * m], n, n, n };
      matrix_ptrs[i] = data_basis;
      data_basis = &data_basis[size];
    }
    matrix_ptrs[xlen] = data_basis;
    dist_double(matrix_ptrs, &comm[l]);

    free(ipiv_data);
    free(ipiv_ptrs);
    free(matrix_data);
    free(matrix_ptrs);
    sampleBodies_free(&samples);
  }
}

void basis_free(struct Base* basis) {
  double* data = basis->Uo[0].A;
  if (data)
    free(data);
  if (basis->Multipoles)
    free(basis->Multipoles);
  free(basis->Lchild);
  free(basis->Uo);
}

void evalS(void(*ef)(double*), struct Matrix* S, const struct Base* basis, const struct Body* bodies, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t* multipoles = basis->Multipoles;

#pragma omp parallel for
  for (int64_t x = 0; x < rels->N; x++) {
    int64_t n = basis->DimsLr[x + ibegin];
    int64_t off_x = basis->dimS * (x + ibegin);

    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t m = basis->DimsLr[y];
      int64_t off_y = basis->dimS * y;
      gen_matrix(ef, m, n, bodies, bodies, S[yx].A, S[yx].LDA, &multipoles[off_y], &multipoles[off_x]);
      upper_tri_reflec_mult('L', 1, &basis->R[y], &S[yx]);
      upper_tri_reflec_mult('R', 1, &basis->R[x + ibegin], &S[yx]);
    }
  }
}

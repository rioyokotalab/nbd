
#include <basis.hpp>
#include <linalg.hpp>
#include <comm-mpi.hpp>
#include <build_tree.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

template <typename T>
void memcpy2d(T* dst, const T* src, int64_t rows, int64_t cols, int64_t ld_dst, int64_t ld_src) {
  if (rows == ld_dst && rows == ld_src)
    std::copy(src, src + rows * cols, dst);
  else 
    for (int64_t i = 0; i < cols; i++)
      std::copy(&src[i * ld_src], &src[i * ld_src + rows], &dst[i * ld_dst]);
}

int64_t generate_far(int64_t flen, int64_t far[], int64_t ngbs, const int64_t ngbs_body[], const int64_t ngbs_len[], int64_t nbody) {
  int64_t near = 0;
  for (int64_t i = 0; i < ngbs; i++)
    near = near + ngbs_len[i];
  int64_t avail = nbody - near;
  flen = avail < flen ? avail : flen;
  for (int64_t i = 0; i < flen; i++)
    far[i] = (double)(i * avail) / flen;
  int64_t* begin = far;
  int64_t slen = 0;
  for (int64_t i = 0; i < ngbs; i++) {
    int64_t bound = ngbs_body[i] - slen;
    int64_t* next = begin;
    while (next != far + flen && *next < bound)
      next = next + 1;
    for (int64_t* p = begin; p != next; p++)
      *p = *p + slen;
    begin = next;
    slen = slen + ngbs_len[i];
  }
  for (int64_t* p = begin; p != far + flen; p++)
    *p = *p + near;
  return flen;
}

void buildBasis(const EvalDouble& eval, Base basis[], Cell* cells, const CSR* rel_near, int64_t mrank, int64_t leaf, int64_t branches, int64_t levels,
  const ColCommMPI* comm, const double* bodies, int64_t nbodies, int64_t sp_pts) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0, ibegin = 0, nodes = 0;
    content_length(&nodes, &xlen, &ibegin, &comm[l]);
    int64_t iend = ibegin + nodes;
    basis[l].Dims = std::vector<int64_t>(xlen, 0);
    basis[l].DimsLr = std::vector<int64_t>(xlen, 0);

    Matrix* arr_m = (Matrix*)calloc(xlen * 2, sizeof(Matrix));
    basis[l].Uo = arr_m;
    basis[l].R = &arr_m[xlen];
    std::vector<int64_t> celli(xlen, 0);
    basis[l].LocalChild = std::vector<std::pair<int64_t, int64_t>>(xlen, std::make_pair(-1, -1));
    basis[l].LocalParent = std::vector<std::pair<int64_t, int64_t>>(xlen, std::make_pair(-1, -1));

    for (int64_t i = 0; i < xlen; i++) {
      int64_t gi = comm[l].iGlobal(i);
      int64_t childi = l < levels ? comm[l + 1].iLocal(cells[gi].Child[0]) : -1;
      int64_t clen = cells[gi].Child[1] - cells[gi].Child[0];
      celli[i] = gi;
      basis[l].LocalChild[i] = std::make_pair(childi, clen);

      if (childi >= 0 && l < levels)
        for (int64_t j = 0; j < clen; j++) {
          basis[l].Dims[i] = basis[l].Dims[i] + basis[l + 1].DimsLr[childi + j];
          basis[l + 1].LocalParent[childi + j] = std::make_pair(i, j);
        }
      else
        basis[l].Dims[i] = cells[celli[i]].Body[1] - cells[celli[i]].Body[0];
    }
    neighbor_bcast_sizes_cpu(&basis[l].Dims[0], &comm[l]);
    int64_t seg_dim = l == levels ? leaf : (mrank * branches);
    int64_t seg_skeletons = 3 * seg_dim;
    int64_t seg_matrix = seg_dim * seg_dim * 2;
    std::vector<double> Skeletons(xlen * seg_skeletons, 0.);
    std::vector<double> matrix_data(nodes * seg_matrix, 0.);
    
    if (l < levels) {
      int64_t seg = basis[l + 1].dimS;
      for (int64_t i = 0; i < nodes; i++) {
        int64_t gi = celli[i + ibegin];
        int64_t childi = l < levels ? comm[l + 1].iLocal(cells[gi].Child[0]) : -1;
        int64_t clen = cells[gi].Child[1] - cells[gi].Child[0];
        int64_t y = 0;
        for (int64_t j = 0; j < clen; j++) {
          int64_t len = basis[l + 1].DimsLr[childi + j];
          memcpy(&Skeletons[(i + ibegin) * seg_skeletons + y * 3], &basis[l + 1].M[(childi + j) * seg * 3], len * 3 * sizeof(double));
          memcpy2d(&matrix_data[i * seg_matrix + y * (seg_dim + 1)], 
            &basis[l + 1].R_cpu[(childi + j) * seg * seg], len, len, seg_dim, seg);
          y = y + len;
        }
      }
      neighbor_bcast_cpu(&Skeletons[0], seg_skeletons, &comm[l]);
      dup_bcast_cpu(&Skeletons[0], seg_skeletons * xlen, &comm[l]);
    }
    else 
      for (int64_t i = 0; i < xlen; i++) {
        int64_t len = cells[celli[i]].Body[1] - cells[celli[i]].Body[0];
        int64_t offset = 3 * cells[celli[i]].Body[0];
        memcpy(&Skeletons[i * seg_skeletons], &bodies[offset], len * 3 * sizeof(double));
        if (ibegin <= i && i < iend)
          for (int64_t j = 0; j < len; j++)
            matrix_data[seg_matrix * (i - ibegin) + j * (seg_dim + 1)] = 1.;
      }

    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = basis[l].Dims[i + ibegin];
      double* mat = &matrix_data[seg_matrix * i];
      double* Xbodies = &Skeletons[(i + ibegin) * seg_skeletons];

      int64_t ci = celli[i + ibegin];
      int64_t nbegin = rel_near->RowIndex[ci];
      int64_t nlen = rel_near->RowIndex[ci + 1] - nbegin;
      const int64_t* ngbs = &rel_near->ColIndex[nbegin];
      std::vector<double> Cbodies;
      std::vector<int64_t> remote(sp_pts), body(nlen), lens(nlen);

      for (int64_t j = 0; j < nlen; j++) {
        int64_t cj = ngbs[j];
        body[j] = cells[cj].Body[0];
        lens[j] = cells[cj].Body[1] - cells[cj].Body[0];
        if (cj != ci) {
          int64_t lj = comm[l].iLocal(cj);
          int64_t len = 3 * basis[l].Dims[lj];
          Cbodies.insert(Cbodies.end(), &Skeletons[lj * seg_skeletons], &Skeletons[lj * seg_skeletons + len]);
        }
      }
      int64_t len_f = generate_far(sp_pts, &remote[0], nlen, &body[0], &lens[0], nbodies);

      std::vector<double> Fbodies(len_f * 3);
      for (int64_t j = 0; j < len_f; j++)
        for (int64_t k = 0; k < 3; k++)
          Fbodies[j * 3 + k] = bodies[remote[j] * 3 + k];
      
      compute_basis(eval, mrank, ske_len, mat, seg_dim, &Xbodies[0], Cbodies.size() / 3, &Cbodies[0], Fbodies.size() / 3, &Fbodies[0]);
      basis[l].DimsLr[i + ibegin] = std::min(mrank, ske_len);
    }
    neighbor_bcast_sizes_cpu(basis[l].DimsLr.data(), &comm[l]);

    basis[l].dimN = seg_dim;
    basis[l].dimS = std::min(mrank, seg_dim);
    basis[l].dimR = basis[l].dimN - basis[l].dimS;
    int64_t stride = basis[l].dimN * basis[l].dimN;
    int64_t stride_r = basis[l].dimS * basis[l].dimS;
    int64_t LD = basis[l].dimN;

    basis[l].M = (double*)calloc(basis[l].dimS * xlen * 3, sizeof(double));
    basis[l].U = (double*)calloc(stride * xlen + nodes * basis[l].dimR, sizeof(double));
    basis[l].R_cpu = (double*)calloc(stride_r * xlen, sizeof(double));

    for (int64_t i = 0; i < xlen; i++) {
      double* M_ptr = basis[l].M + i * basis[l].dimS * 3;
      double* Uo_ptr = basis[l].U + i * stride;
      double* Uc_ptr = Uo_ptr + basis[l].dimS * basis[l].dimN;
      double* R_ptr = basis[l].R_cpu + i * stride_r;

      int64_t Nc = basis[l].Dims[i] - basis[l].DimsLr[i];
      int64_t No = basis[l].DimsLr[i];
      int64_t M = basis[l].Dims[i];

      if (ibegin <= i && i < iend) {
        int64_t gi = celli[i];
        int64_t childi = l < levels ? comm[l + 1].iLocal(cells[gi].Child[0]) : -1;
        int64_t clen = cells[gi].Child[1] - cells[gi].Child[0];
        if (childi >= 0 && l < levels) {
          int64_t row = 0;
          for (int64_t j = 0; j < clen; j++) {
            int64_t N = basis[l + 1].DimsLr[childi + j];
            int64_t Urow = j * basis[l + 1].dimS;
            memcpy2d(&Uc_ptr[Urow], &matrix_data[(i - ibegin) * seg_matrix + No * seg_dim + row], N, Nc, LD, seg_dim);
            memcpy2d(&Uo_ptr[Urow], &matrix_data[(i - ibegin) * seg_matrix + row], N, No, LD, seg_dim);
            row = row + N;
          }
        }
        else {
          memcpy2d(Uc_ptr, &matrix_data[(i - ibegin) * seg_matrix + No * seg_dim], M, Nc, LD, seg_dim);
          memcpy2d(Uo_ptr, &matrix_data[(i - ibegin) * seg_matrix], M, No, LD, seg_dim);
        }
        memcpy2d(R_ptr, &matrix_data[(i - ibegin) * seg_matrix + M * seg_dim], No, No, basis[l].dimS, seg_dim);
        memcpy(M_ptr, &Skeletons[i * seg_skeletons], 3 * No * sizeof(double));

        double* Ui_ptr = basis[l].U + xlen * stride + (i - ibegin) * basis[l].dimR; 
        for (int64_t j = Nc; j < basis[l].dimR; j++)
          Ui_ptr[j] = 1.;
      }

      basis[l].Uo[i] = (Matrix) { Uo_ptr, basis[l].dimN, basis[l].dimS, basis[l].dimN };
      basis[l].R[i] = (Matrix) { R_ptr, basis[l].dimS, basis[l].dimS, basis[l].dimS };
    }
    neighbor_bcast_cpu(basis[l].M, 3 * basis[l].dimS, &comm[l]);
    dup_bcast_cpu(basis[l].M, 3 * basis[l].dimS * xlen, &comm[l]);
    neighbor_bcast_cpu(basis[l].U, stride, &comm[l]);
    dup_bcast_cpu(basis[l].U, stride * xlen, &comm[l]);
    neighbor_bcast_cpu(basis[l].R_cpu, stride_r, &comm[l]);
    dup_bcast_cpu(basis[l].R_cpu, stride_r * xlen, &comm[l]);
  }
}


void basis_free(Base* basis) {
  free(basis->Uo);
  if (basis->M)
    free(basis->M);
  if (basis->U)
    free(basis->U);
  if (basis->R_cpu)
    free(basis->R_cpu);
}

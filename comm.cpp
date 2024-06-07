
#include <comm.hpp>
#include <build_tree.hpp>

#include <algorithm>
#include <numeric>
#include <cmath>

MPI_Comm MPI_Comm_split_unique(std::vector<MPI_Comm>& unique_comms, int color, int mpi_rank) {
  MPI_Comm comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, color, mpi_rank, &comm);

  if (comm != MPI_COMM_NULL) {
    auto iter = std::find_if(unique_comms.begin(), unique_comms.end(), [comm](MPI_Comm c) -> bool { 
      int result; MPI_Comm_compare(comm, c, &result); return result == MPI_IDENT || result == MPI_CONGRUENT; });
    if (iter == unique_comms.end())
      unique_comms.emplace_back(comm);
    else {
      MPI_Comm_free(&comm);
      comm = *iter;
    }
  }
  return comm;
}

std::pair<int64_t, int64_t> local_to_pnx(int64_t ilocal, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  int64_t iter = 0;
  while (iter < (int64_t)ProcBoxes.size() && std::get<1>(ProcBoxes[iter]) <= ilocal) {
    ilocal = ilocal - std::get<1>(ProcBoxes[iter]);
    iter = iter + 1;
  }
  if (0 <= ilocal && iter < (int64_t)ProcBoxes.size())
    return std::make_pair(iter, ilocal);
  else
    return std::make_pair(-1, -1);
}

std::pair<int64_t, int64_t> global_to_pnx(int64_t iglobal, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  int64_t iter = 0;
  while (iter < (int64_t)ProcBoxes.size() && (std::get<0>(ProcBoxes[iter]) + std::get<1>(ProcBoxes[iter])) <= iglobal)
    iter = iter + 1;
  if (iter < (int64_t)ProcBoxes.size() && std::get<0>(ProcBoxes[iter]) <= iglobal)
    return std::make_pair(iter, iglobal - std::get<0>(ProcBoxes[iter]));
  else
    return std::make_pair(-1, -1);
}

int64_t pnx_to_local(std::pair<int64_t, int64_t> pnx, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  if (std::get<0>(pnx) >= 0 && std::get<0>(pnx) < (int64_t)ProcBoxes.size() && std::get<1>(pnx) >= 0) {
    int64_t iter = 0, slen = 0;
    while (iter < std::get<0>(pnx)) {
      slen = slen + std::get<1>(ProcBoxes[iter]);
      iter = iter + 1;
    }
    return std::get<1>(pnx) + slen;
  }
  else
    return -1;
}

int64_t pnx_to_global(std::pair<int64_t, int64_t> pnx, const std::vector<std::pair<int64_t, int64_t>>& ProcBoxes) {
  if (std::get<0>(pnx) >= 0 && std::get<0>(pnx) < (int64_t)ProcBoxes.size() && std::get<1>(pnx) >= 0)
    return std::get<1>(pnx) + std::get<0>(ProcBoxes[std::get<0>(pnx)]);
  else
    return -1;
}

void get_level_procs(std::vector<std::pair<int64_t, int64_t>>& Procs, std::vector<std::pair<int64_t, int64_t>>& Levels, 
  int64_t mpi_rank, int64_t mpi_size, const std::vector<std::pair<int64_t, int64_t>>& Child, int64_t levels) {
  int64_t ncells = (int64_t)Child.size();
  std::vector<int64_t> levels_cell(ncells);
  Procs[0] = std::make_pair(0, mpi_size);
  levels_cell[0] = 0;

  for (int64_t i = 0; i < ncells; i++) {
    int64_t child = std::get<0>(Child[i]);
    int64_t lenC = std::get<1>(Child[i]);
    int64_t lenP = std::get<1>(Procs[i]) - std::get<0>(Procs[i]);
    int64_t p = std::get<0>(Procs[i]);
    
    if (child >= 0 && lenC > 0) {
      double divP = (double)lenP / (double)lenC;
      for (int64_t j = 0; j < lenC; j++) {
        int64_t p0 = j == 0 ? 0 : (int64_t)std::floor(j * divP);
        int64_t p1 = j == (lenC - 1) ? lenP : (int64_t)std::floor((j + 1) * divP);
        p1 = std::max(p1, p0 + 1);
        Procs[child + j] = std::make_pair(p + p0, p + p1);
        levels_cell[child + j] = levels_cell[i] + 1;
      }
    }
  }
  
  int64_t begin = 0;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t ibegin = std::distance(levels_cell.begin(), 
      std::find(levels_cell.begin() + begin, levels_cell.end(), i));
    int64_t iend = std::distance(levels_cell.begin(), 
      std::find(levels_cell.begin() + begin, levels_cell.end(), i + 1));
    int64_t pbegin = std::distance(Procs.begin(), 
      std::find_if(Procs.begin() + ibegin, Procs.begin() + iend, [=](std::pair<int64_t, int64_t>& p) -> bool {
        return std::get<0>(p) <= mpi_rank && mpi_rank < std::get<1>(p);
      }));
    int64_t pend = std::distance(Procs.begin(), 
      std::find_if_not(Procs.begin() + pbegin, Procs.begin() + iend, [=](std::pair<int64_t, int64_t>& p) -> bool {
        return std::get<0>(p) <= mpi_rank && mpi_rank < std::get<1>(p);
      }));
    Levels[i] = std::make_pair(pbegin, pend);
    begin = iend;
  }
}

void buildComm(CellComm* comms, int64_t ncells, const Cell* cells, const CSR* cellFar, const CSR* cellNear, int64_t levels) {
  int __mpi_rank = 0, __mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  int64_t mpi_rank = __mpi_rank;
  int64_t mpi_size = __mpi_size;

  std::vector<MPI_Comm> unique_comms;
  std::vector<std::pair<int64_t, int64_t>> Child(ncells), Procs(ncells), Levels(levels + 1);
  std::transform(cells, &cells[ncells], Child.begin(), [](const Cell& c) {
    return std::make_pair(c.Child[0], c.Child[1] - c.Child[0]);
  });
  get_level_procs(Procs, Levels, mpi_rank, mpi_size, Child, levels);

  for (int64_t i = levels; i >= 0; i--) {
    int64_t mbegin = std::get<0>(Levels[i]);
    int64_t mend = std::get<1>(Levels[i]);
    int64_t p = std::get<0>(Procs[mbegin]);
    int64_t lenp = std::get<1>(Procs[mbegin]) - p;

    std::vector<int64_t> ProcTargets;
    for (int64_t j = 0; j < mpi_size; j++) {
      int is_ngb = 0;
      for (int64_t k = cellNear->RowIndex[mbegin]; k < cellNear->RowIndex[mend]; k++)
        if (std::get<0>(Procs[cellNear->ColIndex[k]]) == j)
          is_ngb = 1;
      for (int64_t k = cellFar->RowIndex[mbegin]; k < cellFar->RowIndex[mend]; k++)
        if (std::get<0>(Procs[cellFar->ColIndex[k]]) == j)
          is_ngb = 1;
      
      int color = (is_ngb && p == mpi_rank) ? 1 : MPI_UNDEFINED;
      MPI_Comm comm = MPI_Comm_split_unique(unique_comms, color, mpi_rank);

      if (comm != MPI_COMM_NULL) {
        int root = 0;
        if (j == p)
          MPI_Comm_rank(comm, &root);
        MPI_Allreduce(MPI_IN_PLACE, &root, 1, MPI_INT, MPI_SUM, comm);
        comms[i].Comm_box.emplace_back(root, comm);
      }
      if (is_ngb)
        ProcTargets.emplace_back(j);
    }
    comms[i].Proc = std::distance(ProcTargets.begin(), std::find(ProcTargets.begin(), ProcTargets.end(), p));

    int color = MPI_UNDEFINED;
    int64_t cc = std::get<0>(Child[mbegin]);
    int64_t clen = std::get<1>(Child[mbegin]);
    if (lenp > 1 && cc >= 0)
      for (int64_t j = 0; j < clen; j++)
        if (std::get<0>(Procs[cc + j]) == mpi_rank)
          color = p;
    comms[i].Comm_merge = MPI_Comm_split_unique(unique_comms, color, mpi_rank);
  
    color = lenp > 1 ? p : MPI_UNDEFINED;
    comms[i].Comm_share = MPI_Comm_split_unique(unique_comms, color, mpi_rank);

    std::pair<int64_t, int64_t> local = std::make_pair(mbegin, mend - mbegin);
    comms[i].ProcBoxes = std::vector<std::pair<int64_t, int64_t>>(ProcTargets.size(), local);

    for (int64_t j = 0; j < (int64_t)comms[i].Comm_box.size(); j++)
      MPI_Bcast(&comms[i].ProcBoxes[j], sizeof(std::pair<int64_t, int64_t>), MPI_BYTE, std::get<0>(comms[i].Comm_box[j]), std::get<1>(comms[i].Comm_box[j]));
    if (comms[i].Comm_share != MPI_COMM_NULL)
      MPI_Bcast(&comms[i].ProcBoxes[0], sizeof(std::pair<int64_t, int64_t>) * comms[i].ProcBoxes.size(), MPI_BYTE, 0, comms[i].Comm_share);

    for (int64_t j = 0; j < (int64_t)comms[i].ProcBoxes.size(); j++)
      for (int64_t k = 0; k < std::get<1>(comms[i].ProcBoxes[j]); k++) {
        int64_t ki = k + std::get<0>(comms[i].ProcBoxes[j]);
        int64_t li = pnx_to_local(std::make_pair(j, k), comms[i].ProcBoxes);
        int64_t lc = std::get<0>(Child[ki]);
        int64_t lclen = std::get<1>(Child[ki]);
        if (i < levels) {
          std::pair<int64_t, int64_t> pnx = global_to_pnx(lc, comms[i + 1].ProcBoxes);
          lc = pnx_to_local(pnx, comms[i + 1].ProcBoxes);
          if (lc >= 0)
            std::for_each(comms[i + 1].LocalParent.begin() + lc, comms[i + 1].LocalParent.begin() + (lc + lclen), 
              [&](std::pair<int64_t, int64_t>& x) { std::get<0>(x) = li; std::get<1>(x) = std::distance(&comms[i + 1].LocalParent[lc], &x); });
          else
            lclen = 0;
        }
        comms[i].LocalChild.emplace_back(lc, lclen);
      }
    comms[i].LocalParent = std::vector<std::pair<int64_t, int64_t>>(comms[i].LocalChild.size(), std::make_pair(-1, -1));
  }
}

void cellComm_free(CellComm* comms, int64_t levels) {
  std::vector<MPI_Comm> mpi_comms;

  for (int64_t i = 0; i <= levels; i++) {
    for (int64_t j = 0; j < (int64_t)comms[i].Comm_box.size(); j++)
      mpi_comms.emplace_back(std::get<1>(comms[i].Comm_box[j]));
    if (comms[i].Comm_merge != MPI_COMM_NULL)
      mpi_comms.emplace_back(comms[i].Comm_merge);
    if (comms[i].Comm_share != MPI_COMM_NULL)
      mpi_comms.emplace_back(comms[i].Comm_share);
    
    comms[i].Comm_box.clear();
    comms[i].Comm_merge = MPI_COMM_NULL;
    comms[i].Comm_share = MPI_COMM_NULL;
  }

  std::sort(mpi_comms.begin(), mpi_comms.end());
  mpi_comms.erase(std::unique(mpi_comms.begin(), mpi_comms.end()), mpi_comms.end());

  for (int64_t i = 0; i < (int64_t)mpi_comms.size(); i++)
    MPI_Comm_free(&mpi_comms[i]);
}

void relations(CSR rels[], const CSR* cellRel, int64_t levels, const CellComm* comm) {
 
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes, neighbors, ibegin;
    content_length(&nodes, &neighbors, &ibegin, &comm[i]);
    ibegin = comm[i].iGlobal(ibegin);
    CSR* csc = &rels[i];

    csc->M = neighbors;
    csc->N = nodes;
    int64_t ent_max = nodes * csc->M;
    csc->RowIndex.resize(nodes + 1);
    csc->ColIndex.resize(ent_max);

    int64_t count = 0;
    for (int64_t j = 0; j < nodes; j++) {
      int64_t lc = ibegin + j;
      csc->RowIndex[j] = count;
      int64_t cbegin = cellRel->RowIndex[lc];
      int64_t ent = cellRel->RowIndex[lc + 1] - cbegin;
      for (int64_t k = 0; k < ent; k++) {
        csc->ColIndex[count + k] = comm[i].iLocal(cellRel->ColIndex[cbegin + k]);
      }
      count = count + ent;
    }

    if (count < ent_max)
      csc->ColIndex.resize(count);
    csc->RowIndex[nodes] = count;
  }
}

int64_t CellComm::iLocal(int64_t iglobal) const {
  return pnx_to_local(global_to_pnx(iglobal, ProcBoxes), ProcBoxes);
}

int64_t CellComm::iGlobal(int64_t ilocal) const {
  return pnx_to_global(local_to_pnx(ilocal, ProcBoxes), ProcBoxes);
}

void content_length(int64_t* local, int64_t* neighbors, int64_t* local_off, const CellComm* comm) {
  int64_t slen = 0, offset = -1, len_self = -1;
  for (int64_t i = 0; i < (int64_t)comm->ProcBoxes.size(); i++) {
    if (i == comm->Proc)
    { offset = slen; len_self = std::get<1>(comm->ProcBoxes[i]); }
    slen = slen + std::get<1>(comm->ProcBoxes[i]);
  }
  if (local)
    *local = len_self;
  if (neighbors)
    *neighbors = slen;
  if (local_off)
    *local_off = offset;
}

int64_t neighbor_bcast_sizes_cpu(int64_t* data, const CellComm* comm) {
  int64_t max = 0;
  if (comm->Comm_box.size() > 0 || comm->Comm_share != MPI_COMM_NULL) {
    double start_time = MPI_Wtime();
    int64_t y = 0;
    for (int64_t p = 0; p < (int64_t)comm->Comm_box.size(); p++) {
      int64_t llen = std::get<1>(comm->ProcBoxes[p]);
      int64_t* loc = &data[y];
      MPI_Bcast(loc, llen, MPI_INT64_T, std::get<0>(comm->Comm_box[p]), std::get<1>(comm->Comm_box[p]));
      y = y + llen;
    }
    content_length(NULL, &y, NULL, comm);
    if (comm->Comm_share != MPI_COMM_NULL)
      MPI_Bcast(data, y, MPI_DOUBLE, 0, comm->Comm_share);

    for (int64_t i = 0; i < y; i++)
      max = std::max(max, data[i]);
    MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
    if (comm->timer)
      comm->timer->record_mpi(start_time, MPI_Wtime());
  }
  return max;
}

void neighbor_bcast_cpu(double* data, int64_t seg, const CellComm* comm) {
  if (comm->Comm_box.size() > 0) {
    double start_time = MPI_Wtime();
    int64_t y = 0;
    for (int64_t p = 0; p < (int64_t)comm->Comm_box.size(); p++) {
      int64_t llen = std::get<1>(comm->ProcBoxes[p]) * seg;
      double* loc = &data[y];
      MPI_Bcast(loc, llen, MPI_DOUBLE, std::get<0>(comm->Comm_box[p]), std::get<1>(comm->Comm_box[p]));
      y = y + llen;
    }
    if (comm->timer)
      comm->timer->record_mpi(start_time, MPI_Wtime());
  }
}

void neighbor_reduce_cpu(double* data, int64_t seg, const CellComm* comm) {
  if (comm->Comm_box.size() > 0) {
    double start_time = MPI_Wtime();
    int64_t y = 0;
    for (int64_t p = 0; p < (int64_t)comm->Comm_box.size(); p++) {
      int64_t llen = std::get<1>(comm->ProcBoxes[p]) * seg;
      double* loc = &data[y];
      if (p == comm->Proc)
        MPI_Reduce(MPI_IN_PLACE, loc, llen, MPI_DOUBLE, MPI_SUM, std::get<0>(comm->Comm_box[p]), std::get<1>(comm->Comm_box[p]));
      else
        MPI_Reduce(loc, loc, llen, MPI_DOUBLE, MPI_SUM, std::get<0>(comm->Comm_box[p]), std::get<1>(comm->Comm_box[p]));
      y = y + llen;
    }
    if (comm->timer)
      comm->timer->record_mpi(start_time, MPI_Wtime());
  }
}

void level_merge_cpu(double* data, int64_t len, const CellComm* comm) {
  if (comm->Comm_merge != MPI_COMM_NULL) {
    double start_time = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, data, len, MPI_DOUBLE, MPI_SUM, comm->Comm_merge);
    if (comm->timer)
      comm->timer->record_mpi(start_time, MPI_Wtime());
  }
}

void dup_bcast_cpu(double* data, int64_t len, const CellComm* comm) {
  if (comm->Comm_share != MPI_COMM_NULL) {
    double start_time = MPI_Wtime();
    MPI_Bcast(data, len, MPI_DOUBLE, 0, comm->Comm_share);
    if (comm->timer)
      comm->timer->record_mpi(start_time, MPI_Wtime());
  }
}



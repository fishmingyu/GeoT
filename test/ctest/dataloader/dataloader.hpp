#ifndef DATA_LOADER
#define DATA_LOADER

#include "../util/check.cuh"
#include "../util/ramArray.cuh"
#include "mmio.hpp"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <tuple>
#include <typeinfo>
#include <vector>

// Load sparse matrix from an mtx file. Only non-zero positions are loaded,
// and values are dropped.
void read_mtx_file(const char *filename, int &nrow, int &ncol, int &nnz,
                   std::vector<int> &csr_indptr_buffer,
                   std::vector<int> &csr_indices_buffer,
                   std::vector<int> &coo_rowind_buffer) {
  FILE *f;

  if ((f = fopen(filename, "r")) == NULL) {
    printf("File %s not found", filename);
    exit(EXIT_FAILURE);
  }

  MM_typecode matcode;
  // Read MTX banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  if (mm_read_mtx_crd_size(f, &nrow, &ncol, &nnz) != 0) {
    printf("Could not process this file.\n");
    exit(EXIT_FAILURE);
  }
  // printf("Reading matrix %d rows, %d columns, %d nnz.\n", nrow, ncol, nnz);

  /// read tuples

  std::vector<std::tuple<int, int>> coords;
  int row_id, col_id;
  float dummy;
  for (int64_t i = 0; i < nnz; i++) {
    if (fscanf(f, "%d", &row_id) == EOF) {
      std::cout << "Error: not enough rows in mtx file.\n";
      exit(EXIT_FAILURE);
    } else {
      fscanf(f, "%d", &col_id);
      if (mm_is_integer(matcode) || mm_is_real(matcode)) {
        fscanf(f, "%f", &dummy);
      } else if (mm_is_complex(matcode)) {
        fscanf(f, "%f", &dummy);
        fscanf(f, "%f", &dummy);
      }
      // mtx format is 1-based
      coords.push_back(std::make_tuple(row_id - 1, col_id - 1));
    }
  }

  /// make symmetric

  if (mm_is_symmetric(matcode)) {
    std::vector<std::tuple<int, int>> new_coords;
    for (auto iter = coords.begin(); iter != coords.end(); iter++) {
      int i = std::get<0>(*iter);
      int j = std::get<1>(*iter);
      if (i != j) {
        new_coords.push_back(std::make_tuple(i, j));
        new_coords.push_back(std::make_tuple(j, i));
      } else
        new_coords.push_back(std::make_tuple(i, j));
    }
    std::sort(new_coords.begin(), new_coords.end());
    coords.clear();
    for (auto iter = new_coords.begin(); iter != new_coords.end(); iter++) {
      if ((iter + 1) == new_coords.end() || (*iter != *(iter + 1))) {
        coords.push_back(*iter);
      }
    }
  } else {
    std::sort(coords.begin(), coords.end());
  }
  /// generate csr from coo

  csr_indptr_buffer.clear();
  csr_indices_buffer.clear();

  int curr_pos = 0;
  csr_indptr_buffer.push_back(0);
  for (int64_t row = 0; row < nrow; row++) {
    while ((curr_pos < nnz) && (std::get<0>(coords[curr_pos]) == row)) {
      csr_indices_buffer.push_back(std::get<1>(coords[curr_pos]));
      coo_rowind_buffer.push_back(std::get<0>(coords[curr_pos]));
      curr_pos++;
    }
    // assert((std::get<0>(coords[curr_pos]) > row || curr_pos == nnz));
    csr_indptr_buffer.push_back(curr_pos);
  }
  nnz = csr_indices_buffer.size();
}

template <typename Index>
void compressedRow(Index nrow, std::vector<Index> &rowind,
                   std::vector<Index> &rowptr) {
  int curr_pos = 0;
  rowptr.push_back(0);
  int nnz = rowind.size();
  for (int64_t row = 0; row < nrow; row++) {
    while ((curr_pos < nnz) && ((rowind[curr_pos]) == row)) {
      curr_pos++;
    }
    rowptr.push_back(curr_pos);
  }
}

template <typename Index>
void transpose(Index ncol, std::vector<Index> &row, std::vector<Index> &col,
               std::vector<Index> &row_t, std::vector<Index> &col_t) {
  int nnz = col.size();
  Index *hist = new Index[ncol];
  Index *col_tmp = new Index[ncol + 1];
  memset(hist, 0x0, sizeof(Index) * ncol);
  for (int t = 0; t < nnz; t++)
    hist[col[t]]++;
  col_tmp[0] = 1;
  for (int c = 1; c <= ncol; ++c)
    col_tmp[c] = col_tmp[c - 1] + hist[c - 1];
  for (int nid = 0; nid < nnz; nid++) {
    int col_ = col[nid];
    int q = col_tmp[col_];
    row_t[q - 1] = col[nid];
    col_t[q - 1] = row[nid];
    col_tmp[col_]++;
  }
  delete[] hist;
  delete[] col_tmp;
}

template <class Index, class DType> struct SpMatCsrDescr_t {
  SpMatCsrDescr_t(int ncol_, std::vector<Index> &indptr,
                  std::vector<Index> &indices) {
    nrow = indptr.size() - 1;
    ncol = ncol_;
    nnz = indices.size();
    sp_csrptr.create(nrow + 1, indptr);
    sp_csrind.create(nnz, indices);
    sp_data.create(nnz);
    sp_data.fill_default_one();
  }
  void upload() {
    sp_csrptr.upload();
    sp_csrind.upload();
    sp_data.upload();
  }
  int nrow;
  int ncol;
  int nnz;
  util::RamArray<Index> sp_csrptr;
  util::RamArray<Index> sp_csrind;
  util::RamArray<DType> sp_data;
};

template <class Index, class DType>
std::tuple<SpMatCsrDescr_t<Index, DType>, SpMatCsrDescr_t<Index, DType>>
DataLoader(const char *filename) {
  int H_nrow, H_ncol, H_nnz;
  std::vector<Index> H_csrptr, H_csrind, H_coorow;
  read_mtx_file(filename, H_nrow, H_ncol, H_nnz, H_csrptr, H_csrind, H_coorow);
  printf("H_t.nrow %d H_t.ncol %d H_nnz %d\n", H_nrow, H_ncol, H_nnz);
  std::vector<Index> H_t_csrptr, H_t_csrind(H_nnz, 0), H_t_coorow(H_nnz, 0);
  transpose<Index>(H_ncol, H_coorow, H_csrind, H_t_coorow, H_t_csrind);
  compressedRow(H_ncol, H_t_coorow, H_t_csrptr);
  SpMatCsrDescr_t<Index, DType> H(H_ncol, H_csrptr, H_csrind),
      H_t(H_nrow, H_t_csrptr, H_t_csrind);
  return {H, H_t};
}

template <class Index, class DType>
SpMatCsrDescr_t<Index, DType> SingleDataLoader(const char *filename) {
  int H_nrow, H_ncol, H_nnz;
  std::vector<Index> H_csrptr, H_csrind, H_coorow;
  read_mtx_file(filename, H_nrow, H_ncol, H_nnz, H_csrptr, H_csrind, H_coorow);
  SpMatCsrDescr_t<Index, DType> H(H_ncol, H_csrptr, H_csrind);
  return H;
}



#endif
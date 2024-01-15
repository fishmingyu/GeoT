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
#include <random>

// Load sparse matrix from an mtx file. Only non-zero positions are loaded,
// and values are dropped.
void read_mtx_file(const char *filename, int &nrow, int &ncol, int &nnz,
                   std::vector<int64_t> &csr_indptr_buffer,
                   std::vector<int64_t> &csr_indices_buffer,
                   std::vector<int64_t> &coo_rowind_buffer) {
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

  std::vector<std::tuple<int64_t, int64_t>> coords;
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
    std::vector<std::tuple<int64_t, int64_t>> new_coords;
    for (auto iter = coords.begin(); iter != coords.end(); iter++) {
      int64_t i = std::get<0>(*iter);
      int64_t j = std::get<1>(*iter);
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

  int64_t curr_pos = 0;
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

template <typename IndexType>
void compressedRow(IndexType nrow, std::vector<IndexType> &rowind,
                   std::vector<IndexType> &rowptr) {
  int64_t curr_pos = 0;
  rowptr.push_back(0);
  int64_t nnz = rowind.size();
  for (int64_t row = 0; row < nrow; row++) {
    while ((curr_pos < nnz) && ((rowind[curr_pos]) == row)) {
      curr_pos++;
    }
    rowptr.push_back(curr_pos);
  }
}

template <typename IndexType>
void transpose(int ncol, std::vector<IndexType> &row,
               std::vector<IndexType> &col, std::vector<IndexType> &row_t,
               std::vector<IndexType> &col_t) {
  int64_t nnz = col.size();
  IndexType *hist = new IndexType[ncol];
  IndexType *col_tmp = new IndexType[ncol + 1];
  memset(hist, 0x0, sizeof(IndexType) * ncol);
  for (int64_t t = 0; t < nnz; t++)
    hist[col[t]]++;
  col_tmp[0] = 1;
  for (int64_t c = 1; c <= ncol; ++c)
    col_tmp[c] = col_tmp[c - 1] + hist[c - 1];
  for (int64_t nid = 0; nid < nnz; nid++) {
    int64_t col_ = col[nid];
    int64_t q = col_tmp[col_];
    row_t[q - 1] = col[nid];
    col_t[q - 1] = row[nid];
    col_tmp[col_]++;
  }
  delete[] hist;
  delete[] col_tmp;
}

template <class IndexType, class ValueType> struct SpMatCsrDescr_t {
  SpMatCsrDescr_t(int64_t ncol_, std::vector<IndexType> &indptr,
                  std::vector<IndexType> &indices) {
    nrow = indptr.size() - 1;
    ncol = ncol_;
    nnz = indices.size();
    sp_csrptr.create(nrow + 1, indptr);
    sp_csrind.create(nnz, indices);
    sp_data.create(nnz);
    sp_data.fill_default_one();
  }
  void tocuda() {
    sp_csrptr.tocuda();
    sp_csrind.tocuda();
    sp_data.tocuda();
  }
  int64_t nrow;
  int64_t ncol;
  int64_t nnz;
  util::RamArray<IndexType> sp_csrptr;
  util::RamArray<IndexType> sp_csrind;
  util::RamArray<ValueType> sp_data;
};

template <class IndexType, class ValueType> struct SpMatCooDescr_t {
  SpMatCooDescr_t(int64_t ncol_, std::vector<IndexType> &rowind,
                  std::vector<IndexType> &indices) {
    nnz = indices.size();
    sp_rowind.create(nnz, rowind);
    sp_colind.create(nnz, indices);
  }
  void tocuda() {
    sp_rowind.tocuda();
    sp_colind.tocuda();
  }
  int64_t nnz;
  util::RamArray<IndexType> sp_rowind;
  util::RamArray<IndexType> sp_colind;
  util::RamArray<ValueType> sp_data;
};

template <class IndexType> struct IndexDescr_t {
  IndexDescr_t(std::vector<IndexType> &ptr, std::vector<IndexType> &indices) {
    keys = ptr.size() - 1;
    nnz = indices.size();
    sp_indices.create(nnz, indices);
  }
  void tocuda() { sp_indices.tocuda(); }
  int64_t nnz;
  int64_t keys;
  util::RamArray<IndexType> sp_indices;
};

template <typename ValueType, typename IndexType>
IndexDescr_t<IndexType> DataLoader(const char *filename) {
  int nrow, ncol, nnz;
  std::vector<IndexType> csrptr, col, row;
  read_mtx_file(filename, nrow, ncol, nnz, csrptr, col, row);
  IndexDescr_t<IndexType> indexDescr(csrptr, row);
  return indexDescr; // sorted row index
}

template <typename IndexType>
int generateIndex(int range, int min_seg, int max_seg, int total_count, double cv, std::vector<IndexType>& result) {
    // range: Elements of index `i` in [0, range)
    // max_seg: Maximum repetition `mi` of each element of index `i`
    // avg: Desired average of `mi`
    double avg = static_cast<double>(total_count) / range;
    result.resize(total_count);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(avg, avg * cv); // mean max_seg/2, std dev max_seg/4

    int current_sum = 0;
    int dst_len = 0;
    for (int i = 0; i < range; ++i) {
        int mi = static_cast<int>(std::round(distribution(generator)));

        // Ensure mi is within bounds
        mi = std::max(mi, min_seg);
        mi = std::min(mi, max_seg);

        // Adjust the last element to match the desired total count
        if (i == range - 1) {
            mi = total_count - current_sum;
        }

        for (int j = 0; j < mi; ++j) {
            result[current_sum + j] = i;
        }
        current_sum += mi;
        if (mi > 0) dst_len += 1;

        // Early exit if we reached the total count
        if (current_sum >= total_count) break;
    }
    std::cout << "range = " << range << ", nnz = " << total_count << ", max_seg = " << max_seg << " cv = " << cv << ", dst_len = " << dst_len << std::endl;
    return dst_len;
}

#endif
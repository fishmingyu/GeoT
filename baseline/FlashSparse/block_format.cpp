/* This file is revised from FlashSparse repo.
 * https://github.com/ParCIS/FlashSparse/tree/main/FlashSparse/Block */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <set>
#include <torch/torch.h>
#include <vector>

struct Value_tf32 {
  std::vector<float> value;
  std::vector<int> colum;
  int row;
  int pad;
};

std::vector<torch::Tensor> blockProcess_tf32(torch::Tensor row1,
                                             torch::Tensor column1,
                                             torch::Tensor degree1, int window1,
                                             int wide1) {
  // auto start = std::chrono::high_resolution_clock::now();
  int window = window1;
  int wide = wide1;
  auto *row = row1.data_ptr<int>();
  auto column = column1.accessor<int, 1>();
  auto degree = degree1.accessor<float, 1>();

  int rows = row1.size(0) - 1;
  int rowsNew = rows / window;
  std::map<int, Value_tf32> res;
#pragma omp parallel for
  for (int i = 0; i < rowsNew; i++) {
    Value_tf32 v;

    std::vector<int> mergedVector(&column[row[i * window]],
                                  &column[row[i * window + window]]);
    std::unordered_map<int, int> elementCounts;
    for (const auto &element : mergedVector) {
      elementCounts[element]++;
    }
    std::vector<std::pair<int, int>> countVector(elementCounts.begin(),
                                                 elementCounts.end());
    std::sort(countVector.begin(), countVector.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    v.row = countVector.size();
    v.pad = ((v.row / wide + 1)) * wide;
    if (v.row % wide == 0)
      v.pad -= wide;

    if (v.row > 0) {
      std::map<int, int> colmap;
      int c = 0;
      for (auto col : countVector) {
        v.colum.push_back(col.first);
        colmap[col.first] = c++;
      }

      std::vector<float> demo;
      demo.resize(v.row * window);
      int bIds = v.pad / wide;
      int p = 0;
      for (int j = i * window; j < (i + 1) * window; j++) {
        for (int m = row[j]; m < row[j + 1]; m++) {
          int bId = colmap[column[m]] / wide;
          int bInId = colmap[column[m]] % wide;
          if (v.row % wide == 0) {
            demo[bId * window * wide + p * wide + bInId] = degree[m];
          } else {
            if (bId < (bIds - 1))
              demo[bId * window * wide + p * wide + bInId] = degree[m];
            else
              demo[bId * window * wide + p * (v.row % wide) + bInId] =
                  degree[m];
          }
        }
        p++;
      }
      v.value = demo;
    }

#pragma omp critical
    { res[i] = v; }
  }

  std::vector<int> rowNew;
  rowNew.push_back(0);
  std::vector<int> colNew;
  std::vector<float> valueNew;

  for (const auto &pair : res) {
    rowNew.push_back(rowNew.back() + pair.second.row);
    colNew.insert(colNew.end(), pair.second.colum.begin(),
                  pair.second.colum.end());
    valueNew.insert(valueNew.end(), pair.second.value.begin(),
                    pair.second.value.end());
  }

  auto rowTensor1 =
      torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
  auto colTensor1 =
      torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
  auto valueTensor1 =
      torch::from_blob(valueNew.data(), valueNew.size(), torch::kFloat32);

  torch::Tensor rowTensor = torch::empty_like(rowTensor1);
  rowTensor.copy_(rowTensor1);
  torch::Tensor colTensor = torch::empty_like(colTensor1);
  colTensor.copy_(colTensor1);
  torch::Tensor valueTensor = torch::empty_like(valueTensor1);
  valueTensor.copy_(valueTensor1);

  return {rowTensor, colTensor, valueTensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("blockProcess_tf32", &blockProcess_tf32,
        "Block for TF32 with any shape");
}
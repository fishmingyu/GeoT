#include "./include/npy.hpp"
#include "./include/seg_rule.hpp"

std::vector<long> npy_data_loader(const char *filename) {
  std::string file_path(filename);
  npy::npy_data<long> d = npy::read_npy<long>(file_path);
  std::vector<long> data = d.data;
  return data;
}

int main(int argc, char **argv) {
  // Host problem definition
  if (argc < 3) {
    printf("Input: first get the path of sparse matrix, then get the "
           "feature length of dense matrix\n");
    exit(1);
  }
  char *filename = argv[1];
  int feature_size = atoi(argv[2]);

  auto idx = npy_data_loader(filename);
  int nnz = idx.size();
  int keys = idx.back() + 1;

  util::RamArray<long> indices;
  indices.create(nnz, idx);

  util::RamArray<float> src(nnz * feature_size);
  util::RamArray<float> dst(keys * feature_size);

  src.fill_random_h();
  dst.fill_zero_h();
  // to GPU
  src.tocuda();
  dst.tocuda();
  indices.tocuda();
  // print the file name
  printf("start index scatter test for: %s, N: %d\n", filename, feature_size);
  cudaDeviceSynchronize();
  // warm up
  for (int i = 0; i < 1000; i++)
    warm_up<<<1, 1>>>();
  // create a csv file to store the result
  std::ofstream rule_file;
  rule_file.open("rule_result.csv", std::ios::app);
  segreduce_naive_rule<float>(rule_file, filename, nnz, feature_size,
                              keys, indices, src, dst);
  segreduce_dtree_rule<float>(rule_file, filename, nnz, feature_size, keys,
                        indices, src, dst);
  rule_file.close();
  return 0;
}

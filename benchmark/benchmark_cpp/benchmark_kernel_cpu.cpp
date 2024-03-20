//CPU Version
//Author: Xin Chen
//Date: March 6,2024
//Version: 1.0

//This file aims to implement CPU code
std::vector<long> npy_data_loader(const char *filename) {
  std::string file_path(filename);
  npy::npy_data<long> d = npy::read_npy<long>(file_path);
  std::vector<long> data = d.data;
  return data;
}
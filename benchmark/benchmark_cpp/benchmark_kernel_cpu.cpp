//CPU Version
//Author: Xin Chen
//Date: March 6,2024
//Version: 1.0

//This file aims to implement CPU code

#include "./include/npy.hpp"
#include "./include/seg_func.hpp"

std::vector<long> npy_data_loader(const char *filename) {
  std::string file_path(filename);
  npy::npy_data<long> d = npy::read_npy<long>(file_path);
  std::vector<long> data = d.data;
  return data;
}

int main(int argc, char*argv[])
{
  if (argc !=3)
  {
    printf("Input: first get the path of sparse matrix, then get the "
           "feature length of dense matrix\n");
    exit(1);
  }

  char *filename = argv[1];
  

}
NVCC = $(CUDA_HOME)/bin/nvcc
CC_FLAGS = -O3 -Xcompiler "-fPIC" -shared -std=c++14
TEST := test
BENCHMARK := benchmark_mtx
RM = -rm -rf
OBJ_DIR := objs

OBJS = $(wildcard $(OBJ_DIR)/*.o)

NVCC_FLAGS =\
--use_fast_math \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_90,code=sm_90

INC = -I../../csrc/util -I../../csrc/dataloader -I$(CUDA_HOME)/include

all : $(TEST) $(BENCHMARK)

$(TEST): test.o
	$(NVCC) $(INC) $(LL) -o $@ $^

$(BENCHMARK): benchmark_mtx.o
	$(NVCC) $(INC) $(LL) -o $@ $^

%.o: %.cu
	$(NVCC) $(CC_FLAGS) $(NVCC_FLAGS) $(INC) -c $< 

clean:
	rm -f *.o $(TEST) $(BENCHMARK)
	$(RM) $(OBJ_DIR)
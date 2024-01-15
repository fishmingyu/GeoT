NVCC = $(CUDA_HOME)/bin/nvcc
CC_FLAGS = -Xcompiler "-fPIC" -shared -std=c++14
TARGET := test
RM = -rm -rf
OBJ_DIR := objs

OBJS = $(wildcard $(OBJ_DIR)/*.o)
GPU_OPTS = -O3 --std=c++14 -gencode arch=compute_90,code=sm_90 --use_fast_math
INC = -I./util -I./dataloader -I$(CUDA_HOME)/include

all : $(TARGET)

$(TARGET): test.o
	$(NVCC) $(INC) $(LL) -o $@ $^

%.o: %.cu
	$(NVCC) $(CC_FLAGS) $(GPU_OPTS) $(INC) -c $< 

clean:
	rm -f *.o $(TARGET)
	$(RM) -rf $(TARGET) $(OBJ_DIR)
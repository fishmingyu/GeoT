NVCC = $(CUDA_HOME)/bin/nvcc
CC_FLAGS = -O3 -Xcompiler "-fPIC" -shared -std=c++14
TARGET := test
RM = -rm -rf
OBJ_DIR := objs

OBJS = $(wildcard $(OBJ_DIR)/*.o)

NVCC_FLAGS =\
--use_fast_math \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_90,code=sm_90

INC = -I./util -I./dataloader -I$(CUDA_HOME)/include

all : $(TARGET)

$(TARGET): test.o
	$(NVCC) $(INC) $(LL) -o $@ $^

%.o: %.cu
	$(NVCC) $(CC_FLAGS) $(NVCC_FLAGS) $(INC) -c $< 

clean:
	rm -f *.o $(TARGET)
	$(RM) -rf $(TARGET) $(OBJ_DIR)
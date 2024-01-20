NVCC = $(CUDA_HOME)/bin/nvcc
CC_FLAGS = -Xcompiler "-fPIC" -shared -std=c++14
TARGET := test
RM = -rm -rf
OBJ_DIR := objs

OBJS = $(wildcard $(OBJ_DIR)/*.o)

INC = -I./util -I./dataloader -I$(CUDA_HOME)/include

all : $(TARGET)

$(TARGET): test.o
	$(NVCC) $(INC) $(LL) -o $@ $^

%.o: %.cu
	$(NVCC) $(CC_FLAGS) $(INC) -c $< 

clean:
	rm -f *.o $(TARGET)
	$(RM) -rf $(TARGET) $(OBJ_DIR)
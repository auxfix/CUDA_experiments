CUDA_HOME ?= /usr/local/cuda-13.3
NVCC := $(if $(wildcard $(CUDA_HOME)/bin/nvcc),$(CUDA_HOME)/bin/nvcc,/usr/bin/nvcc)
TARGET := matrix_multiplication_tensor_cores
SRC := matrix_multipication.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcublas

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

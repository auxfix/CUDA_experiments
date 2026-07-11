CUDA_HOME ?= /usr/local/cuda-13.3
NVCC := $(CUDA_HOME)/bin/nvcc
TARGET := hello_cuda
SRC := hello_cuda.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -std=c++17 -O2 -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

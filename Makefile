SM=$(shell ~/sm)

NVCC=nvcc
NVFLAGS=-O0 --generate-line-info -gencode arch=compute_$(SM),code=[compute_$(SM),sm_$(SM)]

CXX=g++
CXXFLAGS=-O0

all:cudabox cpubox

cudabox:cudabox.cu
	$(NVCC) $(NVFLAGS) $? -o $@

cpubox:cpubox.cpp
	$(CXX) $(CXXFLAGS) $? -o $@

clean:
	rm -f cudabox cpubox


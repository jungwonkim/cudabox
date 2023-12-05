SM=$(shell ~/sm)

NVCC=nvcc
NVFLAGS=-O0 --generate-line-info -gencode arch=compute_$(SM),code=[compute_$(SM),sm_$(SM)]

cudabox:cudabox.cu
	$(NVCC) $(NVFLAGS) $? -o $@

clean:
	rm -f cudabox


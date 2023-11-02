NVCC=nvcc
NVFLAGS=-O0 --generate-line-info -gencode arch=compute_80,code=[compute_80,sm_80]
NVFLAGS=-O0 --generate-line-info -gencode arch=compute_90,code=[compute_90,sm_90]

cudabox:cudabox.cu
	$(NVCC) $(NVFLAGS) $? -o $@

clean:
	rm -f cudabox


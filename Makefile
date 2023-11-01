NVCC=nvcc
NVFLAGS=-O0 --generate-line-info -gencode arch=compute_90,code=[compute_90,sm_90]

cudabox:cudabox.cu
	$(NVCC) $(NVFLAGS) $? -o $@

clean:
	rm -f cudabox


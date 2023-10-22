NVCC=nvcc

cudabox:cudabox.cu
	$(NVCC) --generate-line-info -gencode arch=compute_90,code=compute_90 -gencode arch=compute_90,code=sm_90 $? -o $@

clean:
	rm -f cudabox


NVCC=nvcc

cudabox:cudabox.cu
	$(NVCC) -gencode arch=compute_90,code=compute_90 $? -o $@

run:cudabox
	./$? saxpy sgemm random

clean:
	rm -f cudabox


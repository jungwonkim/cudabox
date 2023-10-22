#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <curand_kernel.h>

#define __SHORT_FILE__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define _cuerror(err) do { if (err != cudaSuccess) { printf("[%s:%d:%s] err[%d][%s]\n", __SHORT_FILE__, __LINE__, __func__, err, cudaGetErrorString(err)); fflush(stdout); } } while (0)
#define _timer(name, t0, t1) do { printf("[%s:%d:%s] %-10s timer[%lf]\n", __SHORT_FILE__, __LINE__, __func__, name, t1 - t0); fflush(stdout); } while (0)

#define MEM_SIZE  (1 * 1024 * 1024 * 1024)

cudaError_t err;

template <typename T>
__global__ void axpy(T* s, T a, T *x, T *y)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = a * x[tid] + y[tid];
}

template <typename T>
void run_axpy(T* s, T a, T* x, T* y) {
  int N = MEM_SIZE / 8;
  int dim_block = 1024;
  int dim_grid  = N / dim_block;

  axpy<T><<<dim_grid, dim_block>>>(s, a, x, y);
}

template <typename T>
__global__ void gemm(T* c, T* a, T* b, int k)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  c[y * k + x] = 0.0;
  for(int i = 0; i < k; i++) {
    c[y * k + x] += a[y * k + i] * b[i * k + x];
  }
}

template <typename T>
void run_gemm(T* c, T* a, T* b) {
  int N = 1024;
  dim3 dim_block(16, 16);
  dim3 dim_grid(N / 16, N / 16);

  gemm<T><<<dim_grid, dim_block>>>(c, a, b, N);
}

template <typename T>
__global__ void rand(T *a, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  curand_init(0xdeadcafe, tid, 0, &state);
  int i = curand(&state) % n;
  a[i] += 1.0f;
}

template <typename T>
void run_rand(T* a) {
  int N = MEM_SIZE / 8 / 64;
  int dim_block = 1024;
  int dim_grid  = N / dim_block;

  rand<T><<<dim_grid, dim_block>>>(a, N);
}

double now() {
  static double base_sec = -1;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  if (base_sec < 0) base_sec = tv.tv_sec + 1.e-6 * tv.tv_usec;
  return tv.tv_sec + 1.e-6 * tv.tv_usec - base_sec;
}

int main(int argc, char** argv) {
  double t0, t1;

  void *h_a, *h_b, *h_c;
  void *d_a, *d_b, *d_c;

  size_t cb = MEM_SIZE;

  h_a = malloc(cb);
  h_b = malloc(cb);
  h_c = malloc(cb);

  _cuerror(cudaFree(0));

  _cuerror(cudaMalloc(&d_a, cb));
  _cuerror(cudaMalloc(&d_b, cb));
  _cuerror(cudaMalloc(&d_c, cb));

  _cuerror(cudaMemcpy(d_a, h_a, cb, cudaMemcpyHostToDevice));
  _cuerror(cudaMemcpy(d_b, h_b, cb, cudaMemcpyHostToDevice));
  _cuerror(cudaMemcpy(d_c, h_c, cb, cudaMemcpyHostToDevice));

  const char* kernels[] = {"iaxpy", "saxpy", "daxpy", "igemm", "sgemm", "dgemm", "irand", "srand", "drand"};
  int all = argc == 1;
  if (all) argc = 1 + sizeof(kernels) / sizeof(char*);

  for (int i = 1; i < argc; i++) {
    t0 = now();
    const char* kernel = all ? kernels[i - 1] : argv[i];
    if      (strcmp(kernels[0], kernel) == 0) run_axpy<int>   ((int*)    d_c, 10,   (int*)    d_a, (int*)    d_b);
    else if (strcmp(kernels[1], kernel) == 0) run_axpy<float> ((float*)  d_c, 10.0, (float*)  d_a, (float*)  d_b);
    else if (strcmp(kernels[2], kernel) == 0) run_axpy<double>((double*) d_c, 10.0, (double*) d_a, (double*) d_b);
    else if (strcmp(kernels[3], kernel) == 0) run_gemm<int>   ((int*)    d_c, (int*)    d_a, (int*)    d_b);
    else if (strcmp(kernels[4], kernel) == 0) run_gemm<float> ((float*)  d_c, (float*)  d_a, (float*)  d_b);
    else if (strcmp(kernels[5], kernel) == 0) run_gemm<double>((double*) d_c, (double*) d_a, (double*) d_b);
    else if (strcmp(kernels[6], kernel) == 0) run_rand<int>   ((int*)    d_a);
    else if (strcmp(kernels[7], kernel) == 0) run_rand<float> ((float*)  d_a);
    else if (strcmp(kernels[8], kernel) == 0) run_rand<double>((double*) d_a);
    else printf("[%s:%d] %s\n", __FILE__, __LINE__, argv[i]);
    _cuerror(cudaGetLastError());
    _cuerror(cudaDeviceSynchronize());
    t1 = now();
    _timer(kernel, t0, t1);
  }

  _cuerror(cudaMemcpy(h_c, d_c, cb, cudaMemcpyDeviceToHost));

  _cuerror(cudaFree(d_a));
  _cuerror(cudaFree(d_b));
  _cuerror(cudaFree(d_c));

  return 0;
}


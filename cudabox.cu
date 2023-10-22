#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <curand_kernel.h>

#define __SHORT_FILE__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define _cuerror(err) do { if (err != cudaSuccess) { printf("[%s:%d:%s] err[%d][%s]\n", __SHORT_FILE__, __LINE__, __func__, err, cudaGetErrorString(err)); fflush(stdout); } } while (0)
#define _timer(name, t0, t1) do { printf("[%s:%d:%s] %-10s timer[%lf]\n", __SHORT_FILE__, __LINE__, __func__, name, t1 - t0); fflush(stdout); } while (0)

cudaError_t err;
size_t MEM_SIZE = 1 * 1024 * 1024 * 1024;

__global__ void saxpy(float* s, float a, float *x, float *y)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = a * x[tid] + y[tid];
}

void run_saxpy(float* s, float a, float* x, float* y) {
  int N = MEM_SIZE / sizeof(float);
  int dim_block = 1024;
  int dim_grid  = N / dim_block;

  saxpy<<<dim_grid, dim_block>>>(s, a, x, y);
  _cuerror(cudaGetLastError());
  _cuerror(cudaDeviceSynchronize());
}

__global__ void sgemm(float* c, float* a, float* b, int k)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  c[y * k + x] = 0.0;
  for(int i = 0; i < k; i++) {
    c[y * k + x] += a[y * k + i] * b[i * k + x];
  }
}

void run_sgemm(float* c, float* a, float* b) {
  int N = 8192;
  dim3 dim_block(16, 16);
  dim3 dim_grid(N / 16, N / 16);

  sgemm<<<dim_grid, dim_block>>>(c, a, b, N);
  _cuerror(cudaGetLastError());
  _cuerror(cudaDeviceSynchronize());
}

__global__ void random(float *a, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  curand_init(0xdeadcafe, tid, 0, &state);
  int i = curand(&state) % n;
  a[i] += 1.0f;
}

void run_random(float* a) {
  int N = MEM_SIZE / sizeof(float);
  int dim_block = 1024;
  int dim_grid  = N / dim_block;

  random<<<dim_grid, dim_block>>>(a, N);
  _cuerror(cudaGetLastError());
  _cuerror(cudaDeviceSynchronize());
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

  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;

  size_t cb = MEM_SIZE;

  h_a = (float*) malloc(cb);
  h_b = (float*) malloc(cb);
  h_c = (float*) malloc(cb);

  _cuerror(cudaFree(0));

  _cuerror(cudaMalloc(&d_a, cb));
  _cuerror(cudaMalloc(&d_b, cb));
  _cuerror(cudaMalloc(&d_c, cb));

  _cuerror(cudaMemcpy(d_a, h_a, cb, cudaMemcpyHostToDevice));
  _cuerror(cudaMemcpy(d_b, h_b, cb, cudaMemcpyHostToDevice));
  _cuerror(cudaMemcpy(d_c, h_c, cb, cudaMemcpyHostToDevice));

  for (int i = 1; i < argc; i++) {
    t0 = now();
    if (strcmp("saxpy", argv[i]) == 0) run_saxpy(d_c, 10.0f, d_a, d_b);
    else if (strcmp("sgemm", argv[i]) == 0) run_sgemm(d_c, d_a, d_b);
    else if (strcmp("random", argv[i]) == 0) run_random(d_a);
    else printf("[%s:%d] %s\n", __FILE__, __LINE__, argv[i]);
    t1 = now();
    _timer(argv[i], t0, t1);
  }

  _cuerror(cudaMemcpy(h_a, d_a, cb, cudaMemcpyDeviceToHost));

  _cuerror(cudaFree(d_a));
  _cuerror(cudaFree(d_b));
  _cuerror(cudaFree(d_c));

  return 0;
}


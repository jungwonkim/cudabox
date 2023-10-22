#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <curand_kernel.h>

#define __SHORT_FILE__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define _cuerror(err) do { if (err != cudaSuccess) { printf("[%s:%d:%s] err[%d][%s]\n", __SHORT_FILE__, __LINE__, __func__, err, cudaGetErrorString(err)); fflush(stdout); } } while (0)
#define _info(fmt, ...) do { printf(fmt "\n", __VA_ARGS__); fflush(stdout); } while (0)
#define _debug(fmt, ...) do { printf("D [%s:%d:%s] " fmt "\n", __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); } while (0)

size_t MEM_SIZE = 1 * 1024 * 1024 * 1024;

cudaError_t err;

template <typename T>
__global__ void axpy(T* s, T a, T *x, T *y) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = a * x[tid] + y[tid];
}

template <typename T>
void run_axpy(T* s, T a, T* x, T* y) {
  int N = MEM_SIZE / 8;
  int B = 1024;
  int G = N / B;
  axpy<T><<<G, B>>>(s, a, x, y);
}

template <typename T>
__global__ void gemm(T* c, T* a, T* b, int k) {
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
  dim3 B(16, 16);
  dim3 G(N / 16, N / 16);
  gemm<T><<<G, B>>>(c, a, b, N);
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
  int B = 1024;
  int G = N / B;
  rand<T><<<G, B>>>(a, N);
}

double now() {
  static double base_sec = -1;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  if (base_sec < 0) base_sec = tv.tv_sec + 1.e-6 * tv.tv_usec;
  return tv.tv_sec + 1.e-6 * tv.tv_usec - base_sec;
}

int main(int argc, char** argv) {
  void *h_a, *h_b, *h_c;
  void *d_a, *d_b, *d_c;

  h_a = malloc(MEM_SIZE);
  h_b = malloc(MEM_SIZE);
  h_c = malloc(MEM_SIZE);

  _cuerror(cudaFree(0));

  _cuerror(cudaMalloc(&d_a, MEM_SIZE));
  _cuerror(cudaMalloc(&d_b, MEM_SIZE));
  _cuerror(cudaMalloc(&d_c, MEM_SIZE));

  _cuerror(cudaMemcpy(d_a, h_a, MEM_SIZE, cudaMemcpyHostToDevice));
  _cuerror(cudaMemcpy(d_b, h_b, MEM_SIZE, cudaMemcpyHostToDevice));
  _cuerror(cudaMemcpy(d_c, h_c, MEM_SIZE, cudaMemcpyHostToDevice));

  const char* kernels[] = { "iaxpy", "saxpy", "daxpy",
                            "igemm", "sgemm", "dgemm",
                            "irand", "srand", "drand",
                          };
  int all = argc == 1;
  int nkernels = all ? sizeof(kernels) / sizeof(char*) : argc - 1;

  for (int i = 0; i < nkernels; i++) {
    const char* kernel = all ? kernels[i] : argv[i + 1];
    double t0 = now();
    if      (strcmp(kernels[0], kernel) == 0) run_axpy<int>   ((int*)    d_c, 10,   (int*)    d_a, (int*)    d_b);
    else if (strcmp(kernels[1], kernel) == 0) run_axpy<float> ((float*)  d_c, 10.0, (float*)  d_a, (float*)  d_b);
    else if (strcmp(kernels[2], kernel) == 0) run_axpy<double>((double*) d_c, 10.0, (double*) d_a, (double*) d_b);
    else if (strcmp(kernels[3], kernel) == 0) run_gemm<int>   ((int*)    d_c, (int*)    d_a, (int*)    d_b);
    else if (strcmp(kernels[4], kernel) == 0) run_gemm<float> ((float*)  d_c, (float*)  d_a, (float*)  d_b);
    else if (strcmp(kernels[5], kernel) == 0) run_gemm<double>((double*) d_c, (double*) d_a, (double*) d_b);
    else if (strcmp(kernels[6], kernel) == 0) run_rand<int>   ((int*)    d_c);
    else if (strcmp(kernels[7], kernel) == 0) run_rand<float> ((float*)  d_c);
    else if (strcmp(kernels[8], kernel) == 0) run_rand<double>((double*) d_c);
    else { _info("%-10s no kernel", kernel); continue; }
    _cuerror(cudaGetLastError());
    _cuerror(cudaDeviceSynchronize());
    _info("%-10s %lf", kernel, now() - t0);
  }

  _cuerror(cudaMemcpy(h_a, d_a, MEM_SIZE, cudaMemcpyDeviceToHost));
  _cuerror(cudaMemcpy(h_b, d_b, MEM_SIZE, cudaMemcpyDeviceToHost));
  _cuerror(cudaMemcpy(h_c, d_c, MEM_SIZE, cudaMemcpyDeviceToHost));

  _cuerror(cudaFree(d_a));
  _cuerror(cudaFree(d_b));
  _cuerror(cudaFree(d_c));

  return 0;
}


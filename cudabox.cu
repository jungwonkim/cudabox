#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <curand_kernel.h>

#define __SHORT_FILE__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define _cuerror(err) do { if (err != cudaSuccess) { printf("[%s:%d:%s] err[%d][%s]\n", __SHORT_FILE__, __LINE__, __func__, err, cudaGetErrorString(err)); fflush(stdout); } } while (0)
#define _info(fmt, ...) do { printf(fmt "\n", __VA_ARGS__); fflush(stdout); } while (0)
#define _debug(fmt, ...) do { printf("D [%s:%d:%s] " fmt "\n", __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); } while (0)
#define MEGA (1024 * 1024UL)

cudaError_t err;

size_t MEM_SIZE = 1 * 1024 * MEGA;

template <typename T>
__global__ void comp(T* a) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  T sum = 0;
  for (int i = 0; i < blockDim.x / 4; i++) {
    sum += i * (i + 10) - (i / 7);
  }
  a[x] = sum;
}

template <typename T>
void run_comp(T* a) {
  int N = MEM_SIZE / 8;
  int B = 1024;
  int G = N / B;
  comp<T><<<G, B>>>(a);
}

template <typename T>
__global__ void gevv(T* c, T *a, T *b) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  c[x] = a[x] + b[x];
}

template <typename T>
void run_gevv(T* c, T* a, T* b) {
  int N = MEM_SIZE / 8;
  int B = 1024;
  int G = N / B;
  gevv<T><<<G, B>>>(c, a, b);
}

template <typename T>
__global__ void irvv(T *c, T *a, T *b, int* r) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int i = r[x];
  c[i] = a[i] + b[i];
}

template <typename T>
void run_irvv(T* c, T* a, T* b, int* r) {
  int N = MEM_SIZE / 8;
  int B = 1024;
  int G = N / B;
  irvv<T><<<G, B>>>(c, a, b, r);
}

template <typename T>
__global__ void gemv(T* c, T* a, T* b, int k) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = 0; i < k; i++) {
    c[x] += a[x * k + i] * b[i];
  }
}

template <typename T>
void run_gemv(T* c, T* a, T* b) {
  int N = MEM_SIZE / 1024 / 256;
  int B = 1024;
  int G = N / B;
  gemv<T><<<G, B>>>(c, a, b, N);
}

template <typename T>
__global__ void gemm(T* c, T* a, T* b, int k) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  for(int i = 0; i < k; i++) {
    c[y * k + x] += a[y * k + i] * b[i * k + x];
  }
}

template <typename T>
void run_gemm(T* c, T* a, T* b) {
  int N = MEM_SIZE / 1024 / 256;
  dim3 B(32, 32);
  dim3 G(N / 32, N / 32);
  gemm<T><<<G, B>>>(c, a, b, N);
}

template <typename T>
__global__ void rand(T *a, int n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  curand_init(0, x, 0, &state);
  int i = curand(&state) % n;
  a[i] += i;
}

template <typename T>
void run_rand(T* a) {
  int N = MEM_SIZE / 8;
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

  int *h_r;
  int *d_r;

  if (argc > 1) MEM_SIZE = atol(argv[1]) * MEGA;;

  _info("%-10s [%lu]MB [%.2lf]GB", "MEM_SIZE", MEM_SIZE / MEGA, (double) MEM_SIZE / MEGA / 1024);

  h_a = malloc(MEM_SIZE);
  h_b = malloc(MEM_SIZE);
  h_c = malloc(MEM_SIZE);
  h_r = (int*) malloc(MEM_SIZE);

  srand(0);
  for (size_t i = 0; i < MEM_SIZE / sizeof(int); i++) {
    h_r[i] = rand() % (MEM_SIZE / sizeof(double));
  }

  _cuerror(cudaFree(0));

  _cuerror(cudaMalloc(&d_a, MEM_SIZE));
  _cuerror(cudaMalloc(&d_b, MEM_SIZE));
  _cuerror(cudaMalloc(&d_c, MEM_SIZE));
  _cuerror(cudaMalloc(&d_r, MEM_SIZE));

  _cuerror(cudaMemcpy(d_a, h_a, MEM_SIZE, cudaMemcpyHostToDevice));
  _cuerror(cudaMemcpy(d_b, h_b, MEM_SIZE, cudaMemcpyHostToDevice));
  _cuerror(cudaMemcpy(d_c, h_c, MEM_SIZE, cudaMemcpyHostToDevice));
  _cuerror(cudaMemcpy(d_r, h_r, MEM_SIZE, cudaMemcpyHostToDevice));

  const char* kernels[] = {
    "icomp", "scomp", "dcomp",
    "igevv", "sgevv", "dgevv",
    "iirvv", "sirvv", "dirvv",
    "igemv", "sgemv", "dgemv",
    "igemm", "sgemm", "dgemm",
    "irand", "srand", "drand",
  };
  int all = argc < 3;
  int nkernels = all ? sizeof(kernels) / sizeof(char*) : argc - 2;

  for (int i = 0; i < nkernels; i++) {
    const char* kernel = all ? kernels[i] : argv[i + 2];
    double t0 = now();
    if      (strcmp(kernels[ 0], kernel) == 0) run_comp<int>   ((int*)    d_c);
    else if (strcmp(kernels[ 1], kernel) == 0) run_comp<float> ((float*)  d_c);
    else if (strcmp(kernels[ 2], kernel) == 0) run_comp<double>((double*) d_c);
    else if (strcmp(kernels[ 3], kernel) == 0) run_gevv<int>   ((int*)    d_c, (int*)    d_a, (int*)    d_b);
    else if (strcmp(kernels[ 4], kernel) == 0) run_gevv<float> ((float*)  d_c, (float*)  d_a, (float*)  d_b);
    else if (strcmp(kernels[ 5], kernel) == 0) run_gevv<double>((double*) d_c, (double*) d_a, (double*) d_b);
    else if (strcmp(kernels[ 6], kernel) == 0) run_irvv<int>   ((int*)    d_c, (int*)    d_a, (int*)    d_b, d_r);
    else if (strcmp(kernels[ 7], kernel) == 0) run_irvv<float> ((float*)  d_c, (float*)  d_a, (float*)  d_b, d_r);
    else if (strcmp(kernels[ 8], kernel) == 0) run_irvv<double>((double*) d_c, (double*) d_a, (double*) d_b, d_r);
    else if (strcmp(kernels[ 9], kernel) == 0) run_gemv<int>   ((int*)    d_c, (int*)    d_a, (int*)    d_b);
    else if (strcmp(kernels[10], kernel) == 0) run_gemv<float> ((float*)  d_c, (float*)  d_a, (float*)  d_b);
    else if (strcmp(kernels[11], kernel) == 0) run_gemv<double>((double*) d_c, (double*) d_a, (double*) d_b);
    else if (strcmp(kernels[12], kernel) == 0) run_gemm<int>   ((int*)    d_c, (int*)    d_a, (int*)    d_b);
    else if (strcmp(kernels[13], kernel) == 0) run_gemm<float> ((float*)  d_c, (float*)  d_a, (float*)  d_b);
    else if (strcmp(kernels[14], kernel) == 0) run_gemm<double>((double*) d_c, (double*) d_a, (double*) d_b);
    else if (strcmp(kernels[15], kernel) == 0) run_rand<int>   ((int*)    d_c);
    else if (strcmp(kernels[16], kernel) == 0) run_rand<float> ((float*)  d_c);
    else if (strcmp(kernels[17], kernel) == 0) run_rand<double>((double*) d_c);
    else { _info("%-10s no kernel", kernel); continue; }
    _cuerror(cudaGetLastError());
    _cuerror(cudaDeviceSynchronize());
    _info("%-10s %lf", kernel, now() - t0);
  }

  _cuerror(cudaMemcpy(h_a, d_a, MEM_SIZE, cudaMemcpyDeviceToHost));
  _cuerror(cudaMemcpy(h_b, d_b, MEM_SIZE, cudaMemcpyDeviceToHost));
  _cuerror(cudaMemcpy(h_c, d_c, MEM_SIZE, cudaMemcpyDeviceToHost));
  _cuerror(cudaMemcpy(h_r, d_r, MEM_SIZE, cudaMemcpyDeviceToHost));

  _cuerror(cudaFree(d_a));
  _cuerror(cudaFree(d_b));
  _cuerror(cudaFree(d_c));
  _cuerror(cudaFree(d_r));

  return 0;
}


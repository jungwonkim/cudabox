#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <curand_kernel.h>

#define RED     "\033[22;31m"
#define GREEN   "\033[22;32m"
#define YELLOW  "\033[22;33m"
#define BLUE    "\033[22;34m"
#define PURPLE  "\033[22;35m"
#define CYAN    "\033[22;36m"
#define GRAY    "\033[22;37m"
#define RESET   "\x1b[m"

#define _cudaerr(cudafn) do { cudaError_t err = cudafn; if (err != cudaSuccess) { printf("[%d:%s] CUDA_ERROR[%d] %s\n", __LINE__, __func__, err, cudaGetErrorString(err)); fflush(stdout); } } while (0)
#define _check()         do { printf(PURPLE"[%d:%s] " RESET "\n", __LINE__, __func__); fflush(stdout); } while (0)
#define _info(fmt,  ...) do { printf(fmt "\n", __VA_ARGS__); fflush(stdout); } while (0)
#define _trace(fmt, ...) do { printf(BLUE  "[%d:%s] " fmt RESET "\n", __LINE__, __func__, __VA_ARGS__); fflush(stdout); } while (0)
#define _debug(fmt, ...) do { printf(GREEN "[%d:%s] " fmt RESET "\n", __LINE__, __func__, __VA_ARGS__); fflush(stdout); } while (0)
#define _error(fmt, ...) do { printf(RED   "[%d:%s] " fmt RESET "\n", __LINE__, __func__, __VA_ARGS__); fflush(stdout); } while (0)

#define RUN_KERNEL_INIT(KERNEL) int kernel_idx = 0; int kernel_found = 0; char* kernel = KERNEL;
#define RUN_KERNEL1(FUNC_NAME, ARG1) \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<int>   ((int*)    ARG1); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<float> ((float*)  ARG1); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<double>((double*) ARG1); kernel_found = 1; } 
#define RUN_KERNEL2(FUNC_NAME, ARG1, ARG2) \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<int>   ((int*)    ARG1, (int*)    ARG2); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<float> ((float*)  ARG1, (float*)  ARG2); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<double>((double*) ARG1, (double*) ARG2); kernel_found = 1; }
#define RUN_KERNEL3(FUNC_NAME, ARG1, ARG2, ARG3) \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<int>   ((int*)    ARG1, (int*)    ARG2, (int*)    ARG3); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<float> ((float*)  ARG1, (float*)  ARG2, (float*)  ARG3); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<double>((double*) ARG1, (double*) ARG2, (double*) ARG3); kernel_found = 1; }
#define RUN_KERNEL4(FUNC_NAME, ARG1, ARG2, ARG3, ARG4) \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<int>   ((int*)    ARG1, (int*)    ARG2, (int*)    ARG3, ARG4); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<float> ((float*)  ARG1, (float*)  ARG2, (float*)  ARG3, ARG4); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<double>((double*) ARG1, (double*) ARG2, (double*) ARG3, ARG4); kernel_found = 1; }
#define RUN_KERNEL_CHECK() { if (!kernel_found) { _info("%-10s no kernel", kernel); continue; } }

#define MEGA (1024 * 1024UL)

size_t MEMSIZE = 1 * 1024 * MEGA;
int BLOCKSIZE = 256;
int STRIDE = 2;
int SEED = 0;

const char* KERNELS[] = {
  "icomp", "scomp", "dcomp",
  "icudf", "scudf", "dcudf",
  "igevv", "sgevv", "dgevv",
  "istvv", "sstvv", "dstvv",
  "iirvv", "sirvv", "dirvv",
  "iisvv", "sisvv", "disvv",
  "igemv", "sgemv", "dgemv",
  "igemm", "sgemm", "dgemm",
  "irand", "srand", "drand",
};

template <typename T>
__global__ void comp(T* a) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  T sum = x;
  for (int i = 0; i < blockDim.x / 4; i++) {
    sum += i * (i + 13) / (i - 7);
  }
  a[x] = sum;
}

template <typename T>
void run_comp(T* a) {
  int N = MEMSIZE / sizeof(double);
  int B = BLOCKSIZE;
  int G = N / B;
  comp<T><<<G, B>>>(a);
}

template <typename T>
__global__ void cudf(T* a) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  T sum = 0;
  for (int i = 0; i < blockDim.x / 4; i++) {
    sum += i * (i + 10) - (i / 7);
  }
  a[x] = sum;
}

template <typename T>
void run_cudf(T* a) {
  int N = MEMSIZE / sizeof(double);
  int B = BLOCKSIZE;
  int G = N / B;
  cudf<T><<<G, B>>>(a);
}

template <typename T>
__global__ void gevv(T* c, T *a, T *b) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  c[x] = a[x] + b[x];
}

template <typename T>
void run_gevv(T* c, T* a, T* b) {
  int N = MEMSIZE / sizeof(double);
  int B = BLOCKSIZE;
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
  int N = MEMSIZE / sizeof(double);
  int B = BLOCKSIZE;
  int G = N / B;
  irvv<T><<<G, B>>>(c, a, b, r);
}

template <typename T>
__global__ void isvv(T *c, T *a, T *b, int* s) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int i = s[x];
  c[i] = a[i] + b[i];
}

template <typename T>
void run_isvv(T* c, T* a, T* b, int* s) {
  int N = MEMSIZE / sizeof(double);
  int B = BLOCKSIZE;
  int G = N / B;
  isvv<T><<<G, B>>>(c, a, b, s);
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
  int N = sqrt(MEMSIZE);
  int B = BLOCKSIZE;
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
  int N = sqrt(MEMSIZE / sizeof(double) / sizeof(double));
  dim3 B(sqrt(BLOCKSIZE), sqrt(BLOCKSIZE));
  dim3 G(N / B.x, N / B.y);
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
  int N = MEMSIZE / sizeof(double) / 8;
  int B = BLOCKSIZE;
  int G = N / B;
  rand<T><<<G, B>>>(a, N);
}

template <typename T>
__global__ void stvv(T *c, T *a, T *b, int stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int i = x * stride;
  c[i] = a[i] + b[i];
}

template <typename T>
void run_stvv(T* c, T* a, T* b) {
  int N = MEMSIZE / sizeof(double) / STRIDE;
  int B = BLOCKSIZE;
  int G = N / B;
  stvv<T><<<G, B>>>(c, a, b, STRIDE);
}

double now() {
  static double base_sec = -1;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  if (base_sec < 0) base_sec = tv.tv_sec + 1.e-6 * tv.tv_usec;
  return tv.tv_sec + 1.e-6 * tv.tv_usec - base_sec;
}

int help() {
  printf("Usage: cudabox ");
  for (int i = 0; i < sizeof(KERNELS) / sizeof(char*); i++) printf("%s ", KERNELS[i]);
  printf("\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 2 && (strcmp("help", argv[1]) == 0 || strcmp("-h", argv[1]) == 0)) return help();
  if (getenv("CUDABOX_MEMSIZE"))    MEMSIZE   = atoi(getenv("CUDABOX_MEMSIZE")) * MEGA;
  if (getenv("CUDABOX_BLOCKSIZE"))  BLOCKSIZE = atoi(getenv("CUDABOX_BLOCKSIZE"));
  if (getenv("CUDABOX_STRIDE"))     STRIDE    = atoi(getenv("CUDABOX_STRIDE"));
  if (getenv("CUDABOX_SEED"))       SEED      = atoi(getenv("CUDABOX_SEED"));

  _info("CUDABOX_$ MEMSIZE[%zu]MB BLOCKSIZE[%d] STRIDE[%d] SEED[%d]", MEMSIZE / MEGA, BLOCKSIZE, STRIDE, SEED);

  void *h_a, *h_b, *h_c, *h_a16;
  void *d_a, *d_b, *d_c, *d_a16;

  int *h_r, *h_s;
  int *d_r, *d_s;

  h_a = malloc(MEMSIZE);
  h_b = malloc(MEMSIZE);
  h_c = malloc(MEMSIZE);
  h_r = (int*) malloc(MEMSIZE);
  h_s = (int*) malloc(MEMSIZE);
  h_a16 = malloc(16 * MEMSIZE);

  srand(SEED);
  for (size_t i = 0; i < MEMSIZE / sizeof(int); i++) {
    h_r[i] = rand() % (MEMSIZE / sizeof(double));
    h_s[i] = i;
  }

  _cudaerr(cudaFree(0));

  _cudaerr(cudaMalloc(&d_a, MEMSIZE));
  _cudaerr(cudaMalloc(&d_b, MEMSIZE));
  _cudaerr(cudaMalloc(&d_c, MEMSIZE));
  _cudaerr(cudaMalloc(&d_c, MEMSIZE));
  _cudaerr(cudaMalloc(&d_r, MEMSIZE));
  _cudaerr(cudaMalloc(&d_s, MEMSIZE));
  _cudaerr(cudaMalloc(&d_a16, 16 * MEMSIZE));

  _cudaerr(cudaMemcpy(d_a, h_a, MEMSIZE, cudaMemcpyHostToDevice));
  _cudaerr(cudaMemcpy(d_b, h_b, MEMSIZE, cudaMemcpyHostToDevice));
  _cudaerr(cudaMemcpy(d_c, h_c, MEMSIZE, cudaMemcpyHostToDevice));
  _cudaerr(cudaMemcpy(d_r, h_r, MEMSIZE, cudaMemcpyHostToDevice));
  _cudaerr(cudaMemcpy(d_s, h_s, MEMSIZE, cudaMemcpyHostToDevice));
  _cudaerr(cudaMemcpy(d_a16, h_a16, 16 * MEMSIZE, cudaMemcpyHostToDevice));

  int all = argc == 1;
  int nkernels = all ? sizeof(KERNELS) / sizeof(char*) : argc - 1;
  char** kernels = all ? (char**) KERNELS : argv + 1;

  for (int i = 0; i < nkernels; i++) {
    double t0 = now();
    RUN_KERNEL_INIT(kernels[i]);
    RUN_KERNEL1(run_comp, d_c);
    RUN_KERNEL1(run_cudf, d_c);
    RUN_KERNEL3(run_gevv, d_c, d_a, d_b);
    RUN_KERNEL3(run_stvv, d_c, d_a, d_b);
    RUN_KERNEL4(run_irvv, d_c, d_a, d_b, d_r);
    RUN_KERNEL4(run_isvv, d_c, d_a, d_b, d_s);
    RUN_KERNEL3(run_gemv, d_c, d_a16, d_b);
    RUN_KERNEL3(run_gemm, d_c, d_a, d_b);
    RUN_KERNEL1(run_rand, d_c);
    RUN_KERNEL_CHECK();
    _cudaerr(cudaGetLastError());
    _cudaerr(cudaDeviceSynchronize());
    _info("%-10s %lf", kernel, now() - t0);
  }

#ifdef CUDABOX_D2H
  _cudaerr(cudaMemcpy(h_a, d_a, MEMSIZE, cudaMemcpyDeviceToHost));
  _cudaerr(cudaMemcpy(h_b, d_b, MEMSIZE, cudaMemcpyDeviceToHost));
  _cudaerr(cudaMemcpy(h_c, d_c, MEMSIZE, cudaMemcpyDeviceToHost));
  _cudaerr(cudaMemcpy(h_a16, d_a16, 16 * MEMSIZE, cudaMemcpyDeviceToHost));
#endif

  _cudaerr(cudaFree(d_a));
  _cudaerr(cudaFree(d_b));
  _cudaerr(cudaFree(d_c));
  _cudaerr(cudaFree(d_r));
  _cudaerr(cudaFree(d_s));
  _cudaerr(cudaFree(d_a16));

  return 0;
}


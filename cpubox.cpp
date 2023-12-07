#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

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

#define ARG_PREFIX h_
#define ARG_CONCAT(PREFIX, ARG)   PREFIX##ARG
#define ARG_ECONCAT(PREFIX, ARG)  ARG_CONCAT(PREFIX, ARG)
#define ARG(ARGN)                 ARG_ECONCAT(ARG_PREFIX, ARGN)

#define RUN_KERNEL_INIT(KERNEL) int kernel_idx = 0; int kernel_found = 0; char* kernel = KERNEL;
#define RUN_KERNEL1(FUNC_NAME, ARG1) \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<int>   ((int*)    ARG(ARG1)); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<float> ((float*)  ARG(ARG1)); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<double>((double*) ARG(ARG1)); kernel_found = 1; } 
#define RUN_KERNEL2(FUNC_NAME, ARG1, ARG2) \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<int>   ((int*)    ARG(ARG1), (int*)    ARG(ARG2)); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<float> ((float*)  ARG(ARG1), (float*)  ARG(ARG2)); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<double>((double*) ARG(ARG1), (double*) ARG(ARG2)); kernel_found = 1; }
#define RUN_KERNEL3(FUNC_NAME, ARG1, ARG2, ARG3) \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<int>   ((int*)    ARG(ARG1), (int*)    ARG(ARG2), (int*)    ARG(ARG3)); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<float> ((float*)  ARG(ARG1), (float*)  ARG(ARG2), (float*)  ARG(ARG3)); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<double>((double*) ARG(ARG1), (double*) ARG(ARG2), (double*) ARG(ARG3)); kernel_found = 1; }
#define RUN_KERNEL4(FUNC_NAME, ARG1, ARG2, ARG3, ARG4) \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<int>   ((int*)    ARG(ARG1), (int*)    ARG(ARG2), (int*)    ARG(ARG3), ARG(ARG4)); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<float> ((float*)  ARG(ARG1), (float*)  ARG(ARG2), (float*)  ARG(ARG3), ARG(ARG4)); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<double>((double*) ARG(ARG1), (double*) ARG(ARG2), (double*) ARG(ARG3), ARG(ARG4)); kernel_found = 1; }
#define RUN_KERNEL5(FUNC_NAME, ARG1, ARG2, ARG3, ARG4, ARG5) \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<int>   ((int*)    ARG(ARG1), (int*)    ARG(ARG2), (int*)    ARG(ARG3), ARG(ARG4), ARG(ARG5)); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<float> ((float*)  ARG(ARG1), (float*)  ARG(ARG2), (float*)  ARG(ARG3), ARG(ARG4), ARG(ARG5)); kernel_found = 1; } \
    if (!kernel_found && strcmp(KERNELS[kernel_idx++], kernel) == 0) { FUNC_NAME<double>((double*) ARG(ARG1), (double*) ARG(ARG2), (double*) ARG(ARG3), ARG(ARG4), ARG(ARG5)); kernel_found = 1; }
#define RUN_KERNEL_CHECK() { if (!kernel_found) { _info("%-10s no kernel", kernel); continue; } }

#define MEGA (1024 * 1024UL)

typedef struct _dim3 { unsigned int x, y, z; } dim3;

size_t  MEMSIZE       = 16 * MEGA;
size_t  MEMSIZESQRT;
int     BLOCKSIZE     = 256;
int     STRIDE        = 2;
int     SEED          = 0;
float   SPARSITY      = 0.1;
int     MAXTYPESIZE   = 8;

const char* KERNELS[] = {
  "icomp", "scomp", "dcomp",
  "icudf", "scudf", "dcudf",
  "igevv", "sgevv", "dgevv",
  "istvv", "sstvv", "dstvv",
  "iirvv", "sirvv", "dirvv",
  "iisvv", "sisvv", "disvv",
  "igemv", "sgemv", "dgemv",
  "ispmv", "sspmv", "dspmv",
  "igemm", "sgemm", "dgemm",
  "irand", "srand", "drand",
};

#define CPUBOX_KERNEL int blockDim_x = B; for (int blockIdx_x = 0; blockIdx_x < G; blockIdx_x++) for (int threadIdx_x = 0; threadIdx_x < B; threadIdx_x++)
#define CPUBOX_KERNEL_2D int blockDim_x = B.x; int blockDim_y = B.y; for (int blockIdx_y = 0; blockIdx_y < G.y; blockIdx_y++) for (int blockIdx_x = 0; blockIdx_x < G.x; blockIdx_x++) for (int threadIdx_y = 0; threadIdx_y < B.y; threadIdx_y++) for (int threadIdx_x = 0; threadIdx_x < B.y; threadIdx_x++)

/*
template <typename T>
__global__ void comp(T* a) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  T sum = x;
  for (int i = 0; i < blockDim.x / 4; i++) {
    sum += i * (i + 13) / (i + 7);
  }
  a[x] = sum;
}
*/

template <typename T>
void run_comp(T* a) {
  int N = MEMSIZE / MAXTYPESIZE;
  int B = BLOCKSIZE;
  int G = N / B;
  //comp<T><<<G, B>>>(a);
  CPUBOX_KERNEL {
  int x = blockIdx_x * blockDim_x + threadIdx_x;
  T sum = x;
  for (int i = 0; i < blockDim_x / 4; i++) {
    sum += i * (i + 13) / (i + 7);
  }
  a[x] = sum;
  }
}

/*
template <typename T>
__global__ void cudf(T* a) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  T sum = 0;
  for (int i = 0; i < blockDim.x / 4; i++) {
    sum += i * (i + 10) - (i / 7);
  }
  a[x] = sum;
}
*/

template <typename T>
void run_cudf(T* a) {
  int N = MEMSIZE / MAXTYPESIZE;
  int B = BLOCKSIZE;
  int G = N / B;
  //cudf<T><<<G, B>>>(a);
  CPUBOX_KERNEL {
  int x = blockIdx_x * blockDim_x + threadIdx_x;
  T sum = 0;
  for (int i = 0; i < blockDim_x / 4; i++) {
    sum += i * (i + 10) - (i / 7);
  }
  a[x] = sum;
  }
}

/*
template <typename T>
__global__ void gevv(T* c, T *a, T *b) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  c[x] = a[x] + b[x];
}
*/

template <typename T>
void run_gevv(T* c, T* a, T* b) {
  int N = MEMSIZE / MAXTYPESIZE;
  int B = BLOCKSIZE;
  int G = N / B;
  //gevv<T><<<G, B>>>(c, a, b);
  CPUBOX_KERNEL {
  int x = blockIdx_x * blockDim_x + threadIdx_x;
  c[x] = a[x] + b[x];
  }
}

/*
template <typename T>
__global__ void stvv(T *c, T *a, T *b, int STRIDE) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int i = x * STRIDE;
  c[i] = a[i] + b[i];
}
*/

template <typename T>
void run_stvv(T* c, T* a, T* b) {
  int N = MEMSIZE / MAXTYPESIZE / STRIDE;
  int B = BLOCKSIZE;
  int G = N / B;
  int stride = STRIDE;
  //stvv<T><<<G, B>>>(c, a, b, STRIDE);
  CPUBOX_KERNEL {
  int x = blockIdx_x * blockDim_x + threadIdx_x;
  int i = x * stride;
  c[i] = a[i] + b[i];
  }
}

/*
template <typename T>
__global__ void irvv(T *c, T *a, T *b, int* r) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int i = r[x];
  c[i] = a[i] + b[i];
}
*/

template <typename T>
void run_irvv(T* c, T* a, T* b, int* r) {
  int N = MEMSIZE / MAXTYPESIZE;
  int B = BLOCKSIZE;
  int G = N / B;
  //irvv<T><<<G, B>>>(c, a, b, r);
  CPUBOX_KERNEL {
  int x = blockIdx_x * blockDim_x + threadIdx_x;
  int i = r[x];
  c[i] = a[i] + b[i];
  }
}

/*
template <typename T>
__global__ void isvv(T *c, T *a, T *b, int* s) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int i = s[x];
  c[i] = a[i] + b[i];
}
*/

template <typename T>
void run_isvv(T* c, T* a, T* b, int* s) {
  int N = MEMSIZE / MAXTYPESIZE;
  int B = BLOCKSIZE;
  int G = N / B;
  //isvv<T><<<G, B>>>(c, a, b, s);
  CPUBOX_KERNEL {
  int x = blockIdx_x * blockDim_x + threadIdx_x;
  int i = s[x];
  c[i] = a[i] + b[i];
  }
}

/*
template <typename T>
__global__ void gemv(T* c, T* a, T* b, int k) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = 0; i < k; i++) {
    c[x] += a[x * k + i] * b[i];
  }
}
*/

template <typename T>
void run_gemv(T* c, T* a, T* b) {
  int N = MEMSIZESQRT;
  int B = BLOCKSIZE;
  int G = N / B;
  int k = N;
  //gemv<T><<<G, B>>>(c, a, b, N);
  CPUBOX_KERNEL {
  int x = blockIdx_x * blockDim_x + threadIdx_x;
  for(int i = 0; i < k; i++) {
    c[x] += a[x * k + i] * b[i];
  }
  }
}

/*
template <typename T>
__global__ void spmv(T* y, T* a, T* x, int* ia, int* ja) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int row0 = ia[row];
  int row1 = ia[row + 1];
  for (int j = row0; j < row1; j++) {
    y[row] += a[j] * x[ja[j]];
  }
}
*/

template <typename T>
void run_spmv(T* y, T* a, T* x,  int* ia, int* ja) {
  int N = MEMSIZESQRT;
  int B = BLOCKSIZE;
  int G = N / B;
  //spmv<T><<<G, B>>>(y, a, x, ia, ja);
  CPUBOX_KERNEL {
  int row = blockIdx_x * blockDim_x + threadIdx_x;
  int row0 = ia[row];
  int row1 = ia[row + 1];
  for (int j = row0; j < row1; j++) {
    y[row] += a[j] * x[ja[j]];
  }
  }
}

/*
template <typename T>
__global__ void gemm(T* c, T* a, T* b, int k) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  for(int i = 0; i < k; i++) {
    c[y * k + x] += a[y * k + i] * b[i * k + x];
  }
}
*/

template <typename T>
void run_gemm(T* c, T* a, T* b) {
  int N = static_cast<int>(sqrt(MEMSIZE / MAXTYPESIZE / MAXTYPESIZE));
  dim3 B{static_cast<unsigned int>(sqrt(BLOCKSIZE)), static_cast<unsigned int>(sqrt(BLOCKSIZE)), 1};
  dim3 G{N / B.x, N / B.y, 1};
  int k = N;
  //gemm<T><<<G, B>>>(c, a, b, N);
  CPUBOX_KERNEL_2D {
  int x = blockIdx_x * blockDim_x + threadIdx_x;
  int y = blockIdx_y * blockDim_y + threadIdx_y;
  for(int i = 0; i < k; i++) {
    c[y * k + x] += a[y * k + i] * b[i * k + x];
  }
  }
}

/*
template <typename T>
__global__ void rand(T *a, int n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  curand_init(0, x, 0, &state);
  int i = curand(&state) % n;
  a[i] += i;
}
*/

template <typename T>
void run_rand(T* a) {
  int N = MEMSIZE / MAXTYPESIZE / 8;
  int B = BLOCKSIZE;
  int G = N / B;
  int n = N;
  //rand<T><<<G, B>>>(a, N);
  CPUBOX_KERNEL {
  int x = blockIdx_x * blockDim_x + threadIdx_x;
  int i = rand() % n;
  a[i] += i;
  }
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

void init_vv(int* h_r, int* h_s) {
  for (size_t i = 0; i < MEMSIZE / sizeof(int); i++) {
    h_r[i] = rand() % (MEMSIZE / MAXTYPESIZE);
    h_s[i] = i;
  }
}

void init_spmv(int* h_ia, int* h_ja, size_t size_ja) {
  int* h_ia_temp = (int*) malloc(MEMSIZESQRT * sizeof(int));
  size_t h_ia_temp_sum = 0ULL;
  for (size_t i = 0; i < MEMSIZESQRT; i++) {
    h_ia_temp[i] = rand() % size_ja;
    h_ia_temp_sum += h_ia_temp[i];
  }
  double h_ia_scaling_factor = (double) size_ja / h_ia_temp_sum;
  h_ia_temp_sum = 0ULL;
  for (size_t i = 0; i < MEMSIZESQRT - 1; i++) {
    h_ia_temp[i] = (int) (h_ia_temp[i] * h_ia_scaling_factor);
    h_ia_temp_sum += h_ia_temp[i];
  }
  h_ia_temp[MEMSIZESQRT - 1] = size_ja - h_ia_temp_sum;
  h_ia[0] = 0;
  for (size_t i = 0; i < MEMSIZESQRT; i++) {
    h_ia[i + 1] = h_ia[i] + h_ia_temp[i];
  }
  for (size_t i = 0; i < size_ja; i++) {
    h_ja[i] = rand() % MEMSIZESQRT;
  }
  free(h_ia_temp);
}

int main(int argc, char** argv) {
  if (argc == 2 && (strcmp("help", argv[1]) == 0 || strcmp("-h", argv[1]) == 0)) return help();
  if (getenv("CUDABOX_MEMSIZE"))    MEMSIZE   = atol(getenv("CUDABOX_MEMSIZE")) * MEGA;
  if (getenv("CUDABOX_BLOCKSIZE"))  BLOCKSIZE = atoi(getenv("CUDABOX_BLOCKSIZE"));
  if (getenv("CUDABOX_STRIDE"))     STRIDE    = atoi(getenv("CUDABOX_STRIDE"));
  if (getenv("CUDABOX_SEED"))       SEED      = atoi(getenv("CUDABOX_SEED"));
  if (getenv("CUDABOX_SPARSITY"))   SPARSITY  = atof(getenv("CUDABOX_SPARSITY"));

  MEMSIZESQRT = static_cast<size_t>(sqrt(MEMSIZE));

  _info("CUDABOX_$ MEMSIZE[%zu]MB BLOCKSIZE[%d] STRIDE[%d] SEED[%d] SPARSITY[%f]", MEMSIZE / MEGA, BLOCKSIZE, STRIDE, SEED, SPARSITY);

  void *h_a, *h_b, *h_c;

  /* for vv */
  int *h_r, *h_s;

  /* for mv */
  void *h_a8;

  /* for spmv */
  int *h_ia, *h_ja;
  size_t size_ja = MEMSIZE * SPARSITY;

  int all = argc == 1;
  int nkernels = all ? sizeof(KERNELS) / sizeof(char*) : argc - 1;
  char** kernels = all ? (char**) KERNELS : argv + 1;

  int has_vv = 0;
  int has_mv = 0;
  int has_spmv = 0;
  for (int i = 0; i < nkernels; i++) {
    if (strstr(kernels[i], "vv"))   has_vv   = 1;
    if (strstr(kernels[i], "mv"))   has_mv   = 1;
    if (strstr(kernels[i], "spmv")) has_spmv = 1;
  }

  srand(SEED);

  h_a = malloc(MEMSIZE);
  h_b = malloc(MEMSIZE);
  h_c = malloc(MEMSIZE);
  h_a8 = malloc(8 * MEMSIZE);

  if (has_vv) {
    h_r = (int*) malloc(MEMSIZE);
    h_s = (int*) malloc(MEMSIZE);
    init_vv(h_r, h_s);
  }

  if (has_mv) {
    h_a8 = malloc(8 * MEMSIZE);
  }

  if (has_spmv) {
    h_ia = (int*) malloc((MEMSIZESQRT + 1) * sizeof(int));
    h_ja = (int*) malloc(size_ja * sizeof(int));
    init_spmv(h_ia, h_ja, size_ja);
  }

  for (int i = 0; i < nkernels; i++) {
    double t0 = now();
    RUN_KERNEL_INIT(kernels[i]);
    RUN_KERNEL1(run_comp, c);
    RUN_KERNEL1(run_cudf, c);
    RUN_KERNEL3(run_gevv, c, a, b);
    RUN_KERNEL3(run_stvv, c, a, b);
    RUN_KERNEL4(run_irvv, c, a, b, r);
    RUN_KERNEL4(run_isvv, c, a, b, s);
    RUN_KERNEL3(run_gemv, c, a8, b);
    RUN_KERNEL5(run_spmv, c, a8, b, ia, ja);
    RUN_KERNEL3(run_gemm, c, a, b);
    RUN_KERNEL1(run_rand, c);
    RUN_KERNEL_CHECK();
    _info("%-10s %lf", kernel, now() - t0);
  }

  return 0;
}


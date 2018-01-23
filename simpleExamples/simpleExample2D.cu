#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

#define COALESCED

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) { exit(code); }
	}
}

void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/*******************/
/* addCPU FUNCTION */
/*******************/
void addCPU2D(int *h_a, int *h_b, int *h_c, int M, int N) {

	for (int k = 0; k < M * N; k++) h_c[k] = h_a[k] + h_b[k];

}

/*********************/
/* addGPU2D FUNCTION */
/*********************/
#ifdef COALESCED
__global__ void addGPU2D(int *d_a, int *d_b, int *d_c, int M, int N) {

	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= N) || (tidy >= M)) return;

	d_c[tidy * N + tidx] = d_a[tidy * N + tidx] + d_b[tidy * N + tidx];

}
#else
__global__ void addGPU2D(int *d_a, int *d_b, int *d_c, int M, int N) {

	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= M) || (tidy >= N)) return;

	d_c[tidx * N + tidy] = d_a[tidx * N + tidy] + d_b[tidx * N + tidy];

}
#endif

/********/
/* MAIN */
/********/
int main() {

	const int M = 128;		// --- Number of rows
	const int N = 256;		// --- Number of columns

	// --- Allocating host memory for data and results
	int *h_a = (int *)malloc(M * N * sizeof(int));
	int *h_b = (int *)malloc(M * N * sizeof(int));
	int *h_c = (int *)malloc(M * N * sizeof(int));
	int *h_c_device = (int *)malloc(M * N * sizeof(int));

	// --- Allocating device memory for data and results
	int *d_a, *d_b, *d_c;
	gpuErrchk(cudaMalloc(&d_a, M * N * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_b, M * N * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_c, M * N * sizeof(int)));

	// --- Filling the input vectors on host memory
	for (int k = 0; k < M * N; k++) {
		h_a[k] = k;
		h_b[k] = 2 * k;
	}

	// --- Moving data from host to device
	gpuErrchk(cudaMemcpy(d_a, h_a, M * N * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, M * N * sizeof(int), cudaMemcpyHostToDevice));

	addCPU2D(h_a, h_b, h_c, M, N);

#ifdef COALESCED
	dim3 gridSize(iDivUp(N, BLOCKSIZEX), iDivUp(M, BLOCKSIZEY));
#else
	dim3 gridSize(iDivUp(M, BLOCKSIZEX), iDivUp(N, BLOCKSIZEY));
#endif
	dim3 blockSize(BLOCKSIZEX, BLOCKSIZEY, 1);
	addGPU2D << <gridSize, blockSize >> >(d_a, d_b, d_c, M, N);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(h_c_device, d_c, M * N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int k = 0; k < M * N; k++)
		if (h_c_device[k] != h_c[k]) {
			printf("Host and device results do not match for k = %d: h_c[%d] = %d; h_c_device[%d] = %d\n", k, k, h_c[k], k, h_c_device[k]);
		}

	printf("No errors found.\n");

	return 0;
}
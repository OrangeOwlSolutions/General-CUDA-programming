#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 128

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

/**********************************/
/* adjacentDifferencesCPU FUNCTION */
/**********************************/
void adjacentDifferencesCPU(int *h_a, int *h_b, int N)
{
	for (int k = 1; k < N; k++) {
		int a_i = h_a[k];
		int a_i_minus_one = h_a[k - 1];

		h_b[k] = a_i - a_i_minus_one;
	}
}

/**********************************/
/* adjacentDifferencesGPU FUNCTION */
/**********************************/
__global__ void adjacentDifferencesGPU(int *d_a, int *d_b, int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if ((tid >= N) || (tid == 0)) return;

	int a_i = d_a[tid];
	int a_i_minus_one = d_a[tid - 1];

	d_b[tid] = a_i - a_i_minus_one;
}

/****************************************/
/* adjacentDifferencesSharedGPU FUNCTION */
/****************************************/
__global__ void adjacentDifferencesSharedGPU(int *d_a, int *d_b, int N)
{
	// --- Shorthand for threadIdx.x
	int tidx	= threadIdx.x;
	int tid		= tidx + blockDim.x * blockIdx.x;

	// --- Allocate a __shared__ array, one element per thread
	__shared__ int s_data[BLOCKSIZE];
	
	// --- Each thread reads one element to s_data
	s_data[tidx] = d_a[tid];

	// --- Avoid race condition: ensure all loads complete before continuing
	__syncthreads();
	
	if (tidx > 0)
		d_b[tid] = s_data[tidx] - s_data[tidx - 1];
	else if (tid > 0)
	{
		// --- Handle thread block boundary
		d_b[tid] = s_data[tidx] - d_a[tid - 1];
	}

}

__global__ void adjacentDifferencesSharedWithHaloGPU(int *d_a, int *d_b, int N)
{
	// --- Allocate a __shared__ array, one element per thread
	__shared__ int s_data[BLOCKSIZE + 1];

	// --- Shorthand for threadIdx.x
	int gindexx = threadIdx.x + blockDim.x * blockIdx.x;

	int lindexx = threadIdx.x + 1;

	// --- Each thread reads one element to s_data
	s_data[lindexx] = d_a[gindexx];

	if (threadIdx.x == 0) {
		s_data[0] = (((gindexx - 1) >= 0) && (gindexx <= N)) ? d_a[gindexx - 1] : 0;
	}

	// --- Avoid race condition: ensure all loads complete before continuing
	__syncthreads();

	if (gindexx > 0) d_b[gindexx] = s_data[lindexx] - s_data[lindexx - 1];

}

/************************************************/
/* adjacentDifferencesExternalSharedGPU FUNCTION */
/************************************************/
__global__ void adjacentDifferencesExternalSharedGPU(int *d_a, int *d_b, int N)
{
	// --- Shorthand for threadIdx.x
	int tidx = threadIdx.x;
	int tid = tidx + blockDim.x * blockIdx.x;

	// --- Allocate a __shared__ array, one element per thread
	extern __shared__ int s_data[];

	// --- Each thread reads one element to s_data
	s_data[tidx] = d_a[tid];

	// --- Avoid race condition: ensure all loads complete before continuing
	__syncthreads();

	if (tidx > 0)
		d_b[tid] = s_data[tidx] - s_data[tidx - 1];
	else if (tid > 0)
	{
		// --- Handle thread block boundary
		d_b[tid] = s_data[tidx] - d_a[tid - 1];
	}

}

/********/
/* MAIN */
/********/
int main() {

	const int N = 256;

	// --- Allocating host memory for data and results
	int *h_a = (int *)malloc(N * sizeof(int));
	int *h_b = (int *)malloc(N * sizeof(int));
	int *h_b_device = (int *)malloc(N * sizeof(int));

	// --- Allocating device memory for data and results
	int *d_a, *d_b;
	gpuErrchk(cudaMalloc(&d_a, N * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_b, N * sizeof(int)));
	
	// --- Filling the input vectors on host memory
	for (int k = 0; k < N; k++) {
		h_a[k] = k;
	}

	// --- Moving data from host to device
	gpuErrchk(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));

	adjacentDifferencesCPU(h_a, h_b, N);

	//adjacentDifferencesGPU << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_a, d_b, N);
	//adjacentDifferencesSharedGPU << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_a, d_b, N);
	adjacentDifferencesSharedWithHaloGPU << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_a, d_b, N);
	//adjacentDifferencesExternalSharedGPU << <iDivUp(N, BLOCKSIZE), BLOCKSIZE, BLOCKSIZE * sizeof(int) >> >(d_a, d_b, N);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(h_b_device, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int k = 1; k < N; k++)
		if (h_b_device[k] != h_b[k]) {
			printf("Host and device results do not match for k = %d: h_c[%d] = %d; h_c_device[%d] = %d\n", k, k, h_b[k], k, h_b_device[k]);
			return 0;
		}

	printf("No errors found.\n");

	return 0;
}
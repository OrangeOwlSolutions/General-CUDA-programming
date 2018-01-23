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

/*******************/
/* addCPU FUNCTION */
/*******************/
void addCPU(int *h_a, int *h_b, int *h_c, int N) {

	int h_a_tmp, h_b_tmp;

	for (int k = 0; k < N; k++) {

		if ((k % 2) == 0) {
			h_a_tmp = h_a[k + 1];
			h_b_tmp = h_b[k + 1];
		}
		else {
			h_a_tmp = h_a[k - 1];
			h_b_tmp = h_b[k - 1];
		}

		h_c[k] = h_a_tmp + h_b_tmp;
	}

}

/***************************************************/
/* addGPU FUNCTION WITH NON-CONSECUTIVE LOAD/STORE */
/***************************************************/
__global__ void addGPUNonConsecutive(int *d_a, int *d_b, int *d_c, int N) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N) return;

	int d_a_tmp, d_b_tmp;

	if ((tid % 2) == 0) {
		d_a_tmp = d_a[tid + 1];
		d_b_tmp = d_b[tid + 1];
	}
	else {
		d_a_tmp = d_a[tid - 1];
		d_b_tmp = d_b[tid - 1];
	}

	d_c[tid] = d_a_tmp + d_b_tmp;

}

/********/
/* MAIN */
/********/
int main() {

	const int N = 256;

	// --- Allocating host memory for data and results
	int *h_a = (int *)malloc(N * sizeof(int));
	int *h_b = (int *)malloc(N * sizeof(int));
	int *h_c = (int *)malloc(N * sizeof(int));
	int *h_c_device = (int *)malloc(N * sizeof(int));

	// --- Allocating device memory for data and results
	int *d_a, *d_b, *d_c;
	gpuErrchk(cudaMalloc(&d_a, N * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_b, N * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_c, N * sizeof(int)));

	// --- Filling the input vectors on host memory
	for (int k = 0; k < N; k++) {
		h_a[k] = k;
		h_b[k] = 2 * k;
	}

	// --- Moving data from host to device
	gpuErrchk(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice));

	addCPU(h_a, h_b, h_c, N);

	//addGPU << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_a, d_b, d_c, N);
	addGPUNonConsecutive << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_a, d_b, d_c, N);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(h_c_device, d_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for (int k = 0; k < N; k++) 
		if (h_c_device[k] != h_c[k]) {
			printf("Host and device results do not match for k = %d: h_c[%d] = %d; h_c_device[%d] = %d\n", k, k, h_c[k], k, h_c_device[k]);
		}

	printf("No errors found.\n");

	return 0;
}
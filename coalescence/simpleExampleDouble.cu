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
void addCPU(double *h_a, double *h_b, double *h_c, int N) {

	for (int k = 0; k < N; k++) h_c[k] = h_a[k] + h_b[k];

}

/*******************/
/* addGPU FUNCTION */
/*******************/
__global__ void addGPU(double *d_a, double *d_b, double *d_c, int N) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N) return;

	d_c[tid] = d_a[tid] + d_b[tid];

}

/********/
/* MAIN */
/********/
int main() {

	const int N = 256;

	// --- Allocating host memory for data and results
	double *h_a = (double *)malloc(N * sizeof(double));
	double *h_b = (double *)malloc(N * sizeof(double));
	double *h_c = (double *)malloc(N * sizeof(double));
	double *h_c_device = (double *)malloc(N * sizeof(double));

	// --- Allocating device memory for data and results
	double *d_a, *d_b, *d_c;
	gpuErrchk(cudaMalloc(&d_a, N * sizeof(double)));
	gpuErrchk(cudaMalloc(&d_b, N * sizeof(double)));
	gpuErrchk(cudaMalloc(&d_c, N * sizeof(double)));

	// --- Filling the input vectors on host memory
	for (int k = 0; k < N; k++) {
		h_a[k] = k;
		h_b[k] = 2 * k;
	}

	// --- Moving data from host to device
	gpuErrchk(cudaMemcpy(d_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));

	addCPU(h_a, h_b, h_c, N);

	addGPU << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(d_a, d_b, d_c, N);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(h_c_device, d_c, N * sizeof(double), cudaMemcpyDeviceToHost));

	for (int k = 0; k < N; k++)
		if (h_c_device[k] != h_c[k]) {
		printf("Host and device results do not match for k = %d: h_c[%d] = %f; h_c_device[%d] = %f\n", k, k, h_c[k], k, h_c_device[k]);
		}

	printf("No errors found.\n");

	return 0;
}

#include <thrust/device_vector.h>

#define RADIUS        3
#define BLOCKSIZE    32

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/**********/
/* KERNEL */
/**********/
__global__ void moving_average(int *in, int *out, int N) {

	__shared__ int temp[BLOCKSIZE + 2 * RADIUS];

	int gindexx = threadIdx.x + blockIdx.x * blockDim.x;

	int lindexx = threadIdx.x + RADIUS;

	// --- Read input elements into shared memory
	temp[lindexx] = (gindexx < N) ? in[gindexx] : 0;
	if (threadIdx.x < RADIUS) {
		temp[threadIdx.x] = (((gindexx - RADIUS) >= 0) && (gindexx <= N)) ? in[gindexx - RADIUS] : 0;
		temp[threadIdx.x + (RADIUS + BLOCKSIZE)] = ((gindexx + BLOCKSIZE) < N) ? in[gindexx + BLOCKSIZE] : 0;
	}

	__syncthreads();

	// --- Apply the stencil
	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; offset++) {
		result += temp[lindexx + offset];
	}

	// --- Store the result
	out[gindexx] = result;
}

/********/
/* MAIN */
/********/
int main() {

	const int N = 55 + 2 * RADIUS;

	const int constant = 4;

	thrust::device_vector<int> d_in(N, constant);
	thrust::device_vector<int> d_out(N);

	moving_average << <iDivUp(N, BLOCKSIZE), BLOCKSIZE >> >(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::host_vector<int> h_out = d_out;

	for (int i = 0; i<N; i++)
		printf("Element i = %i; h_out = %i\n", i, h_out[i]);

	return 0;

}
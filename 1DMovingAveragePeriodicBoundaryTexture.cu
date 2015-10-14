#include <stdio.h>

#include "TimingGPU.cuh"
#include "Utilities.cuh"

texture<float, 1, cudaReadModeElementType> signal_texture;

#define BLOCKSIZE 32

/*************************************************/
/* KERNEL FUNCTION FOR MEDIAN FILTER CALCULATION */
/*************************************************/
__global__ void median_filter_periodic_boundary(float * __restrict__ d_vec, const unsigned int N){

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N) {

		float signal_center = tex1D(signal_texture, tid - 0);
		float signal_before = tex1D(signal_texture, tid - 1);
		float signal_after  = tex1D(signal_texture, tid + 1);

		printf("%i %f %f %f\n", tid, signal_before, signal_center, signal_after);

		d_vec[tid] = (signal_center + signal_before + signal_after) / 3.f;
		
	}
}


/********/
/* MAIN */
/********/
int main() {
	
	const int N = 10;

	// --- Input host array declaration and initialization
	float *h_arr = (float *)malloc(N * sizeof(float));
	for (int i = 0; i < N; i++) h_arr[i] = (float)i;

	// --- Output host and device array vectors
	float *h_vec = (float *)malloc(N * sizeof(float));
	float *d_vec;	gpuErrchk(cudaMalloc(&d_vec, N * sizeof(float)));
	
	// --- CUDA array declaration and texture memory binding; CUDA array initialization
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	//Alternatively
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray *d_arr;	gpuErrchk(cudaMallocArray(&d_arr, &channelDesc, N, 1));
	gpuErrchk(cudaMemcpyToArray(d_arr, 0, 0, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

	cudaBindTextureToArray(signal_texture, d_arr); 
    signal_texture.normalized = false; 
    signal_texture.addressMode[0] = cudaAddressModeWrap;
	
	// --- Kernel execution
	median_filter_periodic_boundary<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_vec, N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_vec, d_vec, N * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i=0; i<N; i++) printf("h_vec[%i] = %f\n", i, h_vec[i]);

	printf("Test finished\n");
	
	return 0;
}


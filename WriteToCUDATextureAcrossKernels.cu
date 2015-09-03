#include <stdio.h>

#include "TimingGPU.cuh"
#include "Utilities.cuh"

texture<float, 1, cudaReadModeElementType> signal_texture;

#define BLOCKSIZE 32

/*************************************************/
/* KERNEL FUNCTION FOR MEDIAN FILTER CALCULATION */
/*************************************************/
__global__ void median_filter_periodic_boundary(float * __restrict__ d_out, const unsigned int N){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N) {

		float signal_center = tex1D(signal_texture, (float)(tid + 0.5 - 0) / (float)N);
		float signal_before = tex1D(signal_texture, (float)(tid + 0.5 - 1) / (float)N);
		float signal_after  = tex1D(signal_texture, (float)(tid + 0.5 + 1) / (float)N);

		d_out[tid] = (signal_center + signal_before + signal_after) / 3.f;
		
	}
}

/*************************************************/
/* KERNEL FUNCTION FOR MEDIAN FILTER CALCULATION */
/*************************************************/
__global__ void square(float * __restrict__ d_vec, const size_t pitch, const unsigned int N){

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N) d_vec[tid] = 2.f * tid;

}

/********/
/* MAIN */
/********/
int main() {
	
	const int N = 10;																				 

	// --- Input/output host array declaration and initialization
	float *h_vec = (float *)malloc(N * sizeof(float));
	for (int i = 0; i < N; i++) h_vec[i] = (float)i;

	// --- Input/output host and device array vectors
	size_t pitch;
	float *d_vec;	gpuErrchk(cudaMallocPitch(&d_vec, &pitch, N * sizeof(float), 1));
	printf("pitch = %i\n", pitch);
	float *d_out;	gpuErrchk(cudaMalloc(&d_out, N * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));
	
	// --- CUDA texture memory binding and properties definition
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	//Alternatively
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	size_t texture_offset = 0;
	gpuErrchk(cudaBindTexture2D(&texture_offset, signal_texture, d_vec, channelDesc, N, 1, pitch)); 
    signal_texture.normalized = true; 
    signal_texture.addressMode[0] = cudaAddressModeWrap;
	
	// --- Median filter kernel execution
	median_filter_periodic_boundary<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_out, N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_vec, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
	printf("\n\nFirst filtering\n");
	for (int i=0; i<N; i++) printf("h_vec[%i] = %f\n", i, h_vec[i]);

	// --- Square kernel execution
	square<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_vec, pitch, N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(h_vec, d_vec, N * sizeof(float), cudaMemcpyDeviceToHost));
	printf("\n\nSquaring\n");
	for (int i=0; i<N; i++) printf("h_vec[%i] = %f\n", i, h_vec[i]);

	// --- Median filter kernel execution
	median_filter_periodic_boundary<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_out, N);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	printf("\n\nSecond filtering\n");
	gpuErrchk(cudaMemcpy(h_vec, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i=0; i<N; i++) printf("h_vec[%i] = %f\n", i, h_vec[i]);

	printf("Test finished\n");
	
	return 0;
}


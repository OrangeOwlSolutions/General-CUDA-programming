#include <stdio.h>

#include "TimingGPU.cuh"
#include "Utilities.cuh"

texture<float, 1, cudaReadModeElementType> signal_texture;

#define BLOCKSIZE 256

/***********************************************************************************/
/* KERNEL FUNCTION FOR 1D MOVING AVERAGE WITH PERIODIC BOUNDARY AND TEXTURE MEMORY */
/***********************************************************************************/
__global__ void movingAverage1dPeriodicBoundaryTexture(float * __restrict__ d_result, const int RADIUS, const unsigned int N){

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N) {

		float average = 0.f;
		
		for (int k = -RADIUS; k <= RADIUS; k++) {
			average = average + tex1D(signal_texture, (float)(tid - k + 0.5f)/(float)N);
		}

		d_result[tid] = average / (2.f * (float)RADIUS + 1.f);

	}
}

/*************************/
/* CPU REFERENCE VERSION */
/*************************/
void movingAverage1dCPU(float * __restrict h_result, const float * __restrict h_in, const int RADIUS, const unsigned int N) {

	for (int i = 0; i < N; i++) {
	
		float	average = 0.f;

		for (int k = -RADIUS; k <= RADIUS; k++) {
			
			int		index	= i - k;	
			if		((i - k) <  0)	index = index + N;
			else if ((i - k) >= N)	index = index - N;
			
			average = average + h_in[index];
		}

		h_result[i] = average / (2.f * (float)RADIUS + 1.f);
		
	}
}

/********/
/* MAIN */
/********/
int main() {
	
	const int N			= 10;
	const int RADIUS	= 2;

	// --- Input host array declaration and initialization
	float *h_in = (float *)malloc(N * sizeof(float));
	for (int i = 0; i < N; i++) h_in[i] = (float)i;

	// --- Output host and device array vectors
	float *h_result_GPU = (float *)malloc(N * sizeof(float));
	float *h_result_CPU = (float *)malloc(N * sizeof(float));
	float *d_result;	gpuErrchk(cudaMalloc(&d_result, N * sizeof(float)));
	
	// --- CUDA array declaration and texture memory binding; CUDA array initialization
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	//Alternatively
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray *d_arr;	gpuErrchk(cudaMallocArray(&d_arr, &channelDesc, N, 1));
	gpuErrchk(cudaMemcpyToArray(d_arr, 0, 0, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
	cudaBindTextureToArray(signal_texture, d_arr); 
    // --- Pay attention: cudaAddressModeWrap is available only with normalized = true
	signal_texture.normalized = true; 
    signal_texture.addressMode[0] = cudaAddressModeWrap;
	
	// --- Kernel execution
	movingAverage1dPeriodicBoundaryTexture<<<iDivUp(N, BLOCKSIZE), BLOCKSIZE>>>(d_result, RADIUS, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	gpuErrchk(cudaMemcpy(h_result_GPU, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- CPU execution
	movingAverage1dCPU(h_result_CPU, h_in, RADIUS, N);
	
	for (int i=0; i<N; i++) {
        if (h_result_CPU[i] != h_result_GPU[i]) { printf("Error at i = %i! Host = %f; Device = %f\n", i, h_result_CPU[i], h_result_GPU[i]); return; };
    }

    printf("Test passed\n");
	
	return 0;
}

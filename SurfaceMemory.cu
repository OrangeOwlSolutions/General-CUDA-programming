#include <stdio.h>

#include "TimingGPU.cuh"
#include "Utilities.cuh"

surface<void, cudaSurfaceType1D> surfD;

/*******************/
/* KERNEL FUNCTION */
/*******************/
__global__ void SurfaceMemoryWrite(const int N) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	surf1Dwrite((float)tid, surfD, tid * sizeof(float), cudaBoundaryModeTrap);
}

/********/
/* MAIN */
/********/
int main() {
	
	const int N = 10;
	
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	//Alternatively
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray *d_arr;	gpuErrchk(cudaMallocArray(&d_arr, &channelDesc, N, 1, cudaArraySurfaceLoadStore));
	gpuErrchk(cudaBindSurfaceToArray(surfD, d_arr));

	SurfaceMemoryWrite<<<1, N>>>(N);

	float *h_arr = new float[N];
	gpuErrchk(cudaMemcpyFromArray(h_arr, d_arr, 0, 0, N * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i=0; i<N; i++) printf("h_arr[%i] = %f\n", i, h_arr[i]);

	return 0;
}

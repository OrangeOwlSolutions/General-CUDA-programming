#include <stdio.h>

#include "Utilities.cuh"

#define NUMBLOCKS  512
#define NUMTHREADS 512 * 2

/***************/
/* LOCK STRUCT */
/***************/
struct Lock {

	int *d_state;

	// --- Constructor
	Lock(void) {
		int h_state = 0;										// --- Host side lock state initializer
		gpuErrchk(cudaMalloc((void **)&d_state, sizeof(int)));	// --- Allocate device side lock state
		gpuErrchk(cudaMemcpy(d_state, &h_state, sizeof(int), cudaMemcpyHostToDevice)); // --- Initialize device side lock state
	}

	// --- Destructor
	__host__ __device__ ~Lock(void) { 
#if !defined(__CUDACC__)
		gpuErrchk(cudaFree(d_state)); 
#else

#endif	
	}

	// --- Lock function
	__device__ void lock(void) { while (atomicCAS(d_state, 0, 1) != 0); }

	// --- Unlock function
	__device__ void unlock(void) { atomicExch(d_state, 0); }
};

/*************************************/
/* BLOCK COUNTER KERNEL WITHOUT LOCK */
/*************************************/
__global__ void blockCounterUnlocked(int *nblocks) {
	
	if (threadIdx.x == 0) {	*nblocks = *nblocks + 1; }
}

/**********************************/
/* BLOCK COUNTER KERNEL WITH LOCK */
/**********************************/
__global__ void blockCounterLocked(Lock lock, int *nblocks) {

	if (threadIdx.x == 0) {
		lock.lock();
		*nblocks = *nblocks + 1;
		lock.unlock();
	}
}

/****************************************/
/* BLOCK COUNTER KERNEL WITH WRONG LOCK */
/****************************************/
__global__ void blockCounter2(Lock lock, int *nblocks) {
	
	lock.lock();
	if (threadIdx.x == 0) { *nblocks = *nblocks + 1; }
	lock.unlock();
}

/********/
/* MAIN */
/********/
int main(){

	int h_counting, *d_counting;
	Lock lock;

	gpuErrchk(cudaMalloc(&d_counting, sizeof(int)));

	// --- Unlocked case
	h_counting = 0;
	gpuErrchk(cudaMemcpy(d_counting, &h_counting, sizeof(int), cudaMemcpyHostToDevice));

	blockCounterUnlocked << <NUMBLOCKS, NUMTHREADS >> >(d_counting);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(&h_counting, d_counting, sizeof(int), cudaMemcpyDeviceToHost));
	printf("Counting in the unlocked case: %i\n", h_counting);

	// --- Locked case
	h_counting = 0;
	gpuErrchk(cudaMemcpy(d_counting, &h_counting, sizeof(int), cudaMemcpyHostToDevice));

	blockCounterLocked << <NUMBLOCKS, NUMTHREADS >> >(lock, d_counting);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(&h_counting, d_counting, sizeof(int), cudaMemcpyDeviceToHost));
	printf("Counting in the locked case: %i\n", h_counting);

	gpuErrchk(cudaFree(d_counting));
}

#include <stdio.h>

texture<float, 1, cudaReadModeElementType> texture_clamp;
texture<float, 1, cudaReadModeElementType> texture_border;
texture<float, 1, cudaReadModeElementType> texture_wrap;
texture<float, 1, cudaReadModeElementType> texture_mirror;

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/******************************/
/* CUDA ADDRESS MODE CLAMPING */
/******************************/
__global__ void Test_texture_clamping(const int M) {

    printf("Texture clamping - i = %i; value = %f\n", -threadIdx.x, tex1D(texture_clamp, -(float)threadIdx.x));
    printf("Texture clamping - i = %i; value = %f\n", M + threadIdx.x, tex1D(texture_clamp, (float)(M + threadIdx.x)));

}

/****************************/
/* CUDA ADDRESS MODE BORDER */
/****************************/
__global__ void Test_texture_border(const int M) {

    printf("Texture border - i = %i; value = %f\n", -threadIdx.x, tex1D(texture_border, -(float)threadIdx.x));
    printf("Texture border - i = %i; value = %f\n", M + threadIdx.x, tex1D(texture_border, (float)(M + threadIdx.x)));

}

/**************************/
/* CUDA ADDRESS MODE WRAP */
/**************************/
__global__ void Test_texture_wrap(const int M) {

    printf("Texture wrap - i = %i; value = %f\n", -threadIdx.x, tex1D(texture_wrap, -(float)threadIdx.x/(float)M));
    printf("Texture wrap - i = %i; value = %f\n", M + threadIdx.x, tex1D(texture_wrap, (float)(M + threadIdx.x)/(float)M));

}

/****************************/
/* CUDA ADDRESS MODE MIRROR */
/****************************/
__global__ void Test_texture_mirror(const int M) {

    printf("Texture mirror - i = %i; value = %f\n", -threadIdx.x, tex1D(texture_mirror, -(float)threadIdx.x/(float)M));
    printf("Texture mirror - i = %i; value = %f\n", M + threadIdx.x, tex1D(texture_mirror, (float)(M + threadIdx.x)/(float)M));

}

/********/
/* MAIN */
/********/
void main(){

    const int M = 4;

    // --- Host side memory allocation and initialization
    float *h_data = (float*)malloc(M * sizeof(float));

    for (int i=0; i<M; i++) h_data[i] = (float)i;

    // --- Texture clamping
    cudaArray* d_data_clamping = NULL; gpuErrchk(cudaMallocArray(&d_data_clamping, &texture_clamp.channelDesc, M, 1)); 
    gpuErrchk(cudaMemcpyToArray(d_data_clamping, 0, 0, h_data, M * sizeof(float), cudaMemcpyHostToDevice)); 
    cudaBindTextureToArray(texture_clamp, d_data_clamping); 
    texture_clamp.normalized = false; 
    texture_clamp.addressMode[0] = cudaAddressModeClamp;

    dim3 dimBlock(2 * M, 1); dim3 dimGrid(1, 1);
    Test_texture_clamping<<<dimGrid,dimBlock>>>(M);

    printf("\n\n\n");

    // --- Texture border
    cudaArray* d_data_border = NULL; gpuErrchk(cudaMallocArray(&d_data_border, &texture_border.channelDesc, M, 1)); 
    gpuErrchk(cudaMemcpyToArray(d_data_border, 0, 0, h_data, M * sizeof(float), cudaMemcpyHostToDevice)); 
    cudaBindTextureToArray(texture_border, d_data_border); 
    texture_border.normalized = false; 
    texture_border.addressMode[0] = cudaAddressModeBorder;

    Test_texture_border<<<dimGrid,dimBlock>>>(M);

    printf("\n\n\n");

    // --- Texture wrap
    cudaArray* d_data_wrap = NULL; gpuErrchk(cudaMallocArray(&d_data_wrap, &texture_wrap.channelDesc, M, 1)); 
    gpuErrchk(cudaMemcpyToArray(d_data_wrap, 0, 0, h_data, M * sizeof(float), cudaMemcpyHostToDevice)); 
    cudaBindTextureToArray(texture_wrap, d_data_wrap); 
    texture_wrap.normalized = true; 
    texture_wrap.addressMode[0] = cudaAddressModeWrap;

    Test_texture_wrap<<<dimGrid,dimBlock>>>(M);

    printf("\n\n\n");

    // --- Texture mirror
    cudaArray* d_data_mirror = NULL; gpuErrchk(cudaMallocArray(&d_data_mirror, &texture_mirror.channelDesc, M, 1)); 
    gpuErrchk(cudaMemcpyToArray(d_data_mirror, 0, 0, h_data, M * sizeof(float), cudaMemcpyHostToDevice)); 
    cudaBindTextureToArray(texture_mirror, d_data_mirror); 
    texture_mirror.normalized = true ; 
    texture_mirror.addressMode[0] = cudaAddressModeMirror;

    Test_texture_mirror<<<dimGrid,dimBlock>>>(M);

    printf("\n\n\n");
}

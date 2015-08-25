#include <cuda.h>

__global__ void filterData(const float *d_data,
                           const float *d_numerator, 
                           float *d_filteredData, 
                           const int numeratorLength,
                           const int filteredDataLength)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;

    if (i < filteredDataLength)
    {
        for (int j = 0; j < numeratorLength; j++)
        {
            // The first (numeratorLength-1) elements contain the filter state
            sum += d_numerator[j] * d_data[i + numeratorLength - j - 1];
        }
    }

    d_filteredData[i] = sum;
}

int main(void)
{
    // (Skipping error checks to make code more readable)

    int dataLength = 18042;
    int filteredDataLength = 16384;
    int numeratorLength= 1659;

    // Pointers to data, filtered data and filter coefficients
    // (Skipping how these are read into the arrays)
    float *h_data = new float[dataLength];
    float *h_filteredData = new float[filteredDataLength];
    float *h_filter = new float[numeratorLength];


    // Create device pointers
    float *d_data = nullptr;
    cudaMalloc((void **)&d_data, dataLength * sizeof(float));

    float *d_numerator = nullptr;
    cudaMalloc((void **)&d_numerator, numeratorLength * sizeof(float));

    float *d_filteredData = nullptr;
    cudaMalloc((void **)&d_filteredData, filteredDataLength * sizeof(float));


    // Copy data to device
    cudaMemcpy(d_data, h_data, dataLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numerator, h_numerator, numeratorLength * sizeof(float), cudaMemcpyHostToDevice);  

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (filteredDataLength + threadsPerBlock - 1) / threadsPerBlock;
    filterData<<<blocksPerGrid,threadsPerBlock>>>(d_data, d_numerator, d_filteredData, numeratorLength, filteredDataLength);

    // Copy results to host
    cudaMemcpy(h_filteredData, d_filteredData, filteredDataLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_data);
    cudaFree(d_numerator);
    cudaFree(d_filteredData);

    // Do stuff with h_filteredData...

    // Clean up some more
    delete [] h_data;
    delete [] h_filteredData;
    delete [] h_filter;
}

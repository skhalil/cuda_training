#include <assert.h>
#include <iostream>
#include <stdio.h>
// Here you can set the device ID that was assigned to you
#define MYDEVICE 0


// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);

// Part 3 of 5: implement the kernel
__global__ void myFirstKernel(int *d_a)
{
  //printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d\n", blockIdx.x, blockDim.x, threadIdx.x);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //d_a[i] = i;
  d_a[i] = blockIdx.x + threadIdx.x;  
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    cudaSetDevice(MYDEVICE);
    // pointer for host memory
    int *h_a;

    // pointer for device memory
    int *d_a;

    // define grid and block size
    int numBlocks = 8;
    int numThreadsPerBlock = 8;

    // Part 1 of 5: allocate host and device memory
    size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
    h_a = (int *) malloc(memSize);
    cudaMalloc((void**)&d_a, memSize);

    // Part 2 of 5: configure and launch kernel
    dim3 dimGrid(8,1,1);
    dim3 dimBlock(8,1,1);
    myFirstKernel<<<dimGrid, dimBlock>>>(d_a);

    // block until the device has completed
    cudaDeviceSynchronize();

    // check if kernel execution generated an error
    checkCUDAError("kernel execution");

    // Part 4 of 5: device to host copy
    cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy");

    // Part 5 of 5: verify the data returned to the host is correct
    for (int i = 0; i <  8        ; ++i)
    {
        for (int j = 0; j <       8            ; ++j)
        {
            //printf("h_a[%d] = %d\n", i*numThreadsPerBlock + j, i*numThreadsPerBlock + j);
            //assert(h_a[i * numThreadsPerBlock + j] == i *numThreadsPerBlock + j);
            //printf("h_a[%d] = %d\n", i*numThreadsPerBlock + j, i + j);
            assert(h_a[i * numThreadsPerBlock + j] == i + j);
        }
    }

    // free device memory
    cudaFree(d_a);

    // free host memory
    free(h_a);

    // If the program makes it this far, then the results are correct and
    // there are no run-time errors.  Good work!
    std::cout << "Correct!" << std::endl;

    return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        std::cerr << "Cuda error: " << msg << " " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }                         
}

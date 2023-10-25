#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "cuda.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA kernel 
__global__ void VectorAdd(int *a, int *b, int *c, int length){
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tId< length){
        c[tId] = a[tId] + b[tId];
    }
}

int main(){
    // Get vector length
	int userInput;
    printf("Enter the length of the vectors in range of hundreds and thousands: ");
    scanf("%d", &userInput);

    int *h_a, *h_b, *h_c;   // initialize host vectors 
    int *d_a, *d_b, *d_c;   // initialize device vectors 
    int size = userInput * sizeof(int);

    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);
    h_c = (int *)malloc(size);

    // calculate grid and vector size 
    int blockSize = 256;
    int numBlocks = (userInput + blockSize - 1) / blockSize;  

    // initialize vectors on host
    for (int i = 0; i < userInput; ++i) {
        h_a[i] = h_b[i] = i;
    }

    // GPU memory allocation
    gpuErrchk(cudaMalloc((void **)&d_a, size));
    gpuErrchk(cudaMalloc((void **)&d_b, size));
    gpuErrchk(cudaMalloc((void **)&d_c, size));

    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    VectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, userInput);

    // Copy the array back to host 
    gpuErrchk(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    printf("Vector addition result:\n");
    for (int i = 0; i < userInput; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free GPU memory 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free CPU memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;


}

#include <cuda.h>
#include <stdio.h>

#define N 4

// kernel function
__global__ void thread_function(){
	int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	printf("thread %2d: (%d, %d)\n", threadId, threadIdx.x, threadIdx.y);
}

int main(){
	// define dimensions
	dim3 threadDims;
	threadDims.x = N;
	threadDims.y = N;
	threadDims.z = 1;

	// execute kernel
	thread_function<<<1, threadDims>>>();
	
	// wait for device to finish
	cudaDeviceSynchronize();
}

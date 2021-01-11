#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define N 32
#define NUM_BLOCKS 8

// kernel function
__global__ void A_add_B(int *A, int *B, int *C){
	int i = blockIdx.x * (N / NUM_BLOCKS) + threadIdx.x;
	
	C[i] = A[i] + B[i];
}

int main(){

	// host memory
	int A[N];
	int B[N];
	int i;
	
	// device memory
	int *A_gpu;
	int *B_gpu;
	
	// unified memory
	int *C;
	
	// allocate memory
	cudaMalloc(&A_gpu, sizeof(int) * N);
	cudaMalloc(&B_gpu, sizeof(int) * N);
	cudaMallocManaged(&C, sizeof(int) * N);
	
	// fill vectors with random values [0, 99]
	srand(time(NULL));
	for(i = 0; i < N; i++){
		A[i] = rand() % 100;
		B[i] = rand() % 100;
	}
	
	// copy from host to device
	cudaMemcpy(A_gpu, A, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, N * sizeof(int), cudaMemcpyHostToDevice);
	
	// define dimensions
	dim3 threadDims;
	threadDims.x = N / NUM_BLOCKS;
	threadDims.y = 1;
	threadDims.z = 1;

	// execute kernel
	A_add_B<<<NUM_BLOCKS, threadDims>>>(A_gpu, B_gpu, C);
	
	// wait for device to finish
	cudaDeviceSynchronize();
	
	// print results
	for(i = 0; i < N; i++){
		printf("%d = %d + %d\n", C[i], A[i], B[i]);
	}
	
	// deallocate memory
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C);
}

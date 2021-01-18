#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define N (UINT_MAX / 2 + 1)

int find_max_serial(unsigned int *A){
	unsigned int i;
	unsigned int max;

	max = A[0];
	for(i = 1; i < N; i++){
		max = (A[i] > max) ? A[i] : max;
	}
	
	return max;
}

__global__ void find_max_parallel(unsigned int *A, unsigned int *max_block){
	unsigned int i;
	unsigned int max;
	
	unsigned int low_index = threadIdx.x * (N / blockDim.x);
	unsigned int high_index = (threadIdx.x + 1) * (N / blockDim.x);
	
	max = A[low_index];
	for(i = low_index + 1; i < high_index; i++){
		max = (A[i] > max) ? A[i] : max;
	}
	
	atomicMax(max_block, max);
}

int main(){
	unsigned int *A;
	unsigned int *A_gpu;
	unsigned int i;
	unsigned int max;
	unsigned int *max_gpu;
	
	// cuda events for timing
	cudaEvent_t start, stop;
	float time_elapsed;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// host memory allocation
	A = (unsigned int*) calloc(N, sizeof(unsigned int));
	
	// device memory allocation
	cudaMalloc(&A_gpu, N * sizeof(unsigned int));
	cudaMalloc(&max_gpu, sizeof(unsigned int));
	
	// fill with random values	
	printf("Filling vector with random values...\n");
	srand(time(NULL));
	for(i = 0; i < N; i++){
		A[i] = (unsigned int) (rand() % UINT_MAX);
	}
	
	// copy from host to device
	cudaMemcpy(A_gpu, A, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	// serial calculation
	printf("Starting serial computation...\n");
	
	cudaEventRecord(start);
	max = find_max_serial(A);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_elapsed, start, stop);
	
	printf("Max is %u, Time: %.3f ms\n", max, time_elapsed);
	
	// define dimensions
	dim3 threadDims;
	threadDims.x = 1;
	threadDims.y = 1;
	threadDims.z = 1;
	
	// parallel calculation
	for(i = 2; i <= 1024; i*=2){
		threadDims.x = i;
		
		printf("Starting parallel computation with %u threads...\n", i);
		
		max = 0;
		cudaMemcpy(max_gpu, &max, sizeof(unsigned int), cudaMemcpyHostToDevice);
	
		cudaEventRecord(start);
		find_max_parallel<<<1, threadDims>>>(A_gpu, max_gpu);
		cudaEventRecord(stop);
		
		cudaDeviceSynchronize();
		
		cudaMemcpy(&max, max_gpu, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_elapsed, start, stop);
		
		printf("Max is %u, Time: %.3f ms\n", max, time_elapsed);
	}
	
	// free memory
	free(A);
	cudaFree(A_gpu);
	cudaFree(max_gpu);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	return 0;	
}



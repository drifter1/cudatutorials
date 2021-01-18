#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define N (UINT_MAX / 2 + 1)

int main(){
	unsigned int *A;
	unsigned int i;
	unsigned int max;
	
	// allocate memory
	A = (unsigned int*) calloc(N, sizeof(unsigned int));
	
	// fill with random values	
	srand(time(NULL));	
	for(i = 0; i < N; i++){
		A[i] = (unsigned int) (rand() % UINT_MAX);
	}
	
	// find maximum value
	max = A[0];
	for(i = 1; i < N; i++){
		max = (A[i] > max) ? A[i] : max;
	}
	
	printf("Max is %u\n", max);
	
	// free memory
	free(A);
	
	return 0;	
}



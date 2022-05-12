#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 1024
#define N (1 << 25)

__global__ void init_kernel (int *a, int span)
{
	int i;
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	for (i=id*span; i<(id+1)*span; i++) a[i] = N - i;
}

__global__ void bitonic_sort_kernel (int *a, int span, unsigned bf_sz, int log2stage)
{
	unsigned index1, index2, lower, pair;
	int temp;

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	
	for (pair=id*(span/2); pair<(id+1)*(span/2); pair++) {
		index1 = (pair >> log2stage) << 1;
		index2 = index1 | 1;
		lower = pair & ((1 << log2stage) - 1);
		index1 = (index1 << log2stage) | lower;
		index2 = (index2 << log2stage) | lower;
		if ((bf_sz & index1) == 0) {
			if (a[index1] > a[index2]) {
				temp = a[index1];
				a[index1] = a[index2];
				a[index2] = temp;
			}
		}
		else {
			if (a[index1] < a[index2]) {
                               	temp = a[index1];
                               	a[index1] = a[index2];
                               	a[index2] = temp;
                       	}
		}
	}
}	

int main (int argc, char *argv[])
{
	int i, nthreads, init_log2stage, log2stage;
	int *a;
	unsigned bf_sz, stage;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	if (argc != 2) {
		printf("Need number of threads. Aborting...\n");
		exit(0);
	}

	nthreads = atoi(argv[1]);
	assert((nthreads > 0) && ((nthreads % THREADS_PER_BLOCK) == 0));

	cudaMallocManaged((void**)&a, sizeof(int)*N);

	init_kernel<<<nthreads/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(a, N/nthreads);

	cudaDeviceSynchronize();

	gettimeofday(&tv0, &tz0);

	for (bf_sz = 2, init_log2stage = 0; bf_sz <= N; bf_sz = bf_sz*2, init_log2stage++) {
                for (stage = bf_sz/2, log2stage = init_log2stage; stage > 0; stage = stage/2, log2stage--) {
			bitonic_sort_kernel<<<nthreads/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(a, N/nthreads, bf_sz, log2stage);
		}
	}

	cudaDeviceSynchronize();

	gettimeofday(&tv1, &tz1);

	for (i=0; i<N-1; i++) {
		if (a[i] > a[i+1]) printf("Error at position %d, a[%d] = %d, a[%d] = %d\n", i, i, a[i], i+1, a[i+1]);
	}
	
	printf("Time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}

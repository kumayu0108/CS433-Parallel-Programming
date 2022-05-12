#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 1024
#define N (2*THREADS_PER_BLOCK)

__global__ void init_kernel (int *a)
{
	a[2*threadIdx.x] = 2*THREADS_PER_BLOCK - 2*threadIdx.x;
	a[2*threadIdx.x+1] = 2*THREADS_PER_BLOCK - 2*threadIdx.x - 1;
}

__global__ void bitonic_sort_kernel (int *a)
{
	unsigned bf_sz, stage, index1, index2, lower;
	int init_log2stage, log2stage, temp;
	
	for (bf_sz = 2, init_log2stage = 0; bf_sz <= N; bf_sz = bf_sz*2, init_log2stage++) {
		for (stage = bf_sz/2, log2stage = init_log2stage; stage > 0; stage = stage/2, log2stage--) {
			index1 = (threadIdx.x >> log2stage) << 1;
			index2 = index1 | 1;
			lower = threadIdx.x & ((1 << log2stage) - 1);
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
			__syncthreads();
		}
	}
}	

int main (int argc, char *argv[])
{
	int i;
	int *a;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	cudaMallocManaged((void**)&a, sizeof(int)*N);

	init_kernel<<<1, THREADS_PER_BLOCK>>>(a);

	cudaDeviceSynchronize();

	gettimeofday(&tv0, &tz0);

	bitonic_sort_kernel<<<1, THREADS_PER_BLOCK>>>(a);

	cudaDeviceSynchronize();

	gettimeofday(&tv1, &tz1);

	for (i=0; i<N-1; i++) {
		if (a[i] > a[i+1]) printf("Error at position %d, a[%d] = %d, a[%d] = %d\n", i, i, a[i], i+1, a[i+1]);
	}
	
	printf("Time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}

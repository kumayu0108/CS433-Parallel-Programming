#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 1024
#define N (1 << 25)

__device__ int count = 0;
__device__ volatile int barrier_flag = 0;

__global__ void init_kernel (int *a, int span)
{
	int i;
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	for (i=id*span; i<(id+1)*span; i++) a[i] = N - i;
}

__global__ void bitonic_sort_kernel (int *a, int span)
{
	unsigned bf_sz, stage, index1, index2, lower, pair;
	int init_log2stage, log2stage, temp;

	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int local_sense = 0, last_count;
	
	for (bf_sz = 2, init_log2stage = 0; bf_sz <= N; bf_sz = bf_sz*2, init_log2stage++) {
		for (stage = bf_sz/2, log2stage = init_log2stage; stage > 0; stage = stage/2, log2stage--) {
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
			local_sense = (local_sense ? 0 : 1);
			__syncthreads();
			if (threadIdx.x == 0) {
				last_count = atomicAdd(&count, 1);
				if (last_count == ((N/(span*THREADS_PER_BLOCK)) - 1)) {
					count = 0;
					barrier_flag = local_sense;
				}
			}
			while (barrier_flag != local_sense);
		}
	}
}	

__global__ void bitonic_sort_kernel_cg (int *a, int span)
{
        unsigned bf_sz, stage, index1, index2, lower, pair;
        int init_log2stage, log2stage, temp;

        int id = threadIdx.x + blockIdx.x*blockDim.x;

        for (bf_sz = 2, init_log2stage = 0; bf_sz <= N; bf_sz = bf_sz*2, init_log2stage++) {
                for (stage = bf_sz/2, log2stage = init_log2stage; stage > 0; stage = stage/2, log2stage--) {
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
			cg::grid_group grid = cg::this_grid();
			grid.sync();
                }
        }
}

int main (int argc, char *argv[])
{
	int i, nthreads;
	int *a;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;
	int num_elements_per_thread;

	if (argc != 2) {
		printf("Need number of threads. Aborting...\n");
                exit(0);
        }

	nthreads = atoi(argv[1]);

	cudaMallocManaged((void**)&a, sizeof(int)*N);

	int numBlocksPerSm = 0, numBlocks;
	cudaDeviceProp deviceProp;
	int device = -1;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&deviceProp, device);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, bitonic_sort_kernel, THREADS_PER_BLOCK, 0);
	numBlocks = deviceProp.multiProcessorCount*numBlocksPerSm;
	printf("Max number of blocks per SM: %d, number of SMs: %d, number of blocks: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount, numBlocks);
	while ((numBlocks & (numBlocks - 1)) != 0) numBlocks--;
	if (nthreads > (THREADS_PER_BLOCK*numBlocks)) nthreads=THREADS_PER_BLOCK*numBlocks;
	printf("Number of blocks: %d, Threads per block: %d, Total number of threads: %d\n", nthreads/THREADS_PER_BLOCK, THREADS_PER_BLOCK, nthreads);

	init_kernel<<<nthreads/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(a, N/nthreads);
        cudaDeviceSynchronize();

	num_elements_per_thread = N/nthreads;
	int supportsCoopLaunch = 0;
	cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
	if (supportsCoopLaunch) {
		void *kernelArgs[] = {(void*)&a, (void*)&num_elements_per_thread};
		dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
		dim3 dimGrid(nthreads/THREADS_PER_BLOCK, 1, 1);

		gettimeofday(&tv0, &tz0);
		cudaLaunchCooperativeKernel((void*)bitonic_sort_kernel_cg, dimGrid, dimBlock, kernelArgs);
	}
	else {
		gettimeofday(&tv0, &tz0);
		bitonic_sort_kernel<<<nthreads/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(a, N/nthreads);
	}
	cudaDeviceSynchronize();
        gettimeofday(&tv1, &tz1);

	for (i=0; i<N-1; i++) {
		if (a[i] > a[i+1]) printf("Error at position %d, a[%d] = %d, a[%d] = %d\n", i, i, a[i], i+1, a[i+1]);
	}
	
	printf("Time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define SIZE (1<<28)

__global__ void vadd_kernel (float *a, float *b, float *c, int span)
{
	int i;
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	
	for (i=span*id; i<span*(id+1); i++) {
		c[i] = a[i] + b[i];
	}
}	

int main (int argc, char *argv[])
{
	int i;
	float *a, *b, *c, *gpu_a, *gpu_b, *gpu_c;
	int nthreads;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);
	assert((nthreads & (nthreads - 1)) == 0);

	cudaHostAlloc((void**)&a, sizeof(float)*SIZE, cudaHostAllocMapped);
	for (i=0; i<SIZE; i++) a[i] = 1;

	cudaHostAlloc((void**)&b, sizeof(float)*SIZE, cudaHostAllocMapped);
        for (i=0; i<SIZE; i++) b[i] = 1.5;

	cudaHostAlloc((void**)&c, sizeof(float)*SIZE, cudaHostAllocMapped);

	cudaHostGetDevicePointer ((void**)&gpu_a, (void*)a, 0);
	cudaHostGetDevicePointer ((void**)&gpu_b, (void*)b, 0);
	cudaHostGetDevicePointer ((void**)&gpu_c, (void*)c, 0);

	gettimeofday(&tv0, &tz0);

	if (nthreads < 16) {
		vadd_kernel<<<1, nthreads>>>(gpu_a, gpu_b, gpu_c, SIZE/nthreads);
	}
	else {
		vadd_kernel<<<nthreads/8, 8>>>(gpu_a, gpu_b, gpu_c, SIZE/nthreads);
	}
	cudaDeviceSynchronize();

	gettimeofday(&tv2, &tz2);
	
	printf("Random element: %lf, time: %ld microseconds\n", c[random() % SIZE], (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
	return 0;
}

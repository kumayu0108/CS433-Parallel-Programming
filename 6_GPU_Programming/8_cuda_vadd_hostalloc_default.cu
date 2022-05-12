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
	struct timeval tv0, tv1, tv2;
	struct timezone tz0, tz1, tz2;

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);
	assert((nthreads & (nthreads - 1)) == 0);

	a = (float*)malloc(sizeof(float)*SIZE);
	assert(a != NULL);
	for (i=0; i<SIZE; i++) a[i] = 1;

	b = (float*)malloc(sizeof(float)*SIZE);
        assert(b != NULL);
        for (i=0; i<SIZE; i++) b[i] = 1.5;

	cudaHostAlloc((void**)&c, sizeof(float)*SIZE, cudaHostAllocDefault);

	cudaMalloc((void**)&gpu_a, sizeof(float)*SIZE);
	cudaMalloc((void**)&gpu_b, sizeof(float)*SIZE);
	cudaMalloc((void**)&gpu_c, sizeof(float)*SIZE);

	gettimeofday(&tv0, &tz0);
	cudaMemcpy(gpu_a, a, sizeof(float)*SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, sizeof(float)*SIZE, cudaMemcpyHostToDevice);
	gettimeofday(&tv1, &tz1);

	if (nthreads < 16) {
		vadd_kernel<<<1, nthreads>>>(gpu_a, gpu_b, gpu_c, SIZE/nthreads);
	}
	else {
		vadd_kernel<<<nthreads/8, 8>>>(gpu_a, gpu_b, gpu_c, SIZE/nthreads);
	}
        cudaMemcpy(c, gpu_c, sizeof(float)*SIZE, cudaMemcpyDeviceToHost);

	gettimeofday(&tv2, &tz2);
	
	printf("Random element: %lf, time: %ld microseconds, input copy time: %ld microseconds, compute time (includes output copy): %ld microseconds\n", c[random() % SIZE], (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec), (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec), (tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
	return 0;
}

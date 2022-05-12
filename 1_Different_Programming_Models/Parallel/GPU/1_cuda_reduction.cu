#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define SIZE (1<<30)

__global__ void reduction_kernel (float *a, int span, double *target)
{
	int i;
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	target[id] = 0;
	
	for (i=span*id; i<span*(id+1); i++) {
		target[id] += (a[i]*a[i]);
	}
}	

int main (int argc, char *argv[])
{
	int i;
	float *a, *gpu_a;
	int nthreads;
	double sum = 0;
	double *private_sum, *gpu_private_sum;
	struct timeval tv0, tv1, tv2;
	struct timezone tz0, tz1, tz2;

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);
	private_sum = (double*)malloc(nthreads*sizeof(double));

	a = (float*)malloc(sizeof(float)*SIZE);
	assert(a != NULL);
	for (i=0; i<SIZE; i++) a[i] = 1;

	cudaMalloc((void**)&gpu_a, sizeof(float)*SIZE);
	cudaMalloc((void**)&gpu_private_sum, sizeof(double)*nthreads);

	gettimeofday(&tv0, &tz0);
	cudaMemcpy(gpu_a, a, sizeof(float)*SIZE, cudaMemcpyHostToDevice);
	gettimeofday(&tv1, &tz1);

	if (nthreads < 16) {
		reduction_kernel<<<1, nthreads>>>(gpu_a, SIZE/nthreads, gpu_private_sum);
	}
	else {
		reduction_kernel<<<nthreads/16, 16>>>(gpu_a, SIZE/nthreads, gpu_private_sum);
	}
        cudaMemcpy(private_sum, gpu_private_sum, sizeof(double)*nthreads, cudaMemcpyDeviceToHost);

	for (i=0; i<nthreads; i++) sum += private_sum[i];

	gettimeofday(&tv2, &tz2);
	
	printf("SUM: %lf, time: %ld microseconds, copy time: %ld microseconds, compute time: %ld microseconds\n", sum, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec), (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec), (tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define N (1ULL << 34)

#define START_X 0
#define END_X 1.0

__managed__ float area;

__device__ float compute_f (float x)
{
	return 1.0/(1.0 + x*x);
}

__global__ void area_kernel (float a, float b, unsigned long long num_intervals_per_thread)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	float sub_interval = (b - a)/N;
	float x = a + id*sub_interval*num_intervals_per_thread;
	unsigned long long i;
	float local_area = 0;
	for (i=0; i<num_intervals_per_thread; i++) {
		local_area += (compute_f (x) + compute_f (x+sub_interval));
		x += sub_interval;
	}
	unsigned mask = 0xffffffff;
	for (i=warpSize/2; i>0; i=i/2) local_area += __shfl_xor_sync(mask, local_area, i);
	if (threadIdx.x == 0) {
		atomicAdd(&area, 0.5*local_area);
	}
}	

int main (int argc, char *argv[])
{
	unsigned long long nthreads;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoll(argv[1]);
	assert((nthreads & (nthreads - 1)) == 0);
	if (nthreads > N) nthreads = N;

	area = 0;
	int device = -1;
        cudaGetDevice(&device);
        cudaMemAdvise(&area, sizeof(float), cudaMemAdviseSetPreferredLocation, device);

	gettimeofday(&tv0, &tz0);

	if (nthreads < 32) {
		area_kernel<<<1, nthreads>>>(START_X, END_X, N/nthreads);
	}
	else {
		area_kernel<<<nthreads/32, 32>>>(START_X, END_X, N/nthreads);
	}
	cudaDeviceSynchronize();

	area = (area*(END_X-START_X))/N;

	gettimeofday(&tv2, &tz2);
	
	printf("Area: %.20f, time: %ld microseconds\n", area, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
	return 0;
}

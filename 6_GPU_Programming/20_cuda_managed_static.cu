#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

__managed__ int x;

__global__ void add_kernel ()
{
	atomicAdd(&x, 1);
}	

int main (int argc, char *argv[])
{
	int nthreads;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);
	assert((nthreads & (nthreads - 1)) == 0);

	x = 0;

	gettimeofday(&tv0, &tz0);

	if (nthreads < 16) {
		add_kernel<<<1, nthreads>>>();
	}
	else {
		add_kernel<<<nthreads/8, 8>>>();
	}
	cudaDeviceSynchronize();

	gettimeofday(&tv2, &tz2);
	
	printf("x: %d, time: %ld microseconds\n", x, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
	return 0;
}

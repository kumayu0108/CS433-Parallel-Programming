#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

__managed__ int x = 0;
__device__ int lock = 0;

__global__ void LockTestKernel (void)
{
	int done = 0;

	while (!done) {
		if (!atomicCAS (&lock, 0, 1)) {
			// Lock holder diverges from the rest
			// and executes the following
			x++;
			lock = 0;
			done = 1;
		}
		// All threads reconverge here
	}
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
	assert ((nthreads < 16) || ((nthreads % 8) == 0));

	gettimeofday(&tv0, &tz0);

	if (nthreads < 16) {
		LockTestKernel<<<1, nthreads>>>();
	}
	else {
		LockTestKernel<<<nthreads/8, 8>>>();
	}
	cudaDeviceSynchronize();

	gettimeofday(&tv2, &tz2);
	
	printf("x: %d, time: %ld microseconds\n", x, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
	return 0;
}

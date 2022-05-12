#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

__managed__ int x = 0;
__device__ int lock = 0;

__device__ void Acquire (int *a)
{
	while (atomicCAS(a, 0, 1));
	// The following printf never executes for more than one thread because the threads in a warp fail to reconverge
        // All threads in a warp must execute the following in lock-step (uncomment the printf)
	//printf("Outside loop: id=%d, x=%d\n", threadIdx.x + blockIdx.x*blockDim.x, x);
	// The Acquire function never returns if there are at least two threads
}

__device__ void Release (int *a)
{
	(*a) = 0;
}

__global__ void LockTestKernel (void)
{
	Acquire (&lock);
	x++;
	Release(&lock);
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

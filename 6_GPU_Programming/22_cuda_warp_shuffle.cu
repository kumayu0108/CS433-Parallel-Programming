#include <stdio.h>
#include <cuda.h>

#define NUM_THREAD_BLOCKS 1

__global__ void Shuffle_test ()
{
	int x, y, z;
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	int mask = 0xffffffff;
	x = __shfl_sync(mask, id, (id+1)%warpSize);
	y = __shfl_up_sync(mask, id, 1);
	z = __shfl_down_sync(mask, id, 1);
	printf("[Block %d, Thread %d (id = %d)] got x=%d, y=%d, z=%d\n", blockIdx.x, threadIdx.x, id, x, y, z);
}

int main (int argc, char **argv)
{
	Shuffle_test <<<NUM_THREAD_BLOCKS, 32>>>();

	cudaDeviceSynchronize();

	return 0;
}

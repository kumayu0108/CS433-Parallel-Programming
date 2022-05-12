#include <stdio.h>
#include <cuda.h>

#define NUM_THREAD_BLOCKS_X 4
#define NUM_THREAD_BLOCKS_Y 3
#define NUM_THREAD_BLOCKS_Z 2

#define NUM_THREADS_PER_BLOCK_X 2
#define NUM_THREADS_PER_BLOCK_Y 2
#define NUM_THREADS_PER_BLOCK_Z 3

__global__ void Hello (char* msg)
{
	printf("[Block (%d, %d, %d), Thread (%d, %d, %d)] %s\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, msg);
}

int main (int argc, char **argv)
{
	const char *msg = "It is elementary, Watson!";
	char *gpu_msg;

	cudaMalloc((void**)&gpu_msg, 64);
	cudaMemcpy(gpu_msg, msg, 64, cudaMemcpyHostToDevice);

	dim3 grid_dims, block_dims;
	grid_dims.x = NUM_THREAD_BLOCKS_X;
	grid_dims.y = NUM_THREAD_BLOCKS_Y;
	grid_dims.z = NUM_THREAD_BLOCKS_Z;
	block_dims.x = NUM_THREADS_PER_BLOCK_X;
	block_dims.y = NUM_THREADS_PER_BLOCK_Y;
	block_dims.z = NUM_THREADS_PER_BLOCK_Z;
	Hello <<<grid_dims, block_dims>>>(gpu_msg);

	cudaError_t err = cudaGetLastError();        // Get error code

   	if ( err != cudaSuccess ) {
      		printf("CUDA Error: %s\n", cudaGetErrorString(err));
      		exit(-1);
   	}

	cudaDeviceSynchronize();

	err = cudaGetLastError();        // Get error code

        if ( err != cudaSuccess ) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
                exit(-1);
        }

	printf("[Main] Done!\n");

	return 0;
}

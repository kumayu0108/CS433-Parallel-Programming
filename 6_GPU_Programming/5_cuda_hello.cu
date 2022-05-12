#include <stdio.h>
#include <cuda.h>

#define NUM_THREAD_BLOCKS 2

__global__ void Hello (char* msg)
{
	printf("[Block %d, Thread %d] %s\n", blockIdx.x, threadIdx.x, msg);
	//printf("[Block %d, Thread %d]\n", blockIdx.x, threadIdx.x);
}

int main (int argc, char **argv)
{
	int nthreads;
	const char *msg = "It is elementary, Watson!";
	char *gpu_msg;

	if (argc != 2) {
		printf("Need thread count. Aborting ...\n");
		exit(0);
	}

	nthreads = atoi(argv[1]);
	if (nthreads <= 0) nthreads = 1;

	cudaMalloc((void**)&gpu_msg, 64);
	cudaMemcpy(gpu_msg, msg, 64, cudaMemcpyHostToDevice);

	//Hello <<<NUM_THREAD_BLOCKS, nthreads/NUM_THREAD_BLOCKS>>>(gpu_msg);
	Hello <<<NUM_THREAD_BLOCKS, 1024>>>(gpu_msg);

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

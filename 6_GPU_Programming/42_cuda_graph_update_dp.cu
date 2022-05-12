#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define N (1 << 15)
#define THRESHOLD (N-100)

__global__ void graph_gen (int *NA, long long *NIA, int *propertyV)
{
        int i;
        int id = threadIdx.x + blockIdx.x*blockDim.x;
	int num_neighbors = id ? id : 1;
	long long na_offset = 0;

	for (i=0; i<id; i++) {
		if (i==0) na_offset++;
		else na_offset += i;
	}

	for (i=0; i<num_neighbors; i++) NA[na_offset+i] = (id+i+1) % N;

	NIA[id] = na_offset;
	if (id == N-1) NIA[id+1] = na_offset+num_neighbors;
	propertyV[id] = id;
}

__global__ void graph_kernel_dp (int vertex, int *NA, int *propertyE, int *propertyV, long long start, long long end)
{
        long long i;
        int id = threadIdx.x+blockIdx.x*blockDim.x;

        for (i=start+id*(end-start)/(blockDim.x*gridDim.x); i<start+(id+1)*(end-start)/(blockDim.x*gridDim.x); i++) propertyE[i] = propertyV[NA[i]]-propertyV[vertex];
}

__global__ void graph_kernel (int *NA, long long *NIA, int *propertyV, int*propertyE)
{
	long long i;
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	if (NIA[id+1] - NIA[id] >= THRESHOLD) {
                graph_kernel_dp<<<2,32>>>(id, NA, propertyE, propertyV, NIA[id], NIA[id+1]);
                cudaDeviceSynchronize();
        }
        else {
		for (i=NIA[id]; i<NIA[id+1]; i++) propertyE[i] = propertyV[NA[i]]-propertyV[id];
	}
}	

int main (int argc, char *argv[])
{
	int *NA, *propertyV, *propertyE;
	long long *NIA;
	long long num_edges=0, i, sum=0;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;

	for (i=0; i<N; i++) num_edges += (i ? i : 1);

	cudaMallocManaged((void**)&NA, sizeof(int)*num_edges);
	cudaMallocManaged((void**)&NIA, sizeof(long long)*(N+1));
	cudaMallocManaged((void**)&propertyV, sizeof(int)*N);
	cudaMallocManaged((void**)&propertyE, sizeof(int)*num_edges);

        graph_gen<<<N/1024, 1024>>>(NA, NIA, propertyV);
	cudaDeviceSynchronize();

	gettimeofday(&tv0, &tz0);
	
	graph_kernel<<<N/1024, 1024>>>(NA, NIA, propertyV, propertyE);
	cudaDeviceSynchronize();

	gettimeofday(&tv2, &tz2);

	for (i=0; i<num_edges; i++) sum += propertyE[i];
	
	printf("Sum: %lld, Time: %ld microseconds\n", sum, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
	return 0;
}

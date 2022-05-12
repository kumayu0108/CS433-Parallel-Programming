#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>




#define TOL 1e-5
#define ITER_LIMIT 1000
#define WARPSIZE 32

__managed__ int rspan, cspan;

__managed__ float total_err;

__global__ void grid_solver(float *a, int n) {
    
    int r=1+blockIdx.x*blockDim.x*rspan + rspan*threadIdx.x;
    int c=1+blockIdx.y*blockDim.y*cspan + cspan*threadIdx.y;

    

    float local_diff=0.0;



    for(int i=r;i<r+rspan;i++) {
        for(int j=c;j<c+cspan;j++) {
            float temp=a[i*(n+2)+j];
            a[i*(n+2)+j]=0.2*(a[i*(n+2)+j]+a[(i-1)*(n+2)+j]+a[(i+1)*(n+2)+j]+a[i*(n+2)+j-1]+a[i*(n+2)+j+1]);
            local_diff+=fabsf(temp-a[i*(n+2)+j]);
        }
    }


    unsigned mask = 0xffffffff;

    

	for (int i=warpSize/2; i>0; i=i/2) local_diff += __shfl_xor_sync(mask, local_diff, i);


    


	if (threadIdx.x == 0 && threadIdx.y==0) {
		atomicAdd(&total_err, local_diff);
	}

    //atomicAdd(&total_err, local_diff);
}


__global__ void init_kernel(float *a, int n, curandState *states) {             //This will check for edge blocks and initialize them along with the internal blocks
    
    int r=1+blockIdx.x*blockDim.x*rspan + rspan*threadIdx.x;
    int c=1+blockIdx.y*blockDim.y*cspan + cspan*threadIdx.y;

    int id=threadIdx.y+threadIdx.x*blockDim.y;
    int block_id=blockIdx.y+blockIdx.x*gridDim.y;

    
    //I assume (id*5)+(block_id*7) is a sufficiently random seed for each thread. this can be made entirely random but i will not spend time with this.

    curand_init((id*5)+(block_id*7), id, 0, &states[id]);


    for(int i=r;i<r+rspan;i++) {
        for(int j=c;j<c+cspan;j++) {
            a[i*(n+2)+j]=curand_uniform(&states[id]);
        }
    }
    //edge cases:
    
    if(r==1) {
        for(int j=c;j<c+cspan;j++)
            a[j]=curand_uniform(&states[id]);
        if(c==1) {
            a[0]=curand_uniform(&states[id]);
            a[n+1]=curand_uniform(&states[id]);
        }
    }
    if(r+rspan==n+1) {
        for(int j=c;j<c+cspan;j++)
            a[(n+1)*(n+2)+j]=curand_uniform(&states[id]);

        if(c==1) {
            a[(n+1)*(n+2)]=curand_uniform(&states[id]);
            a[(n+2)*(n+2)-1]=curand_uniform(&states[id]);
        }
    }

    if(c==1) {
        for(int i=r;i<r+rspan;i++) {
            a[i*(n+2)]=curand_uniform(&states[id]);
        }
    }
    if(c+cspan==n+1) {
        for(int i=r;i<r+rspan;i++) {
            a[(i+1)*(n+2)-1]=curand_uniform(&states[id]);
        }
    }
}



int main(int argc, char *argv[]) {
    int n, t;
    int elem_per_thread;

    struct timeval tv0, tv2;
	struct timezone tz0, tz2;

    dim3 grid_dims, block_dims;

    float *a;

    if(argc!=3) {
        printf("Please give n and t, in that order\n");
        exit(0);
    }
    n=atoi(argv[1]);
    t=atoi(argv[2]);

    

    cudaMallocManaged((void**)&a, sizeof(float)*(n+2)*(n+2));

    int device=-1;
    cudaGetDevice(&device);

    cudaMemAdvise(a, sizeof(float)*(n+2)*(n+2), cudaMemAdviseSetPreferredLocation, device);


    block_dims.x=8;
    block_dims.y=WARPSIZE/block_dims.x;

                  

    elem_per_thread=n*n/t;

    
    cspan=1;                    
    rspan=elem_per_thread;
    while(cspan<rspan) {                                    //making it approximately a square for efficiency, preferring more columns than more rows
        cspan*=2;
        rspan/=2;
    }
    
    grid_dims.x=n/(block_dims.x*rspan);
    grid_dims.y=n/(block_dims.y*cspan);

    //This is for random number generator in the cuda kernel
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, t*sizeof(curandState));





    init_kernel<<<grid_dims, block_dims>>>(a, n, dev_random);

    cudaDeviceSynchronize();


    gettimeofday(&tv0, &tz0);

    int count=0;
    while(count !=ITER_LIMIT) {
        total_err=0;
        grid_solver<<<grid_dims, block_dims>>>(a, n);
        count++;
        cudaDeviceSynchronize();
        total_err/=(n*n);
        if(total_err<TOL)
            break;
    }

    gettimeofday(&tv2, &tz2);  
    

    
    /*
    for(int i=0;i<n+2;i++) {
        for(int j=0;j<n+2;j++) {
            printf("%.3f ",a[i*(n+2)+j]);
        }
        printf("\n");
    }*/
    

	
	printf("No. of Iterations: %d | Error: %.8f | time: %ld microseconds\n", count, total_err, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
	

    return 0;


}
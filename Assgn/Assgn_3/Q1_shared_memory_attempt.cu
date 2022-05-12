#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
//#include <cooperative_groups/scan.h>

namespace cg = cooperative_groups;


#define TOL 1e-5
#define ITER_LIMIT 1000
#define WARPSIZE 32


__managed__ int rspan, cspan;

__managed__ int shared_sizex, shared_sizey;

__managed__ float total_err=0.0;

__managed__ int count=0, done=0;

__global__ void grid_solver(float *a, int n) {
    //do
    int r=1+blockIdx.x*blockDim.x*rspan + rspan*threadIdx.x;
    int c=1+blockIdx.y*blockDim.y*cspan + cspan*threadIdx.y;

    

    float local_diff=0.0;
    int p, q, temp, i, j;
    unsigned mask = 0xffffffff;

    extern __shared__ float a_loc[];
            printf("%.3f \n",9.999);


    for(i=0;i<rspan+2;i++) {
        for(j=0;j<cspan+2;j++) {
            a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j]=a[(i+r-1)*(n+2)+(j+c-1)];
            //All threads together load the entire block's share memory(edge cases are taken care of by loading 2 extra dimensions in x and y. some extra loads, but only once)
            
        }
        //printf("\n");
    }
                //printf("%.3f \n",8.888);

    __syncthreads();
    
    /*
if(r==1 && c==1) {
    for(i=0;i<rspan+2;i++) {
        for(j=0;j<cspan+2;j++) {
            printf("%.2f", a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j]);
        }
        printf("\n");
    }
}*/
    

    //Shared memory is set up and loaded for each thread
        //printf("%.3f \n",1.429);

    

    //for synchronize purposes
    cg::grid_group grid = cg::this_grid();

    done=0;
    //printf("%.3f\n",a_loc[blockIdx.x*rspan]);


    while (!done)
    {
        local_diff = 0;


        if(r == 1 && c == 1)
        {
            //printf("%f\n", total_err);
            total_err = 0;

        }
        //printf("%d %d\n", r, c);

        grid.sync();

        //printf("%d %d\n", r*10, c*10);

        //This will operate strictly on the interior, not on the boundary
        for(i=1;i<=rspan;i++) {
            for(j=1;j<=cspan;j++) {
                p=threadIdx.x*rspan+i;
                q=threadIdx.y*cspan+j;
//printf("%d %d\n", p, q);

                temp=a_loc[p*shared_sizex+q];
                a_loc[p*shared_sizex+q]=0.2*(a_loc[p*shared_sizex+q]+a_loc[(p+1)*shared_sizex+q]+a_loc[(p-1)*shared_sizex+q]+a_loc[p*shared_sizex+q+1]+a_loc[p*shared_sizex+q-1]);
                local_diff+=fabsf(temp-a_loc[p*shared_sizex+q]);
            }
            
        }
//printf("%.3f", local_diff);

        for (i=warpSize/2; i>0; i=i/2) local_diff += __shfl_xor_sync(mask, local_diff, i);
        __syncthreads();

//printf("%.3f", local_diff);
        if (threadIdx.x == 0 && threadIdx.y==0) {
            atomicAdd(&total_err, local_diff);
        }
        if(r==1 && c==1)                //Only 1 thread does
            count++;
        //printf("[iter: %5d] diff: %6f, local: %6f\n", iter, diff/(n * n), local_diff);
        
        grid.sync();

        //1 iteration done
        
        
        if((total_err / (n*n)) < TOL || (count == ITER_LIMIT)){
            done = 1;
        }



        for(i=1;i<=rspan;i++) {
            j=1;
            a[(i+r-1)*(n+2)+(j+c-1)]=a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j];

            j=cspan;
            a[(i+r-1)*(n+2)+(j+c-1)]=a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j];
        }
        for(j=1;j<=cspan;j++) {
            i=1;
            a[(i+r-1)*(n+2)+(j+c-1)]=a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j];

            i=rspan;
            a[(i+r-1)*(n+2)+(j+c-1)]=a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j];
        }      

        


        grid.sync();

        

        //Copy new boundaries to a_loc
        for(i=0;i<=rspan+1;i++) {
            j=0;
            a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j]=a[(i+r-1)*(n+2)+(j+c-1)];

            j=cspan+1;
            a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j]=a[(i+r-1)*(n+2)+(j+c-1)];
        }
        for(j=0;j<=cspan+1;j++) {
            i=0;
            a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j]=a[(i+r-1)*(n+2)+(j+c-1)];

            i=rspan+1;
            a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j]=a[(i+r-1)*(n+2)+(j+c-1)];
        }
                


        grid.sync();

        //Some useless coypying will be done the above part can be further optimized
        // to only replace the edges of blocks only
    }
    //Now copy a_loc back to a

    for(i=1;i<=rspan;i++) {
        for(j=1;j<=cspan;j++) {
            a[(i+r-1)*(n+2)+(j+c-1)]=a_loc[(threadIdx.x*rspan+i)*shared_sizex+threadIdx.y*cspan+j];
        }
    }
    
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


    //gettimeofday(&tv0, &tz0);

    init_kernel<<<grid_dims, block_dims>>>(a, n, dev_random);

    cudaDeviceSynchronize();

    //gettimeofday(&tv2, &tz2);

    shared_sizex=block_dims.x*rspan+2;
    shared_sizey=block_dims.y*cspan+2;


    int supportsCoopLaunch = 0;
	cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
	if (supportsCoopLaunch) {
		void *kernelArgs[] = {(void*)&a, (void*)&n};

		gettimeofday(&tv0, &tz0);
		cudaLaunchCooperativeKernel((void*)grid_solver, grid_dims, block_dims, kernelArgs, sizeof(float)*shared_sizex*shared_sizey);
	}

/*
    gettimeofday(&tv0, &tz0);

    grid_solver<<<grid_dims, block_dims, sizeof(float)*shared_sizex*shared_sizey>>>(a, n);
*/
    cudaDeviceSynchronize();

    gettimeofday(&tv2, &tz2); 

    

    /*
    for(int i=0;i<n+2;i++) {                                //for debugging
        for(int j=0;j<n+2;j++) {
            printf("%.3f ",a[i*(n+2)+j]);
        }
        printf("\n");
    }
    */

	
	printf("No. of Iterations: %d | Error: %.8f | time: %ld microseconds\n", count, total_err, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
	

    return 0;


}
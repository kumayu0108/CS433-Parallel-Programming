#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <time.h>

int main(int argc, char *argv[]){
    if (argc != 3) {
		printf ("Need size of matrix and number of threads.\n");
		exit(1);
	}
    int N = atoi(argv[1]);
    int numThr = atoi(argv[2]);

	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

    float *a = (float *)calloc(N*N, sizeof(float));
    float *b = (float *)calloc(N, sizeof(float));
    float *c = (float *)calloc(N, sizeof(float));
    for(int i = 0; i < N*N; i++){a[i] = (float)i/(N*N);}
    for(int i = 0; i < N; i++){b[i] = (float)i/(N*N);}

    // for(int i = 0; i < 16; i++){printf("%f ", a[i]);}printf("\n");
    // for(int i = 0; i < 4; i++){printf("%f ", b[i]);}printf("\n");

	gettimeofday(&tv0, &tz0);
#pragma omp parallel num_threads(numThr)
    {
        int y,i;
        float* results_private = (float*)calloc(N, sizeof(float));
        for(y = 0; y < N ; y++) {
            #pragma omp for
            for(i = 0; i < N; i++) {
                results_private[y] += b[i]*a[y*N + i];   
            }
        }
        #pragma omp critical
        {
            for(y=0; y<N; y++) c[y] += results_private[y];
        }
        // for(int i = 0; i < 4; i++){printf("%f ", results_private[i]);}printf("\n");
        free(results_private);
    }
	gettimeofday(&tv1, &tz1);

    srand(time(0));
	int rowC = random() % N;
    float x = 0;
	for (int i=0; i<N; i++) x += a[rowC*N + i]*b[i];
	float error = fabs(c[rowC] - x);
	printf("Error: %0.12f, computed value: %0.12f, actual value: %0.12f, time: %ld microseconds\n", error, c[rowC], x, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));

}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <time.h>
#include <pthread.h>

int N, numThr;
float *a, *b, *c;

void *solver(void *param){
    int id = *(int*)(param);
    float sm = 0;
    for(int i = id*(N/numThr); i < (id+1)*(N/numThr); i++){
        sm = 0;
        for(int j = 0; j < N; j++){
            sm += a[i*N+j]*b[j];
        }
        c[i] = sm;
    }
}

int main(int argc, char *argv[]){
    if (argc != 3) {
		printf ("Need size of matrix and number of threads.\n");
		exit(1);
	}
    N = atoi(argv[1]);
    numThr = atoi(argv[2]);

	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

    a = (float *)calloc(N*N, sizeof(float));
    b = (float *)calloc(N, sizeof(float));
    c = (float *)calloc(N, sizeof(float));
    for(int i = 0; i < N*N; i++){a[i] = (float)i/(N*N);}
    for(int i = 0; i < N; i++){b[i] = (float)i/(N*N);}

    // for(int i = 0; i < 16; i++){printf("%f ", a[i]);}printf("\n");
    // for(int i = 0; i < 4; i++){printf("%f ", b[i]);}printf("\n");
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    int *id = (int*)malloc(numThr*sizeof(int));
 	for (int i=0; i<numThr; i++) id[i] = i;
    pthread_t *tid = (pthread_t*)malloc(numThr*sizeof(pthread_t));

	gettimeofday(&tv0, &tz0);
	for (int i=1; i<numThr; i++) {
		pthread_create(&tid[i], &attr, solver, &id[i]);
   	}

    float sm = 0;
    for(int i = 0; i < (N/numThr); i++){
        sm = 0;
        for(int j = 0; j < N; j++){
            sm += a[i*N+j]*b[j];
        }
        c[i] = sm;
    }

    for (int i=1; i<numThr; i++) {
		pthread_join(tid[i], NULL);
	}

	gettimeofday(&tv1, &tz1);

    srand(time(0));
	int rowC = random() % N;
    float x = 0;
	for (int i=0; i<N; i++) x += a[rowC*N + i]*b[i];
	float error = fabs(c[rowC] - x);
	printf("Error: %0.12f, computed value: %0.12f, actual value: %0.12f, time: %ld microseconds\n", error, c[rowC], x, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));

}
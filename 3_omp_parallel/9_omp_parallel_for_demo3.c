#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>

#define N (1<<15)

float **a;
int nthreads;

int main (int argc, char *argv[])
{
	int i, j, pid;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);

	a = (float**)malloc(sizeof(float*)*N);
	assert(a != NULL);
	for (i=0; i<N; i++) {
		a[i] = (float*)malloc(sizeof(float)*(i+1));
		for (j=0; j<i+1; j++) a[i][j] = 1.0;
	}

	gettimeofday(&tv0, &tz0);

#pragma omp parallel for num_threads (nthreads) private(j) schedule(dynamic,2)
	for (i=0; i<N; i++) {
	   for (j=0; j<i; j++) a[i][j] = sin(a[i][j]);
 	}

	gettimeofday(&tv1, &tz1);

	printf("Random element: %f, time: %ld microseconds\n", a[random()%N][random()%N], (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}

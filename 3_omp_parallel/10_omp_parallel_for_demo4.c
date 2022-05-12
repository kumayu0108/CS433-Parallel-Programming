#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>

#define N (1<<15)

int main (int argc, char *argv[])
{
	int i, j, nthreads, low, high;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;
	int *a, *b;
	float **c;

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);

	a = (int*)malloc(sizeof(int)*N);
	assert(a != NULL);
	for (i=0; i<N; i++) {
		a[i] = random() % N;
	}

	b = (int*)malloc(sizeof(int)*N);
        assert(b != NULL);
	for (i=0; i<N; i++) {
		b[i] = a[i] + (random()%N);
		if (b[i] >= N) b[i] = N-1;
        }

	c = (float**)malloc(sizeof(float*)*N);
        assert(c != NULL);
        for (i=0; i<N; i++) {
		c[i] = (float*)malloc(sizeof(float)*N);
		assert(c[i] != NULL);
		for (j=0; j<N; j++) c[i][j] = 0;
	}

	gettimeofday(&tv0, &tz0);

#pragma omp parallel for num_threads(nthreads) private(j, low, high)
	for (i=0; i<N; i++) {
		low = a[i];
		high = b[i];
		if (low > high) break;
		for (j=low; j<=high; j++) c[i][j] = sin(((float)a[i])/b[i]);
	}

	gettimeofday(&tv1, &tz1);

	float sum = 0;
	for (i=0; i<N; i++) for (j=0; j<N; j++) sum += c[i][j];

	printf("Sum: %f, time: %ld microseconds\n", sum, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}

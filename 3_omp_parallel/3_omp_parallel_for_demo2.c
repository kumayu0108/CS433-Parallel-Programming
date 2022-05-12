#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>

#define N 100000000

int main (int argc, char *argv[])
{
	double area = 0.0, pi, x;
	int nthreads;
	int i, j;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);

	gettimeofday(&tv0, &tz0);

#pragma omp parallel for num_threads (nthreads)
        for (i=0; i<N; i++) {
	   x = (i + 0.5)/N;
#pragma omp critical
	   area += 4.0/(1.0 + x*x);
	}

	pi = area/N;
   	
	gettimeofday(&tv1, &tz1);

	printf("pi = %.15lf, Time: %ld microseconds\n", pi, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}
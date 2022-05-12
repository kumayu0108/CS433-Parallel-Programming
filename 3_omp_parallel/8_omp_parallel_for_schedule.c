#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>

int main (int argc, char *argv[])
{
	int i, pid;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;
	int nthreads;
	float a[32];

	if (argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	nthreads = atoi(argv[1]);

	for (i=0; i<32; i++) a[i] = 1.0;

	gettimeofday(&tv0, &tz0);

#pragma omp parallel for schedule (guided, 2) num_threads (nthreads)
	for (i=0; i<32; i++) {
#pragma omp critical
	   printf("id: %d, index: %d\n", omp_get_thread_num(), i);

	   a[i] = sin(a[i]);
 	}

	gettimeofday(&tv1, &tz1);

	printf("Random element: %f, time: %ld microseconds\n", a[random()%32], (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	return 0;
}

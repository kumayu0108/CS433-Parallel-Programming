#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#define SIZE (1<<30)

#define PROG_INPUT 0
#define PARTIAL_SUM 1

int main(int argc, char ** argv)
{
	double sum = 0;
	float *a;
	int nprocs, myid, i;
	double private_sum = 0;

	struct timeval tv0, tv1;
        struct timezone tz0, tz1;

	char hostname[256];
	gethostname(hostname, 256);

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	if (myid == 0) {
		a = (float*)malloc(sizeof(float)*SIZE);
       	 	assert(a != NULL);
        	for (i=0; i<SIZE; i++) a[i] = 1;

		gettimeofday(&tv0, &tz0);

		for (i=1; i<nprocs; i++) {
			MPI_Send(&a[(SIZE/nprocs)*i], SIZE/nprocs, MPI_FLOAT, i, PROG_INPUT, MPI_COMM_WORLD);
		}
	}
	else {
		a = (float*)malloc(sizeof(float)*SIZE/nprocs);
		assert(a != NULL);

		MPI_Recv(a, SIZE/nprocs, MPI_FLOAT, 0, PROG_INPUT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	for (i=0; i<SIZE/nprocs; i++) private_sum += (a[i]*a[i]);
	printf("%s: I am proc %d. My partial sum is %lf.\n", hostname, myid, private_sum);

	if (myid == 0) {
		sum = private_sum;

		for (i=1; i<nprocs; i++) {
			MPI_Recv(&private_sum, 1, MPI_DOUBLE, i, PARTIAL_SUM, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sum += private_sum;
		}

		gettimeofday(&tv1, &tz1);

		printf("%s: I am proc %d. Total sum is %lf. Time: %ld microseconds.\n", hostname, myid, sum, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	}
	else {
		MPI_Send(&private_sum, 1, MPI_DOUBLE, 0, PARTIAL_SUM, MPI_COMM_WORLD);
	}
	
	MPI_Finalize();
	return 0;
}
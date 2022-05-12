#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <math.h>

#define TOL 1e-10
#define ITER_LIMIT 1000

void InitializeA (int n, float *X)
{
   for (int i=0; i<n; i++) {
      for (int j=0; j<n; j++) {
	 if (i == j) X[i*n+j] = 0;
	 else X[i*n+j] = -((float)(random() % 100)/1000000.0);
      }
   }
}

void InitializeB (int n, float *X)
{
   for (int i=0; i<n; i++) X[i] = ((float)(random() % 100)/1.0);
}

int main(int argc, char ** argv)
{
   int pid, P, done, n, iters = 0;
   float local_diff, total_diff, *A_in, *b_in, *x, *A, *b, *next_x; 
   
   FILE *fp;
   char buffer[64];

   struct timeval tv0, tv1;
   struct timezone tz0, tz1;

   MPI_Init(&argc, &argv);

   MPI_Comm_rank(MPI_COMM_WORLD, &pid);
   MPI_Comm_size(MPI_COMM_WORLD, &P);

   n = atoi(argv[1]);
   done = 0;

   if (!pid) {
      A_in = (float *) malloc (n * n * sizeof(float));
      InitializeA(n, A_in);

      b_in = (float *) malloc (n * sizeof(float));
      InitializeB(n, b_in);
   }
   A = (float *) malloc (((n * n)/P) * sizeof(float));
   b = (float *) malloc ((n/P) * sizeof(float));
   x = (float *) malloc (n * sizeof(float));
   if (!pid) InitializeB(n, x);
   next_x = (float *) malloc (n/P * sizeof(float));

   MPI_Bcast (x, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Scatter (A_in, (n * n)/P, MPI_FLOAT, A, (n * n)/P, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Scatter (b_in, n/P, MPI_FLOAT, b, n/P, MPI_FLOAT, 0, MPI_COMM_WORLD);

   if (!pid) {
      sprintf(buffer, "mpi_iterative_equation_solver_outfile_%d_%d.txt", n, P);
      fp = fopen(buffer, "w");
      gettimeofday(&tv0, &tz0);
   }

   while (!done) {
      local_diff = 0.0;
      for (int i=0; i<n/P; i++) {
	 next_x[i] = b[i];
         for (int j=0; j<n; j++) {
	    next_x[i] += (A[i*n+j]*x[j]);
	 }
	 local_diff += fabs(next_x[i] - x[i+(pid*(n/P))]);
      }
      MPI_Allreduce(&local_diff, &total_diff, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      iters++;
      if ((total_diff/n < TOL) ||
          (iters == ITER_LIMIT)) done = 1;
      if (!pid) fprintf(fp, "[%d] diff = %.15f\n", iters, total_diff/n);
      MPI_Allgather (next_x, n/P, MPI_FLOAT, x, n/P, MPI_FLOAT, MPI_COMM_WORLD);
   }
   if (!pid) {
      gettimeofday(&tv1, &tz1);
      fclose(fp);
      printf("Time: %ld microseconds.\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
   }
   MPI_Finalize();
   return 0;
}

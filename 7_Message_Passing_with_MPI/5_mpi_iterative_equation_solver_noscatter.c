#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <math.h>

#define TOL 1e-10
#define ITER_LIMIT 1000

void InitializeA (int r, int c, float *X, int pid)
{
   for (int i=0; i<r; i++) {
      for (int j=0; j<c; j++) {
	 if ((pid*r)+i == j) X[i*c+j] = 0;
	 else X[i*c+j] = -((float)((i*c+j+pid*r*c) % 100)/1000000.0);
      }
   }
}

void InitializeB (int n, float *X, int pid)
{
   for (int i=0; i<n; i++) X[i] = ((float)((i+pid*n) % 100)/1.0);
}

int main(int argc, char ** argv)
{
   int pid, P, done, n, iters = 0;
   float local_diff, total_diff, *x, *A, *b, *next_x; 
   
   FILE *fp;
   char buffer[64];

   struct timeval tv0, tv1;
   struct timezone tz0, tz1;

   MPI_Init(&argc, &argv);

   MPI_Comm_rank(MPI_COMM_WORLD, &pid);
   MPI_Comm_size(MPI_COMM_WORLD, &P);

   n = atoi(argv[1]);
   done = 0;

   x = (float *) malloc (n * sizeof(float));
   if (!pid) InitializeB(n, x, pid);

   MPI_Bcast (x, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

   A = (float *) malloc (((n * n)/P) * sizeof(float));
   InitializeA(n/P, n, A, pid);

   b = (float *) malloc ((n/P) * sizeof(float));
   InitializeB(n/P, b, pid);

   next_x = (float *) malloc ((n/P) * sizeof(float));

   if (!pid) {
      sprintf(buffer, "mpi_iterative_equation_solver_noscatter_outfile_%d_%d.txt", n, P);
      fp = fopen(buffer, "w");
   }

   MPI_Barrier(MPI_COMM_WORLD);
   if (!pid) gettimeofday(&tv0, &tz0);

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

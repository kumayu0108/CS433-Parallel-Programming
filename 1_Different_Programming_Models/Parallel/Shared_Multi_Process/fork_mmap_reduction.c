#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#define SIZE (1<<30)
#define SEMAPHORE_KEY 19

typedef struct shared_mem_s {
   double global_sum;
   int semaphore_id;
} shared_mem_t;

void solver (int id, double *private_sum, int num_threads, float *a, shared_mem_t *shmem)
{
   int i;
   struct sembuf Poplist, Voplist;

   (*private_sum) = 0;

   for (i=(SIZE/num_threads)*id; i<(SIZE/num_threads)*(id+1); i++) {
      (*private_sum) += (a[i]*a[i]);
   }
   
   Poplist.sem_num = 0;
   Poplist.sem_op = -1;
   Poplist.sem_flg = SEM_UNDO;

   Voplist.sem_num = 0;
   Voplist.sem_op = 1;
   Voplist.sem_flg = SEM_UNDO;

   semop(shmem->semaphore_id, &Poplist, 1);
   shmem->global_sum += (*private_sum);
   semop(shmem->semaphore_id, &Voplist, 1);
}


int main (int argc, char *argv[])
{
	int i, mmap_fd, num_threads;
	float *a;
	double private_sum;
	shared_mem_t *shmem;
	short initsem = 1;
        struct sembuf Poplist, Voplist;

        struct timeval tv0, tv1;
        struct timezone tz0, tz1;

        if (argc != 2) {
           printf("Need number of threads!\n");
           exit(1);
        }
        num_threads = atoi(argv[1]);

	a = (float*)malloc(sizeof(float)*SIZE);
	assert(a != NULL);
	for (i=0; i<SIZE; i++) a[i] = 1;

        if (num_threads > 1) {
           mmap_fd = open("/dev/zero", O_RDWR);
    	   if (mmap_fd == -1) {
 		printf("Cannot open /dev/zero!\nAborting...\n");
		exit(1);
	   } 

	   shmem = (shared_mem_t*)mmap(NULL, sizeof(shared_mem_t), PROT_READ|PROT_WRITE, MAP_SHARED, mmap_fd, 0);
	   if (shmem == MAP_FAILED) {
		printf("mmap failed!\nAborting...\n");
		exit(1);
	   }

	   shmem->semaphore_id = semget(SEMAPHORE_KEY, 1, 0777|IPC_CREAT);
           semctl(shmem->semaphore_id, 1, SETALL, &initsem);

	   gettimeofday(&tv0, &tz0);
	   shmem->global_sum = 0;
	   for (i=1; i<num_threads; i++) {
	   	if (fork() == 0) {
		   // This is child
		   solver(i, &private_sum, num_threads, a, shmem);
		   break;
		}
	   }
	   if (i == num_threads) {
	      // This is parent
	      private_sum = 0;
	      for (i=0; i<SIZE/num_threads; i++) private_sum += (a[i]*a[i]);

	      Poplist.sem_num = 0;
              Poplist.sem_op = -1;
              Poplist.sem_flg = SEM_UNDO;

              Voplist.sem_num = 0;
              Voplist.sem_op = 1;
              Voplist.sem_flg = SEM_UNDO;

	      semop(shmem->semaphore_id, &Poplist, 1);
	      shmem->global_sum += private_sum;
	      semop(shmem->semaphore_id, &Voplist, 1);

	      for (i=1; i<num_threads; i++) wait(NULL);
              gettimeofday(&tv1, &tz1);
	      printf("Total sum: %lf, Time: %ld microseconds\n", shmem->global_sum, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
	      semctl(shmem->semaphore_id, 1, IPC_RMID, NULL);
              munmap(shmem, sizeof(shared_mem_t));
	   }
        }
        else {
	   gettimeofday(&tv0, &tz0);
	   private_sum = 0;
           for (i=0; i<SIZE; i++) private_sum += a[i];
	   gettimeofday(&tv1, &tz1);
           printf("Total sum: %lf, Time: %ld microseconds\n", private_sum, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
        }
	return 0;
}
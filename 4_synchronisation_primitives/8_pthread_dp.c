#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#define THINKING 0
#define HUNGRY 1
#define EATING 2

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t *cond_mutex;
pthread_cond_t  *cond;
int num_threads, num_rounds, *state;

/* The following function tests if philosopher `i' can eat.
   The logic is simple: if the neighbors of `i' are not eating AND
                        philosopher `i' is hungry, then `i' should eat.
                        In that case, philosopher `i' is woken up by
                        signaling the condition variable of `i'.
*/
void test (int i)
{
	if ((state[(i+num_threads-1)%num_threads] != EATING) &&
	    (state[i] == HUNGRY) &&
            (state[(i+1)%num_threads] != EATING)) {
           pthread_mutex_lock(&cond_mutex[i]);
	   state[i] = EATING;
	   pthread_cond_signal(&cond[i]);
	   pthread_mutex_unlock(&cond_mutex[i]);
	}
}

/* Philosopher `i' tries to pick up the chopsticks.
   She must check if she can eat by invoking test(i).
   If she cannot eat, she must wait on the condition variable of `i'.
*/
void pickup (int i)
{
	pthread_mutex_lock(&mutex);
	state[i] = HUNGRY;
	test(i);
        pthread_mutex_unlock(&mutex);
        pthread_mutex_lock(&cond_mutex[i]);
	if (state[i] != EATING) {
		pthread_cond_wait(&cond[i], &cond_mutex[i]);
	}
        pthread_mutex_unlock(&cond_mutex[i]);
}

/* Philosopher `i' puts down the chopsticks.
   She must test if her neighbors can now eat.
*/
void putdown (int i)
{
	pthread_mutex_lock(&mutex);
	state[i] = THINKING;
	test((i+num_threads-1)%num_threads);
	test((i+1)%num_threads);
	pthread_mutex_unlock(&mutex);
}

void *solver (void *param)
{
	int i, id = *(int*)(param);
	struct timeval tv0, tv1;
        struct timezone tz0, tz1;
	
	gettimeofday(&tv0, &tz0);	
	for (i=0; i<num_rounds; i++) {
		pickup(id);
		printf("Thread %d eating!\n", id);
		sleep(1);
		putdown(id);
		printf("Thread %d thinking!\n", id);
		sleep(1);
	}
	gettimeofday(&tv1, &tz1);
        printf("Time from %d: %ld microseconds\n", id, (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
}	

int main (int argc, char *argv[])
{
	int i, *id;
	pthread_t *tid;
	pthread_attr_t attr;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	if (argc != 3) {
		printf ("Need number of threads and number of rounds.\n");
		exit(1);
	}
	num_threads = atoi(argv[1]);
	num_rounds = atoi(argv[2]);
	tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
	id = (int*)malloc(num_threads*sizeof(int));
 	for (i=0; i<num_threads; i++) id[i] = i;

	state = (int*)malloc(num_threads*sizeof(int));
	for (i=0; i<num_threads; i++) state[i] = THINKING;

	cond_mutex = (pthread_mutex_t*)malloc(num_threads*sizeof(pthread_mutex_t));
	for (i=0; i<num_threads; i++) {
		pthread_mutex_init(&cond_mutex[i], NULL);
	}

	cond = (pthread_cond_t*)malloc(num_threads*sizeof(pthread_cond_t));
	for (i=0; i<num_threads; i++) {
		pthread_cond_init(&cond[i], NULL);
	}

	pthread_attr_init(&attr);

	for (i=1; i<num_threads; i++) {
		pthread_create(&tid[i], &attr, solver, &id[i]);
   	}

	gettimeofday(&tv0, &tz0);
	for (i=0; i<num_rounds; i++) {
		pickup(0);
		printf("Thread 0 eating!\n");
		sleep(1);
		putdown(0);
		printf("Thread 0 thinking!\n");
		sleep(1);
	}
	gettimeofday(&tv1, &tz1);
	printf("Time from 0: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));

	for (i=1; i<num_threads; i++) {
		pthread_join(tid[i], NULL);
	}
	return 0;
}

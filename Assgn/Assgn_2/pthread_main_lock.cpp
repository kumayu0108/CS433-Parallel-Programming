#include<iostream>
#include<stdio.h>
#include<algorithm>
#include<assert.h>
#include<pthread.h>
#include<vector>
#include <sys/time.h>
#include"sync_library.h"

using namespace std;

#define N 1e7
int x = 0, y = 0;
int lock = 0, numThr;

volatile int choosing[64*16*32];
volatile int ticket[64*16*32];

volatile int tkkt = 0, rls_cnt = 0;

void *solver(void *param){

    void *ret;
    int id = *(int *)param;

    for(int i = 0; i < N; i++){
        // Acquire_ticket(&tkkt, &rls_cnt);
        // Acquire_tts(&lock);
        // Acquire_xchg(&lock);
        Acquire_Bake(id, numThr, choosing, ticket);
        assert(x == y);
        x = y + 1;
        y++;
        Release_Bake(id, ticket);
        // Release_xchg(&lock);
        // Release_tts(&lock);
        // Release_ticket(&rls_cnt);
    }

    return ret;
}

int main(int argc, char *argv[]){

    pthread_t *tid;
	pthread_attr_t attr;

	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

    numThr = atoi(argv[1]);

    int *id = (int*)malloc(numThr*sizeof(int));
    for (int i=0; i<numThr; i++) {id[i] = i; choosing[i * __cacheBlocksJump] = false; ticket[i*__cacheBlocksJump] = 0;}
    tid = (pthread_t*)malloc(numThr*sizeof(pthread_t));
    pthread_attr_init(&attr);

    gettimeofday(&tv0, &tz0);
    for (int i=1; i<numThr; i++) {
		pthread_create(&tid[i], &attr, solver, &id[i]);
   	}
    solver(&id[0]);

    for (int i=1; i<numThr; i++) {
		pthread_join(tid[i], NULL);
	}
    gettimeofday(&tv1, &tz1);

    assert(x == y);
    assert(x == N*numThr);
    // cout<<x;

    printf("time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
    return 0;
}
int __cacheBlocksJump = 8;

unsigned char CompareAndSet(int oldVal, int newVal, int *ptr) {
    int oldValOut;
    unsigned char result;
    asm("lock cmpxchgl %4, %1 \n setzb %0"
                :"=qm"(result),  "+m" (*ptr), "=a" (oldValOut)
                :"a" (oldVal),  "r" (newVal)
                : );
    return result;
}

void Acquire_Bake(int id, int numThr, volatile int *chs, volatile int *tkt){
    chs[id*16*__cacheBlocksJump] = 1;
    asm("mfence":::"memory");
    int mx = 0;
    for(int i = 0; i < numThr; i++){mx = mx > tkt[i*16*__cacheBlocksJump] ? mx : tkt[i*16*__cacheBlocksJump];}
    tkt[id*16*__cacheBlocksJump] = mx + 1;
    asm("mfence":::"memory");
    chs[id*16*__cacheBlocksJump] = 0;
    asm("mfence":::"memory");

    for(int j = 0; j < numThr; j++){
        while(chs[j*16*__cacheBlocksJump]);
        // cout<<".."<<id<<"..\n";
        while(tkt[j*16*__cacheBlocksJump] != 0 && ((tkt[j*16*__cacheBlocksJump] < tkt[id*16*__cacheBlocksJump]) || (tkt[j*16*__cacheBlocksJump] == tkt[id*16*__cacheBlocksJump] && j < id)));
    }
}

void Release_Bake(int id, volatile int *tkt){
    // asm("":::"memory");
    asm("mfence":::"memory");
    tkt[id*16*__cacheBlocksJump] = 0;
    // asm("mfence":::"memory");
}


void Acquire_xchg(int *lock){
    while (!CompareAndSet(0, 1, lock));
}

void Release_xchg(int *lock){
    asm("":::"memory");
    *lock = 0;
}

void Acquire_tts(int *lock){
    while(1){
        if(CompareAndSet(0, 1, lock))
            break;
        while((*lock));
    }
}

void Release_tts(int *lock){
    asm("":::"memory");
    *lock = 0;
}

void Acquire_ticket(volatile int *tkt, volatile int *release_count){
    unsigned char result = 0;
    int oldValOut, val;
    while(!result){
        result = 0;
        val = *tkt;
        asm("lock cmpxchgl %4, %1 \n setzb %0"
                :"=qm"(result),  "+m" (*tkt), "=a" (oldValOut)
                :"a" (val),  "r" (val + 1)
                : );
    }
    while(1){int tmp = (*release_count); if(tmp == val)break;}
}

void Release_ticket(volatile int *release_count){
    int tmp = (*release_count);
    (*release_count) = tmp + 1;
    asm("mfence":::"memory");
}
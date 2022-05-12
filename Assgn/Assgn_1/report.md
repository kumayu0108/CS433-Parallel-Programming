
## Q1
For the first question, travelling salesman problem, I used the Dynamic Programming approach to solve the problem and implemented my parallelisation using OpenMP directives.

The main equation governing the solution of dynamic programming method is :-
            dp[S][i] = min({ dp[S - {i}][j] + dist[j][i] })
S represents a subset of the set of vertices given, i represents any point within the subset (except the first point), S - {i} represents the subset after removing 'i', 
dist[j][i] stands for distance between j and i. Here dp[S][i] is meant to calculate the minimum cost tour that visits each vertex in set S once, starting at 1 and ending at 'i'.

Thus, the only dependence that exists in this equation is loop carried flow dependence, which means that we can completely parallelise the calculation for calculating dp[S][i] for a fixed sized subset.
This is what I do using the vector named subsets, where subsets[4] stores all subsets containing 1 of size 4; and so on. Then, I initialise the dp array to infinity (INT32_MAX). 
To calculate dp[S][i], I calculate a unique hash value for each subset, which is mapped to the subset's index in dp[S] ; using an unordered map. 

To parallelise the code, I divide the following code -

```
for(int i = 2; i < n; i++){
    int from = tid * (subsets[i].size()/num_thr), to = (tid + 1) * (subsets[i].size()/num_thr);
    if(tid == num_thr - 1)to = subsets[i].size();

    for(int j = from; j < to; j++){
            .....
```

The whole parallel segment of the code is -
```
#pragma omp parallel num_threads(num_thr) 
{   
    int tid = omp_get_thread_num();
    for(int i = 2; i < n; i++){ //size of subset
        int from = tid * (subsets[i].size()/num_thr), to = (tid + 1) * (subsets[i].size()/num_thr);
        if(tid == num_thr - 1)to = subsets[i].size();

        for(int j = from; j < to; j++){ //particular subset
            int S_ind = mp[calcHash(subsets[i][j])];
            for(int k = 1; k < subsets[i][j].size(); k++){
                int S_minus_ind = mp[calcHash2(subsets[i][j], k)];
                for(int l = 1; l < subsets[i][j].size(); l++){
                    if(k == l)continue;
                    if(dp[S_minus_ind][subsets[i][j][l]] + ary[subsets[i][j][l]][subsets[i][j][k]] < dp[S_ind][subsets[i][j][k]]){
                        dp[S_ind][subsets[i][j][k]] = dp[S_minus_ind][subsets[i][j][l]] + ary[subsets[i][j][l]][subsets[i][j][k]];
                        backtrack[S_ind][subsets[i][j][k]] = subsets[i][j][l];
                    }
                }
            }
        }
#pragma omp barrier
    }
}
```
I use a static allocation method to divide the second for loop and place a barrier at the end so that all threads start calculating values for next subset size together. Note - Here i is subset size and j is used to loop over subsets. Note- Using the flag -DINIT would run the program on predefined value of n, which can be changed in the program.

I have defined a vector to store the dp and backtracking values to calculate the path. Test Cases that I tried and subsequent results -
1. inp/1_inp  -  The number of vertices is 4. The time taken increases here with increasing threads due to the small size of input.
```
Number Of Threads   Time (avg) (Microseconds)
    1                   76
    2                   232
    3                   319
    4                   356
    5                   421
    6                   721
```

2. inp/1_inp2 - The number of vertices is 10. The execution time decreases until 4 threads and then almost remains constant as the overhead of parallelisation increases.
```
Number Of Threads   Time (avg) (Microseconds)
    1                   3844
    2                   2486
    3                   2201
    4                   2071
    5                   2258
    6                   2290
```

3. inp/1_inp3 - The number of vertices is 13. The time decreases on increasing number of threads, but the time becomes constant after 6 threads.
```
Number Of Threads   Time (avg) (Microseconds)
    1                   14423
    2                   13567
    3                   10613
    4                   9285
    5                   8158
    6                   7673
    7                   7728
```

4.  inp/1_inp4 - The number of vertices is 16. The time decreases on increasing number of threads and a sudden speedup is observed at thread = 6 due to cache fitting the whole data.
```
Number Of Threads   Time (avg) (Microseconds)
    1                   96466
    2                   64811
    3                   54732
    4                   44684
    5                   44395
    6                   38682
    7                   38634
```

5.  inp/1_inp5 - The number of vertices is 20. The time decreases on increasing number of threads until thread = 11, and remains constant afterwards; however the speedup obtained doesn't scale as much as it does initially on increasing threads due to increasing overheads.
```
Number Of Threads   Time (avg) (Microseconds)
    1                   1965054
    2                   1079322
    3                   912987
    4                   755417
    5                   648853
    6                   649082
    7                   620898
    8                   569808
    9                   541715
    10                  525053
    11                  496784
    12                  513637
    13                  549258
```

## Q2
Note- Using the flag -DINIT would run the program on predefined value of n, which can be changed in the program.

NOTE - Since floating point operations introduce errors which propagate to the end and may result in large errors, put the diagonal elements = 1, to ensure these error are kept minimal and right values are obtained.


 I use the concept of pipelining to parallelise the algorithm which I implement using OpenMP. For ith equation, the equation would be $\sum_j^i a_{ji} x_j = b_i$ ; here $x_i$ represents the variable that needs to be determined. Thus, assuming that the values for $x_j | j < i$ are already known, we can write $x_i$ as $x_i = \frac{b_i}{a_{ii}} - \frac{\sum^{i-1}_j a_{ji} x_j}{a_{ii}}$. Thus, writing these equations as - 
```
x_0 = b_0/a_00
x_1 = - (a_01*x_0 / a_11) + (b_1/a_11)
x_2 = - (a_02*x_0 / a_22) - (a_12*x_1 / a_22) + (b_2/a_22)
x_3 = - (a_03*x_0 / a_33) - (a_13*x_1 / a_33) - (a_23*x_2 / a_33) + (b_3/a_33)
```
Thus, it can be clearly seen that on getting the value for $x_0$, we can do the calculations - $(a_{01}*x_0 / a_{11}) , (a_{02}*x_0 / a_{22}), (a_{03}*x_0 / a_{33})$ in parallel. The main parallel section of the code is -
```
#pragma omp parallel num_threads(numThr) 
{    
    for(int i = 1; i < n; i++){
#pragma omp for 
        for(int j = i; j <= n - 1; j++){
            ans[j] = ans[j] - (ans[i-1]*(coef[j][i-1]/coef[j][j]));
        }
#pragma omp barrier
    }
}
```

Following are the results for different test cases that I tried out -

1. inp/2_inp - Value of n is 3. The time taken increases constantly because of the small sized test case and therefore the inclusion of more overheads of initialising threads. Time jumps sharply at start also due to this reason as for one thread there would be no overheads.
```
Number Of Threads   Time (avg) (Microseconds)
    1                   11
    2                   142
    3                   230
    4                   262
    5                   348
    6                   399
```

2. inp/2_inp2 - Value of n is 100. A trend similar to the first test case is observed in this test case also, where the time keeps on increasing on increasing thread count.
```
Number Of Threads   Time (avg) (Microseconds)
    1                   370
    2                   423
    3                   605
    4                   717
    5                   749
    6                   819
```

3. inp/2_inp3 - Value of n is 1000. Time decreases but after just reaching thread count of 3, the time plataeus and remains at that level.  
```
Number Of Threads   Time (avg) (Microseconds)
    1                   5384
    2                   3726
    3                   3601
    4                   3808
    5                   3940
    6                   3835
```

4. inp/2_inp4 - Value of n is 10000. Time taken decreases until 6 threads but then increases, due to increasing overheads. After that the time starts decreasing due to the whole array fitting in cache. 
```
Number Of Threads   Time (avg) (Microseconds)
    1                   735873
    2                   383536
    3                   273720
    4                   181348
    5                   137642
    6                   115995
    7                   219054
    8                   182395
    9                   161424
    10                  139610
    11                  122048
```

5. inp/2_inp5 - Value of n is 100000. Time taken decreases on increasing thread count. 
```
Number Of Threads   Time (avg) (Microseconds)
    1                   202501547
    2                   106801294
    4                   60613473
    5                   44635907
    8                   40081334
```

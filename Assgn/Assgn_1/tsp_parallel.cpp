#include<iostream>
#include<stdio.h>
#include<algorithm>
#include<vector>
#include<unordered_map>
#include<omp.h>
#include<sys/time.h>
#include<fstream>
#include<cstring> 
#include<sstream>

using namespace std;

long long p = 1e9+7, m = 31;
unordered_map <long long, long long> mp;

long long ary[100][100], n; 
vector<vector<long long>> dp, backtrack;
int num_thr = 4;

long long power(long long x, long y) { 
    long long res = 1;
    x = x % p; 
    while (y > 0) 
    { 
        if (y & 1) 
            res = (res*x) % p; 
        y = y>>1; // y = y/2 
        x = (x*x) % p; 
    } 
    return res; 
} 

long long calcHash(vector<int> &tmp){
    long long hash = 0;
    // for(int i = 0; i < tmp.size(); i++){hash = (hash * m + tmp[i]) % p; }
    for(int i = 0; i < tmp.size(); i++){hash = hash | (1<<tmp[i]);}
    return hash;
}

long long calcHash2(vector<int> &tmp, int ind){
    long long hash = 0;
    // for(int i = 0; i < tmp.size(); i++){if(i == ind)continue; hash = (hash * m + tmp[i]) % p; }
    for(int i = 0; i < tmp.size(); i++){if(i == ind)continue; hash = hash | (1<<tmp[i]); }
    return hash;
}

void rec(vector<int> tmp, vector<vector<vector<int>>> &subsets, int ind){
    if(ind == n){if(tmp.size() != 0)subsets[tmp.size()-1].push_back(tmp); return;}

    rec(tmp, subsets, ind + 1);
    tmp.push_back(ind);
    rec(tmp, subsets, ind + 1);
}


// dp(S, i) = min({ dist(j, i) + dp(S - {i}, j) })
int main(int argc, char *argv[]){

    struct timeval tv0, tv1;
	struct timezone tz0, tz1;

    string inp = argv[1];
    string outp = argv[2];
    // cout<<inp<<" "<<outp;
    num_thr = atoi(argv[3]);

    int row = 0;
    string line;

#ifndef INIT
    ifstream myfile(inp);
    if (myfile.is_open())
    {
        getline(myfile, line);
        std::stringstream huh(line);
        huh>>n;
        
        while ( getline (myfile,line) )
        {
            std::stringstream heh(line);
            int wght;
            for(int i = row + 1; i < n; i++){
                heh>>wght;
                ary[row][i] = wght;
                ary[i][row] = wght;
            }
            row++;
        }
        myfile.close();
    }
#else 
    n = 20;
    for(int i = 0; i < n; i++){
        ary[i][i] = 0;
        for(int j = i+1; j < n; j++){
            ary[i][j] = rand() % 200;
            ary[j][i] = ary[i][j];
        }
    }
#endif
    dp.resize(power(2, n), vector <long long> (n)); backtrack.resize(power(2, n), vector <long long> (n));
    gettimeofday(&tv0, &tz0);

#pragma omp parallel for num_threads(num_thr)
    for(int i = 0; i < power(2,n); i++){for(int j = 0; j < n; j++)dp[i][j] = INT32_MAX;}

    vector <vector<vector<int>>> subsets(n);
    vector <int> tmp = {0};
    rec(tmp, subsets, 1);

    int bla = 0; //mp[hash(S)] = bla; where bla is the index of dp[][] array
    for(int i = 0; i < subsets.size(); i++){
        for(int j = 0; j < subsets[i].size(); j++){mp[calcHash(subsets[i][j])] = bla++; } 
    }

    for(int j = 0; j < subsets[1].size(); j++){ //particular subset
        int S_ind = mp[calcHash(subsets[1][j])];
        dp[S_ind][subsets[1][j][1]] = ary[0][subsets[1][j][1]];
    }

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
    for(int i = 1; i < n; i++){
        if(dp[mp[calcHash(subsets[subsets.size() - 1][0])]][i] + ary[i][0] < dp[mp[calcHash(subsets[subsets.size() - 1][0])]][0]){
            dp[mp[calcHash(subsets[subsets.size() - 1][0])]][0] = dp[mp[calcHash(subsets[subsets.size() - 1][0])]][i] + ary[i][0];
            backtrack[mp[calcHash(subsets[subsets.size() - 1][0])]][0] = i;
        }
    }
    vector <int> out, tmp2;
    out.push_back(0);
    tmp = subsets[subsets.size() - 1][0];
    int ele = backtrack[mp[calcHash(subsets[subsets.size() - 1][0])]][0];
    int start = backtrack[mp[calcHash(subsets[subsets.size() - 1][0])]][ele];
    for(int i = 0; i < n-1; i++){
        out.push_back(ele);
        tmp2.clear();
        for(int j = 0; j < tmp.size(); j++){if(tmp[j] == ele)continue; tmp2.push_back(tmp[j]);}
        ele = start;
        start = backtrack[mp[calcHash(tmp2)]][start];
        tmp = tmp2;
    }
    out.push_back(0);

    gettimeofday(&tv1, &tz1);
    
    ofstream myoutfile(outp);
     if (myoutfile.is_open())
    {    
        for(int i = 0; i < out.size(); i++)myoutfile<<out[i]+1<<" ";myoutfile<<"\n";
        myoutfile<<dp[mp[calcHash(subsets[subsets.size() - 1][0])]][0];
        myoutfile.close();
    }
    
	printf("time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));

    return 0;
}
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
#define TYPE __float128

int sz = 100;

void initialiseInput(vector <vector<TYPE>> &coef, vector <TYPE> &val){
    coef.resize(sz); val.resize(sz);
    vector <TYPE> tmpAns;
    for(int i = 0; i < sz; i++){
        TYPE tmpsm = 0;
        tmpAns.push_back(i+1);
        for(int j = 0; j < i; j++){
            coef[i].push_back((rand() % 2 ? 1 : -1) * ((TYPE)(rand() % 10) + 1));
            tmpsm += coef[i].back() * tmpAns[j];
        }
        coef[i].push_back(1);
        tmpsm += tmpAns.back();
        val[i] = tmpsm;
    }
}

int main(int argc, char *argv[]){
    struct timeval tv0, tv1;
	struct timezone tz0, tz1;

    int n, numThr = atoi(argv[3]);
    vector<TYPE> val;
    vector<vector<TYPE>> coef;

#ifndef INIT
    ifstream file(argv[1]);

    if(file.is_open()){
        string line;
        getline(file, line);
        stringstream tt(line);
        tt >> n;
        coef.resize(n);
        val.resize(n);
        int row = 0;
        while(getline(file, line)){
            stringstream ss(line);
            if(row < n){
                long double tmp;
                for(int i = 0; i <= row; i++){
                    ss >> tmp;
                    coef[row].push_back(tmp);
                }
                row++;
            }
            else {
                for(int i = 0; i < n; i++){
                    long double tmp;
                    ss >> tmp;
                    val[i] = tmp;
                }
            }
        }
        file.close();
    }
#else
    n = sz;
    initialiseInput(coef, val);
#endif

    gettimeofday(&tv0, &tz0);
    vector<TYPE> ans(n, 0);

    for(int i = 0; i < n; i++){
        ans[i] = val[i]/coef[i][i];
    } 

#pragma omp parallel num_threads(numThr) 
{    
    for(int i = 1; i <= n; i++){
#pragma omp for 
        for(int j = i; j <= n - 1; j++){
            ans[j] = ans[j] - (ans[i-1]*(coef[j][i-1]/coef[j][j]));
        }
#pragma omp barrier
    }
}

    gettimeofday(&tv1, &tz1);

    ofstream myfile(argv[2]);
    if(myfile.is_open()){
        for(int i = 0; i < n; i++){
            myfile << (long double)(ans[i]) << " ";
        }
        myfile.close();
    }

    printf("time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));

    return 0;
}
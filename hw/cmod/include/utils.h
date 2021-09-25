#ifndef __UTILS_H__
#define __UITLS_H__

/* Matrix utility functions for testbench*/

#include <iostream>
#include <vector>
#include <testbench/nvhls_rand.h>

#include <random>

using namespace::std;

template<typename T>
vector<vector<T>> GetMat(int rows, int cols) {
  
  default_random_engine generator (random_device{}());
  binomial_distribution<int> distribution(255,0.5); //(0-255)
  vector<vector<T>> mat(rows, vector<T>(cols)); 
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int tmp = distribution(generator) - 128;   // -128~127
      //mat[i][j] = nvhls::get_rand<T::width>();
      mat[i][j] = tmp;
    }
  }
  return mat;
}

template<typename T>
void PrintMat(vector<vector<T>> mat) {
  int rows = (int) mat.size();
  int cols = (int) mat[0].size();
  for (int i = 0; i < rows; i++) {
    cout << "\t";
    for (int j = 0; j < cols; j++) {
      cout << mat[i][j] << "\t";
    }
    cout << endl;
  }
  cout << endl;
}

template<typename T, typename U>
vector<vector<U>> MatMul(vector<vector<T>> mat_A, vector<vector<T>> mat_B) {
  // mat_A _N*_M
  // mat_B _M*_P
  // mat_C _N*_P
  int _N = (int) mat_A.size();
  int _M = (int) mat_A[0].size();
  int _P = (int) mat_B[0].size();
  
  assert(_M == (int) mat_B.size());
  vector<vector<U>> mat_C(_N, vector<U>(_P, 0)); 

  for (int i = 0; i < _N; i++) {
    for (int j = 0; j < _P; j++) {
      mat_C[i][j] = 0;
      for (int k = 0; k < _M; k++) {
        mat_C[i][j] += mat_A[i][k]*mat_B[k][j];
      }
    }
  }
  return mat_C;
}



#endif


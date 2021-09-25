/*
 * Copyright (c) 2016-2018, Harvard University.  All rights reserved.
*/
#ifndef HELPER_H
#define HELPER_H


#include <map>
#include <vector>
#include <deque>
#include <utility>
#include <sstream>
#include <string>
#include <cstdlib>
#include <math.h> // testbench only
#include <queue>

#include <iostream>
#include <sstream>
#include <iomanip>

// helper function to set N-byte vector


// for string delimiter
std::vector<std::string> split (std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

template<int num_bytes>
NVUINTW(8*num_bytes) set_bytes(std::string str) {
  std::string d = "_";
  std::vector<std::string> str_list = split(str, d);
  NVUINT8 bytes[num_bytes];
  
  assert(str_list.size() == num_bytes);
  for (int i = 0; i < num_bytes; i++) {
    std::istringstream converter(str_list[i]);
    unsigned int value;
    converter >> std::hex >> value;
    //cout << hex << value << endl;
    bytes[num_bytes-i-1] = value;
  }
  
  NVUINTW(8*num_bytes) tmp;
  for (int i = 0; i < num_bytes; i++) {
    tmp.set_slc(8*i, bytes[i]);
  }
  //cout << hex << tmp << endl;
  return tmp;
}

float sigmoid(float x)
{
     float exp_value;
     float return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}

template<unsigned int W1, unsigned int F1>
float fixed2float(const NVINTW(W1) in) {
  float out;
  
  out = in;
  out = out / (1 << F1);
  
  return out;
}


template < typename T >
std::vector< std::vector<T> > to_2d(const unsigned long nrows, const unsigned long ncols, const std::vector<T>& flat_vec)
{
    // sanity check
    if( ncols == 0 || flat_vec.size()%ncols != 0 ) throw std::domain_error( "bad #cols" ) ;
    


    std::vector< std::vector<T> > mtx ;
    const auto begin = std::begin(flat_vec) ;
    for( std::size_t row = 0 ; row < nrows ; ++row ) mtx.push_back( { begin + row*ncols, begin + (row+1)*ncols } ) ;
    return mtx ;
}

template<typename Scalar> 
void PrintVector(const std::vector<Scalar>& data)
{
  for (unsigned int i = 0; i < data.size(); i++)
    cout << data[i] << "\t"; 
  cout << endl;
}


template<typename Scalar> 
void PrintMatrix(const std::vector<std::vector<Scalar>>& data)
{
  for (unsigned int i = 0; i < data.size(); i++)
    PrintVector(data[i]);
  cout << endl;
}


template<typename Scalar> 
std::vector<std::vector<Scalar>> TransposeMatrix(const std::vector<std::vector<Scalar>>& data)
{
  std::vector<std::vector<Scalar>> out;
  
  
  for (unsigned j = 0; j < data[0].size(); j++) {
    std::vector<Scalar> out_row;
    for (unsigned i = 0; i < data.size(); i++) {
      out_row.push_back(data[i][j]);
    }
    out.push_back(out_row);
  }
  return out;
}

template<typename Scalar> 
void PrintMatrixShape(const std::vector<std::vector<Scalar>>& data) {
  std::cout << "[" << data.size() << ", " << data[0].size() << "]" << std::endl;
  const unsigned int ncols = data[0].size();
  for (unsigned int i = 0; i < data.size(); i++) {
    if( ncols != data[i].size()) throw std::domain_error( "bad #cols" ) ;  
  
  }
}

template<typename Scalar> 
std::vector<Scalar> VectorPadding(const std::vector<Scalar>& data, const int usize)
{
  std::vector<Scalar> out = data;
  unsigned size_new = usize * (1 + (data.size()-1)/usize);
  out.resize(size_new, 0.0);
  return out;
}

template<typename Scalar> 
std::vector<std::vector<Scalar>> MatrixPadding(const std::vector<std::vector<Scalar>>& data, const int usize)
{
  std::vector<std::vector<Scalar>> out;
  unsigned size_new = usize * (1 + (data.size()-1)/usize);
  out.resize(size_new); 
  

  // for each row 
  unsigned int i;
  for (i = 0; i < data.size(); i++) {
    out[i] = VectorPadding(data[i], usize);
  }
    
  for (     ; i < out.size(); i++) {
    std::vector<Scalar> tmp;
    tmp.resize(out[0].size(), 0.0);
    out[i] = tmp;
  }    
  
  return out;
}

template<typename Scalar> 
std::vector<Scalar> MatrixVectorMul(const std::vector<std::vector<Scalar>>& M, const std::vector<Scalar>& v, const std::vector<Scalar>& b) {
  std::vector<Scalar> out;
  out.resize(M.size(), 0.0);

  if( M.size() != b.size()) throw std::domain_error( "bad M.size() != b.size()" );          
  for (unsigned int i = 0; i < M.size(); i++) {
    for (unsigned int j = 0; j < v.size(); j++) {
      if( M[i].size() != v.size()) throw std::domain_error( "bad M[i].size() != v.size()" );          
      out[i] += M[i][j] * v[j];
    }
    out[i] += b[i];
  } 
  
  return out;
}

template<typename Scalar> 
std::vector<Scalar> MatrixVectorMul(const std::vector<std::vector<Scalar>>& M, const std::vector<Scalar>& v) {
  std::vector<Scalar> out;
  out.resize(M.size(), 0.0);

  //if( M.size() != b.size()) throw std::domain_error( "bad M.size() != b.size()" );          
  for (unsigned int i = 0; i < M.size(); i++) {
    for (unsigned int j = 0; j < v.size(); j++) {
      if( M[i].size() != v.size()) throw std::domain_error( "bad M[i].size() != v.size()" );          
      out[i] += M[i][j] * v[j];
    }
    //out[i] += b[i];
  } 
  
  return out;
}


template<typename Scalar> 
std::vector<Scalar> VectorAdd(const std::vector<Scalar>& v1, const std::vector<Scalar>& v2){
  if( v1.size() != v2.size()) throw std::domain_error( "bad add v1.size() != v2.size()" );            
  std::vector<Scalar> out;
  out.resize(v1.size(), 0.0);
  for (unsigned int i = 0; i < v1.size(); i++) {
    out[i] = v1[i] + v2[i];
  }
  
  return out;
}

template<typename Scalar> 
std::vector<Scalar> VectorMul(const std::vector<Scalar>& v1, const std::vector<Scalar>& v2){
  if( v1.size() != v2.size()) throw std::domain_error( "bad mul v1.size() != v2.size()" );            
  std::vector<Scalar> out;
  out.resize(v1.size(), 0.0);
  for (unsigned int i = 0; i < v1.size(); i++) {
    out[i] = v1[i] * v2[i];
  }
  
  return out;
}

template<typename Scalar> 
std::vector<Scalar> VectorTanh(const std::vector<Scalar>& v){
  std::vector<Scalar> out;
  out.resize(v.size(), 0.0);
  for (unsigned int i = 0; i < v.size(); i++) {
    out[i] = tanh(v[i]);
  }
  
  return out;
}

template<typename Scalar> 
std::vector<Scalar> VectorSigmoid(const std::vector<Scalar>& v){
  std::vector<Scalar> out;
  out.resize(v.size(), 0.0);
  for (unsigned int i = 0; i < v.size(); i++) {
    out[i] = 1 / (1 + exp(-v[i]));
  }
  
  return out;
}

template<typename Scalar> 
std::vector<Scalar> SoftMax(const std::vector<Scalar>& v){
  float sum_exp = 0.0;
  float max = -10000;
  std::vector<Scalar> out;
  out.resize(v.size(), 0.0); 
  
  
  for (unsigned int i = 0; i < v.size(); i++) {
    if (v[i] > max) {
      max = v[i];
    } 
  } 
 
  for (unsigned int i = 0; i < v.size(); i++) {
    sum_exp += exp(v[i] - max);
  }


  for (unsigned int i = 0; i < v.size(); i++) {
    out[i] = exp(v[i] - max) / sum_exp;
  }
  
  return out;
}



// mean absolute error
template<typename Scalar> 
Scalar VectorMAE(const std::vector<Scalar>& v1, const std::vector<Scalar>& v2){
  if( v1.size() != v2.size()) throw std::domain_error( "bad mul v1.size() != v2.size()" );            
  Scalar err;
  
  for (unsigned int i = 0; i < v1.size(); i++) {
    err = abs(v1[i] - v2[i]);
  }
  err = err / v1.size();
  
  return err;
}

double ReducePrecision(const double in){
  double out;
  float out_ft;
  out_ft = (float) in;
  
  unsigned int x = *reinterpret_cast<unsigned int*>(&out_ft);
  x = x & 0xfff80000;
  out_ft = *reinterpret_cast<float *>(&x);
  out = (double) out_ft;
  
  return out;
}

#endif

/*
 * Copyright (c) 2016-2018, Harvard University.  All rights reserved.
*/

#ifndef __ADPFLOATUTILS__
#define __ADPFLOATUTILS__

#include <nvhls_int.h>
#include <nvhls_types.h>
#include "AdpfloatSpec.h"

// remove later
//#include <bitset> 

// XXX: Only support UseDenormal = 0
// F: output fixed point width
// W, E: adpfloat<W,E> 
template <unsigned int W, unsigned int E, unsigned int F> 
inline void adpfloat_mul_template(const AdpfloatType<W,E> in_a, const AdpfloatType<W,E> in_b, NVINTW(F)& out) {
//  NVUINTW(W-E) in_a_tmp = 0, in_b_tmp = 0; // mantissa width + 1  
//  NVUINTW(2*(W-E)) man_mul;
  NVUINTW(W-E)        in_a_tmp, in_b_tmp; // mantissa width + 1  
  NVINTW(2*(W-E)+1)   man_mul;
  NVUINTW(E+1)        exp_sum;
  NVINTW(F)           out_tmp;
  
  in_a_tmp = in_a.man;
  in_b_tmp = in_b.man;  
  
  if (!in_a.is_zero()) {
    in_a_tmp[W-E-1] = 1;
  }
  if (!in_b.is_zero()) {
    in_b_tmp[W-E-1] = 1;
  }
  exp_sum = in_a.exp+in_b.exp;
  man_mul = in_a_tmp*in_b_tmp;
  if (in_a.sign^in_b.sign) {                      // output negative
    man_mul = -man_mul;
  }
  
  out_tmp = man_mul; 
  out_tmp = out_tmp  << exp_sum;
    
  //cout << "in_a.man " << in_a.man << " " << in_a.man.to_string(AC_BIN) << endl;
  //cout << "in_a_tmp " << in_a_tmp << " " << in_a_tmp.to_string(AC_BIN) << endl;
  //cout << "in_b_tmp " << in_b_tmp << " " << in_b_tmp.to_string(AC_BIN) << endl;
  //cout << "man_mul " << man_mul << " " << man_mul.to_string(AC_BIN) << endl;
  //cout << "out " << out << " " << std::bitset<32>(out.to_int64()) << endl;
  
  out = out_tmp;
  
  return;
}


template <unsigned int W, unsigned int E> 
inline void adpfloat_add_template(const AdpfloatType<W,E> in_a, const AdpfloatType<W,E> in_b, AdpfloatType<W,E>& out) {
  AdpfloatType<W,E> out_tmp;
  
  
  static const unsigned int M = W-E-1;
  //cout << "ina, inb: " << in_a << ", " << in_b << endl;  

  ac_float<M+2,2,E+2,AC_RND>  in_a_ac, in_b_ac, out_ac;
  in_a_ac = in_a.to_ac_float();
  in_b_ac = in_b.to_ac_float();
  out_ac.add(in_a_ac, in_b_ac);
  //cout << "in_a_ac " << in_a_ac << endl;
  //cout << "in_b_ac " << in_b_ac << endl;
  //cout << "out_ac " << out_ac << endl;
  
  out_tmp.set_value_ac_float(out_ac);
  
  out = out_tmp;
}

template <unsigned int W, unsigned int E> 
inline void adpfloat_max_template(const AdpfloatType<W,E>& in_a, const AdpfloatType<W,E>& in_b, AdpfloatType<W,E>& out) {
  static const unsigned int M = W-E-1;
  //cout << "ina, inb: " << in_a << ", " << in_b << endl;  

  ac_float<M+2,2,E+2,AC_RND>  in_a_ac, in_b_ac, out_ac;
  in_a_ac = in_a.to_ac_float();
  in_b_ac = in_b.to_ac_float();
  if (in_a_ac >= in_b_ac) {
     out_ac = in_a_ac;
  }
  else {
     out_ac = in_b_ac;    
  }
  out.set_value_ac_float(out_ac);
}

template <unsigned int W, unsigned int E> 
inline void adpfloat_mean_template(const AdpfloatType<W,E>& in_a, const AdpfloatType<W,E>& in_b, AdpfloatType<W,E>& out) {
  static const unsigned int M = W-E-1;
  //cout << "ina, inb: " << in_a << ", " << in_b << endl;  

  ac_float<M+2,2,E+2,AC_RND>  in_a_ac, in_b_ac, out_ac;
  in_a_ac = in_a.to_ac_float();
  in_b_ac = in_b.to_ac_float();
  out_ac.add(in_a_ac, in_b_ac);
  out_ac = out_ac >> 1;
  //cout << "in_a_ac " << in_a_ac << endl;
  //cout << "in_b_ac " << in_b_ac << endl;
  //cout << "out_ac " << out_ac << endl;
  
  out.set_value_ac_float(out_ac);
}

// non-template type of adpfloat multiplication 
inline void adpfloat_mul(const AdpfloatType<8,3> in_a, 
                  const AdpfloatType<8,3> in_b, 
                  NVINTW(16)& out) {
  adpfloat_mul_template<8, 
                        3, 
                        16> (in_a, in_b, out);
}

inline void adpfloat_add(const AdpfloatType<8,3> in_a, 
                  const AdpfloatType<8,3> in_b, 
                  AdpfloatType<8,3>& out) {
                  
  adpfloat_add_template<8, 
                        3> (in_a, in_b, out);                  
                  
}

inline void adpfloat_max(const AdpfloatType<8,3> in_a, 
                  const AdpfloatType<8,3> in_b, 
                  AdpfloatType<8,3>& out) {
  adpfloat_max_template<8, 
                        3> (in_a, in_b, out);       
}

inline void adpfloat_mean(const AdpfloatType<8,3> in_a, 
                  const AdpfloatType<8,3> in_b, 
                  AdpfloatType<8,3>& out) {
  adpfloat_mean_template<8, 
                        3> (in_a, in_b, out);       
}
#endif

/*
 * Copyright (c) 2016-2018, Harvard University.  All rights reserved.
*/

#ifndef __ADPFLOATSPEC__
#define __ADPFLOATSPEC__

#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_message.h>
#include <math.h> // pow()
#include <bitset> // std::bitset
#include "Spec.h"
#include <TypeToBits.h>

#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_float.h>
#include <ac_math.h>

namespace spec {
  namespace Adpfloat {
    const bool UseDenormal = 0; // Whether the chip use Denormal or not  
    const bool ReserveZero = 1; // if not using denormal, reserve 00....0, 10....0 as zero
  }
}

typedef spec::AdpfloatBiasType AdpfloatBiasType;

// AdpfloatType<8,3>: 8-bit adaptive float type with 3-bit exponent
// XXX: both 1000,0000 or 0000,0000 stands for zero, might implement signed zero later
template <unsigned int W, unsigned int E>
class AdpfloatType : public nvhls_message{
  static const unsigned int M = W-E-1;
  public:
  bool sign;
  NVUINTW(E) exp;
  NVUINTW(M) man;

  static const unsigned int width = W;
  
  // reverse ordering 
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m & man;  //LSB
    m & exp;
    m & sign; //MSB
  }

  AdpfloatType() {
    sign = 0;
    exp = 0;
    man = 0;
  }
  
  void Reset() {
    sign = 0;
    exp = 0;
    man = 0;  
  }
  
  AdpfloatType(const NVUINTW(width) & rawbits) {
    *this = NVUINTToType<AdpfloatType>(rawbits);
  }

  NVUINTW(width) to_rawbits() const {
    return TypeToNVUINT(*this);
  }  
  
  bool is_zero() const {
    bool out = 0;
    if (exp == 0 && man == 0) 
      out = 1;
    return out;
  }
  
  ac_float<M+2,2,E+2,AC_RND> to_ac_float(AdpfloatBiasType adpfloat_bias = 0) const {
    if (spec::Adpfloat::ReserveZero && is_zero()) return 0;    
    NVINTW(M+2) man_plus2;
    man_plus2 = man;
    man_plus2[M] = 1; 
    
    if (sign == 1) {
      man_plus2 = -man_plus2;
    }
    
    ac_float<M+2,2,E+2,AC_RND> out_ac_float;
    out_ac_float.e = exp + adpfloat_bias + spec::kAdpfloatOffset;
    out_ac_float.m.set_slc(0, man_plus2);
    
    return out_ac_float;
  }
  
  void set_value_ac_float(ac_float<M+2,2,E+2,AC_RND> in_ac_float, AdpfloatBiasType adpfloat_bias = 0){
    NVINTW(E+2) exp_tmp;
    exp_tmp = in_ac_float.e - adpfloat_bias - spec::kAdpfloatOffset;

    if (in_ac_float.m == -2) {
      exp_tmp += 1;
    }
    if (in_ac_float == 0 || exp_tmp < 0) {
      sign = 0;
      exp = 0;
      man = 0;
    } 
    // if too large 
    else if (exp_tmp >= (1 << E)){
      if (in_ac_float.m < 0) {
        sign = 1;
      }
      else {
        sign = 0;
      }      
      exp = (1 << E) -1;
      man = (1 << M) -1;
    }
    else {
      exp = exp_tmp;
      if (in_ac_float.m < 0) {
        sign = 1;
        in_ac_float.m = -in_ac_float.m;
      }
      else {
        sign = 0;
      }
      man.set_slc(0, nvhls::get_slc<M>(in_ac_float.m ,0));
    }
    return;
  }
  
  // Non-Synthesizeble
  float to_float(AdpfloatBiasType adpfloat_bias = 0) const { 
    if (spec::Adpfloat::ReserveZero && is_zero()) return 0;
    
    float float_man;
    float_man = man;
    float_man = float_man/(pow(2, man.width));      
    float float_exp;
    float float_sign;
    // cannot use shift on NVUINT
    float_sign = sign? -1: 1;
    
    if (spec::Adpfloat::UseDenormal == 1) {         // if use denormal
      if (exp == 0) {   // denormal range
        float_exp = 1 + adpfloat_bias + spec::kAdpfloatOffset; 
        return float_sign*pow(2, float_exp)*float_man;
      }
      else {            // normal range
        float_exp = exp + adpfloat_bias + spec::kAdpfloatOffset; 
        return float_sign*pow(2, float_exp)*(1+float_man); // 
      }
    }
    else {                                          // if not use denormal
      float_exp = exp + adpfloat_bias + spec::kAdpfloatOffset;
      return float_sign*pow(2, float_exp)*(1+float_man); 
    }
  }
  
  // Non-Synthesizeble
  // Without denormal float range is [0 ~ 2^E-1] + adpfloat_bias + offset 
  // With denormal float range is [denormal, 1 ~ 2^E-1] + adpfloat_bias + offset
  float max_value (AdpfloatBiasType adpfloat_bias = 0) const {
    // max mantissa
    float max_exp = pow(2, exp.width)-1 + (int)adpfloat_bias + spec::kAdpfloatOffset;
    float max_tmp = pow(2,max_exp)*(2 - 1/(pow(2, man.width)));
    return max_tmp;
  }
  
  float min_value (AdpfloatBiasType adpfloat_bias = 0) const {
    float min_exp;
    // TODO: print min Normal
    if (spec::Adpfloat::UseDenormal) {
      min_exp = (float) adpfloat_bias + spec::kAdpfloatOffset + 1;
      return pow(2,min_exp)*(1/(pow(2, man.width)));
    }
    else if (spec::Adpfloat::ReserveZero) {
      min_exp = (float) adpfloat_bias + spec::kAdpfloatOffset;
      return pow(2,min_exp)*(1+ 1/(pow(2, man.width)));
    }
    else {
      min_exp = (float) adpfloat_bias + spec::kAdpfloatOffset;      
      return pow(2,min_exp);
    }
  }
  
  // Rounding needed? 
  // TODO: Only implement non-denormal version (only Zero Reserved version checked)
  void set_value(float value, AdpfloatBiasType adpfloat_bias = 0){
    assert(spec::Adpfloat::UseDenormal == 0 && spec::Adpfloat::ReserveZero == 1);
    bool sign_tmp = (value < 0)?1:0;
    float value_abs = abs(value);
    
    if (value_abs < 0.5*min_value(adpfloat_bias)) {
      sign = 0;
      exp = 0;
      man = 0;
      return ;
    }
    else if (value_abs < min_value(adpfloat_bias)) {
      value_abs = min_value(adpfloat_bias);
    }
    else if (value_abs > max_value(adpfloat_bias)) {
      value_abs = max_value(adpfloat_bias);      
    }

    // get float32 bitseq
    unsigned long x = *reinterpret_cast<unsigned long*>(&value_abs);
    std::bitset<32> bitset_tmp(x);
    
    NVUINTW(32) nvuint32_tmp;
    for (int i = 32-1; i >= 0; i--)
		  nvuint32_tmp[i] = bitset_tmp[i]; 
    
    // 1. get sign bit 
    sign = sign_tmp;
    
    // 2. get exponent bit and shift by relative offset between float and adpfloat
    NVUINTW(8) exp_tmp;
    exp_tmp.set_slc(0, nvuint32_tmp.slc<8>(23));
    exp = exp_tmp - 127 - spec::kAdpfloatOffset - adpfloat_bias;
    
    // get mantissa bit
    NVUINTW(23) man_tmp;
    man_tmp.set_slc(0, nvuint32_tmp.slc<23>(0));
    // TODO: Rounding not implemented  
    man.set_slc(0, man_tmp.slc<W-E-1>(23 - (W-E-1)));
  }
  
  // 20190307 implement rounding 
  template<unsigned int W1, unsigned int F1>
  void set_value_fixed(const NVINTW(W1) in, const AdpfloatBiasType adpfloat_bias = 0) {
    ac_fixed<W1, W1-F1, true, AC_TRN, AC_WRAP> in_ac_fixed;
    in_ac_fixed.set_slc(0, in);
    ac_float<M+2,2,E+W1,AC_RND> in_ac_float;
    in_ac_float = in_ac_fixed;
    
    NVINTW(E+W1) exp_tmp;
    exp_tmp = in_ac_float.e - adpfloat_bias - spec::kAdpfloatOffset;
    if (in_ac_float.m == -2) {
      exp_tmp += 1;
    }
    
    // in = 0 or if exp_tmp too small
    if (in == 0 || exp_tmp < 0) {
      sign = 0;
      exp = 0;
      man = 0;
    } 
    // if too large 
    else if (exp_tmp >= (1 << E)){
      exp = (1 << E) -1;
      man = (1 << M) -1;
    }
    else {
      exp = exp_tmp;
      if (in_ac_float.m < 0) {
        sign = 1;
        in_ac_float.m = -in_ac_float.m;
      }
      else {
        sign = 0;
      }
      man.set_slc(0, nvhls::get_slc<M>(in_ac_float.m ,0));
    }
    return; 
  }
  
  template<unsigned int W1, unsigned int F1> // F1 >> M
  NVINTW(W1) to_fixed(const AdpfloatBiasType adpfloat_bias = 0) const {
    if (spec::Adpfloat::ReserveZero && is_zero()) return 0;
    NVUINTW(M+1) man_plus1; // mantissa width + 1  
    NVINTW(W1) out;
    
    man_plus1.set_slc(0, man);
    man_plus1[M] = 1;
    
    out = man_plus1; 
    out = out << (exp + adpfloat_bias + spec::kAdpfloatOffset - M + F1);
    
    if (sign) {
      out = -out;
    }
    
    return out;
  }
};


template <unsigned int W, unsigned int E>
inline bool operator==(const AdpfloatType<W, E>& lhs, const AdpfloatType<W, E>& rhs)
{
  bool is_equal = true;
  is_equal &= (lhs.sign == rhs.sign);
  is_equal &= (lhs.exp == rhs.exp);
  is_equal &= (lhs.man == rhs.man);

  return is_equal;
}

template <unsigned int W, unsigned int E>
inline std::ostream& operator<<(ostream& os, const AdpfloatType<W, E>& adpfloat)
{
  os << "<sign, exp, man> <" << adpfloat.sign << " " << adpfloat.exp << " " << adpfloat.man << "> ";
  
  return os;
}

#endif 

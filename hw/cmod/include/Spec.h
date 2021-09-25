
#ifndef __SPEC__
#define __SPEC__

#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_message.h>
#include <math.h> // pow()
#include <bitset> // std::bitset
#include <TypeToBits.h>
#include <ArbitratedScratchpad.h>


#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_float.h>
#include <ac_math.h>

namespace spec {
  const int kVectorSize = VECTOR_SIZE;
  typedef typename nvhls::nv_scvector< NVUINT8, kVectorSize> VectorType;

  typedef typename nvhls::nv_scvector< nvhls::nv_scvector<NVUINT8, kVectorSize>, kVectorSize> MatrixType;

  const int kAccumWordWidth = 16;
  typedef NVINTW(kAccumWordWidth) AccumScalarType;

  typedef typename nvhls::nv_scvector<AccumScalarType, kVectorSize> AccumVectorType;

  typedef typename nvhls::nv_scvector< nvhls::nv_scvector<AccumScalarType, kVectorSize>, kVectorSize> AccumMatrixType;

  typedef NVINT8   InputType;
  const int N = kVectorSize;         // # banks = N
  const int Entries = 4096;                // # of entries per bank (524KB for input buffer)
  const int Capacity = N*Entries;          
  const int AuxEntries = 256;
  const int AuxCapacity = N*AuxEntries; 
 
  // Input Memory type
  typedef ArbitratedScratchpad<InputType, Capacity, N, N, 0> InputMemType;
  static const int IndexBits = nvhls::nbits<Entries-1>::val;
  typedef NVUINTW(IndexBits) IndexType;
  typedef NVUINTW(InputMemType::addr_width) AddrType;
  typedef InputMemType::req_t input_req_t;
  typedef InputMemType::rsp_t input_rsp_t;  
 
  // Mask Memory type
  typedef NVUINTW(kVectorSize) MaskType;
  //typedef bool MaskType;
  typedef ArbitratedScratchpad<MaskType, Entries, 1, 1, 0> MaskMemType;
  typedef NVUINTW(MaskMemType::addr_width) MaskAddrType;
  typedef MaskMemType::req_t mask_req_t;
  typedef MaskMemType::rsp_t mask_rsp_t; 

  // Auxiliary Memory type
  typedef ArbitratedScratchpad<InputType, AuxCapacity, N, N, 0> AuxMemType;
  static const int AuxIndexBits = nvhls::nbits<AuxEntries-1>::val;
  typedef NVUINTW(AuxIndexBits) AuxIndexType;
  typedef NVUINTW(AuxMemType::addr_width) AuxAddrType;
  typedef AuxMemType::req_t aux_req_t;
  typedef AuxMemType::rsp_t aux_rsp_t;  


  // DVFS Memory type
  typedef ArbitratedScratchpad<InputType, 256, 1, 1, 0> DVFSMemType;


  const int kActWordWidth = kAccumWordWidth;

  const int kActWordMax = (1 << (kActWordWidth-1)) -1;

  const int kActWordMin = -kActWordMax;

  const int kActNumFrac = 10;
  const int kActNumInt = kActWordWidth - kActNumFrac;
  typedef NVINTW(kActWordWidth) ActScalarType;
  typedef typename nvhls::nv_scvector<ActScalarType, kVectorSize> ActVectorType;

  const int kLayerNormSumWidth = 20;
  typedef NVINTW(kLayerNormSumWidth) LayerNormSumType;

  //const int kSoftMaxSumWidth = 20;
  //typedef NVINTW(kSoftMaxSumWidth) SoftMaxSumType; 

  //AdaptivFloat settings
  const int kAdpfloatWordWidth = 8;
  const int kAdpfloatExpWidth = 3;  // 0 ~ 7 (or denormal+ 1~7) 
  const int kAdpfloatManWidth = kAdpfloatWordWidth-kAdpfloatExpWidth-1;
  const int kAdpfloatBiasWidth = 3; // 0 ~ 7
  typedef NVUINTW(kAdpfloatBiasWidth) AdpfloatBiasType;
  const int kAdpfloatOffset = -10;

  class AccelConfig: public nvhls_message{
   public:
    bool    is_relu;
    bool    is_bias;
    NVINT8 weight_bias;
    NVUINT3 adf_accum_bias;
    NVUINT5 accum_right_shift;
    static const unsigned int width = 1+1+8+3+5;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& is_relu;
      m& is_bias;
      m& weight_bias;
      m& adf_accum_bias;
      m& accum_right_shift;
    }
  };

  //Matrix multiplication between (N0xM)x(MxN1)
  class MatrixConfig: public nvhls_message{
   public:
    NVUINT10 N0;
    NVUINT10 N1;
    NVUINT12 M;
    static const unsigned int width = 10+10+12;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& N0;
      m& N1;
      m& M;
    }
  };

  //Input Buffer Configurations
  class InputBufferConfig: public nvhls_message{
   public:
    NVUINT12 base_input[2];
    //NVUINT12 base_input1;
    static const unsigned int width = 12+12;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& base_input[0];
      m& base_input[1];
    }
  };

  //Input Buffer Base Offset Configurations
  class InputBufferBaseOffsetConfig: public nvhls_message{
   public:
    NVUINT12 base_input_offset[2];
    //NVUINT12 base_input1;
    static const unsigned int width = 12+12;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& base_input_offset[0];
      m& base_input_offset[1];
    }
  };

  //Mode - 0: LayerNorm, 1: SMax, 2: ElemAdd, 3: Enpy
  /*class ModeConfig: public nvhls_message{
   public:
    NVUINT1 mode_input[4];
    static const unsigned int width = 1+1+1+1;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& mode_input[0];
      m& mode_input[1];
      m& mode_input[2];
      m& mode_input[3];
    }
  }; */

  //Aux Buffer Configurations
  class DvfsConfig: public nvhls_message{
   public:
    NVUINT8   enpy_scale;
    //NVUINT16  target_time;
    //NVUINT8   base_dvfs;
    static const unsigned int width = 8;
   
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& enpy_scale;
      //m& target_time;
      //m& base_dvfs;
    }
  };

  class DCOConfigA: public nvhls_message{
   public:
    NVUINT6   dco_val0;
    NVUINT6   dco_val1;
    NVUINT6   dco_val2;
    NVUINT6   dco_val3;
    NVUINT6   dco_val4;
    static const unsigned int width = 6+6+6+6+6;
   
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& dco_val0;
      m& dco_val1;
      m& dco_val2;
      m& dco_val3;
      m& dco_val4;
    }
  };

  class DCOConfigB: public nvhls_message{
   public:
    NVUINT6   dco_val5;
    NVUINT6   dco_val6;
    NVUINT6   dco_val7;
    NVUINT6   dco_val8;
    NVUINT6   dco_val9;
    static const unsigned int width = 6+6+6+6+6;
   
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& dco_val5;
      m& dco_val6;
      m& dco_val7;
      m& dco_val8;
      m& dco_val9;
    }
  };

  class DCOConfigC: public nvhls_message{
   public:
    NVUINT6   dco_val10;
    NVUINT6   dco_val11;
    NVUINT6   dco_val12;
    NVUINT6   dco_val13;
    NVUINT6   dco_val14;
    static const unsigned int width = 6+6+6+6+6;
   
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& dco_val10;
      m& dco_val11;
      m& dco_val12;
      m& dco_val13;
      m& dco_val14;
    }
  };

  class LDOConfigA: public nvhls_message{
   public:
    NVUINT8   ldo_val0;
    NVUINT8   ldo_val1;
    NVUINT8   ldo_val2;
    NVUINT8   ldo_val3;
    static const unsigned int width = 8+8+8+8;
   
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& ldo_val0;
      m& ldo_val1;
      m& ldo_val2;
      m& ldo_val3;
    }
  };


  class LDOConfigB: public nvhls_message{
   public:
    NVUINT8   ldo_val4;
    NVUINT8   ldo_val5;
    NVUINT8   ldo_val6;
    NVUINT8   ldo_val7;
    static const unsigned int width = 8+8+8+8;
   
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& ldo_val4;
      m& ldo_val5;
      m& ldo_val6;
      m& ldo_val7;
    }
  };


  class LDOConfigC: public nvhls_message{
   public:
    NVUINT8   ldo_val8;
    NVUINT8   ldo_val9;
    NVUINT8   ldo_val10;
    NVUINT8   ldo_val11;
    static const unsigned int width = 8+8+8+8;
   
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& ldo_val8;
      m& ldo_val9;
      m& ldo_val10;
      m& ldo_val11;
    }
  };


  class LDOConfigD: public nvhls_message{
   public:
    NVUINT8   ldo_val12;
    NVUINT8   ldo_val13;
    NVUINT8   ldo_val14;
    NVUINT8   ldo_val15;
    static const unsigned int width = 8+8+8+8;
   
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& ldo_val12;
      m& ldo_val13;
      m& ldo_val14;
      m& ldo_val15;
    }
  };

  /*class VddConfig: public nvhls_message{
   public:
    NVUINT8   vdd_scale;
    NVUINT8   base_vdd;
    static const unsigned int width = 8+8;
   
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& vdd_scale;
      m& base_vdd;
    }
  }; */

  class PeriphConfig: public nvhls_message{
   public:
    NVUINT7  base_attn_span;
    NVUINT8  base_gamma;
    NVUINT8  base_beta;
    spec::AdpfloatBiasType adpbias_attn_span;
    spec::AdpfloatBiasType adpbias_gamma;
    spec::AdpfloatBiasType adpbias_beta;
    static const unsigned int width = 7+8+8+3+3+3; 
   
    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& base_attn_span;
      m& base_gamma;
      m& base_beta;
      m& adpbias_gamma;
      m& adpbias_beta;
      m& adpbias_attn_span;
    }
  };

  class GBControlConfig: public nvhls_message{
   public: 
    NVUINT8   num_vector;
    NVUINT8   num_timestep;
    spec::AdpfloatBiasType adpbias_act1;
    spec::AdpfloatBiasType adpbias_act2;
    spec::AdpfloatBiasType adpbias_act3;
    static const unsigned int width = 8+8+3+3+3; 

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& num_vector;
      m& num_timestep;
      m& adpbias_act1;
      m& adpbias_act2;
      m& adpbias_act3;
    }
  };

  class EnpyConfig: public nvhls_message{
   public: 
    //NVUINT3       enpy_status;
    ActScalarType enpy_threshold;
    static const unsigned int width = 16;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      //m& enpy_status;
      m& enpy_threshold;
    }
  };

} 
#endif 

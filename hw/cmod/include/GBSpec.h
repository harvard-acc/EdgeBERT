/*
 * Copyright (c) 2016-2018, Harvard University.  All rights reserved.
*/
#ifndef __GBSPEC_H_
#define __GBSPEC_H_

#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_message.h>
#include "Spec.h"
#include "AxiSpec.h"
#include "AdpfloatSpec.h"

namespace spec {
  namespace GB {  
    namespace Input {
      // Parameters for Global Buffer 
      typedef VectorType WordType;
      class DataReq : public nvhls_message{
       public:
        NVUINT1     is_write;
        NVUINT8     vector_index;        
        NVUINT8     timestep_index;
        WordType    write_data;        
        
        static const unsigned int width = 1 + 8 + 8 + WordType::width;
        template <unsigned int Size>
        void Marshall(Marshaller<Size>& m) {
          m & is_write;
          m & timestep_index;
          m & vector_index;        
          m & write_data;
        }
        DataReq() {
          Reset();
        }   
        void Reset() {
          is_write = 0;
          timestep_index = 0;
          vector_index = 0;
          write_data = 0;
        }
      };     
            
     
      template<unsigned N>
      class DataRsp : public nvhls_message{
       public:
        nvhls::nv_scvector<WordType, N> read_vector;
        
        static const unsigned int width = nvhls::nv_scvector<WordType, N>::width;
        template <unsigned int Size>
        void Marshall(Marshaller<Size>& m) {
          m & read_vector;
        }
        DataRsp() {
          Reset();
        }
        void Reset() {
          read_vector = 0;
        }
      };
     
    };  
    
    namespace Aux {
      typedef VectorType WordType;

      class DataReq : public nvhls_message {
       public:
        NVUINT1     is_write;           // 1: write
        NVUINT8     vector_index;        
        WordType    write_data;        
        
        static const unsigned int width = 1 + 8 + WordType::width;
        template <unsigned int Size>
        void Marshall(Marshaller<Size>& m) {
          m & is_write;
          m & vector_index;        
          m & write_data;
        }
        DataReq() {
          Reset();
        }   
        void Reset() {
          is_write  = 0;
          vector_index = 0;
          write_data = 0;
        }
      };     
     
      class DataRsp : public nvhls_message {
       public:
        WordType read_data;
        
        static const unsigned int width = WordType::width;
        template <unsigned int Size>
        void Marshall(Marshaller<Size>& m) {
          m & read_data;
        }
        DataRsp() {
          Reset();
        }
        void Reset() {
          read_data = 0;
        }
      };
    } // namespace Aux
  } // namespace GB

class LayerNormConfig {
 public:
  NVUINT9  base_gamma;
  NVUINT9  base_beta;
  spec::AdpfloatBiasType adpbias_gamma;
  spec::AdpfloatBiasType adpbias_beta;
  static const unsigned int width = 9+9+3+3; 
 
  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m& base_gamma;
    m& base_beta;
    m& adpbias_gamma;
    m& adpbias_beta;
  }
};

class GBControlConfig {
 public: 
  NVUINT8   num_vector;
  NVUINT8   num_timestep;
  spec::AdpfloatBiasType adpbias_act1;
  spec::AdpfloatBiasType adpbias_act2;
  static const unsigned int width = 8+8+3+3; 

  template <unsigned int Size>
  void Marshall(Marshaller<Size>& m) {
    m& num_vector;
    m& num_timestep;
    m& adpbias_act1;
    m& adpbias_act2;
  }
  
  NVUINT8  vector_counter;
  NVUINT8  timestep_counter;
  
  /*void Reset() {
    num_vector      = 1;
    num_timestep    = 1;
    adpbias_act1    = 0;
    adpbias_act2    = 0;
    ResetCounter();
  }*/ 

  /* void ConfigWrite(const NVUINT8 write_index, const NVUINTW(write_width)& write_data) {
    if (write_index == 0x01) {
      num_vector    = nvhls::get_slc<8>(write_data, 56);
      num_timestep  = nvhls::get_slc<16>(write_data, 80); 
      adpbias_act1       = nvhls::get_slc<spec::kAdpfloatBiasWidth>(write_data, 96);  
      adpbias_act2       = nvhls::get_slc<spec::kAdpfloatBiasWidth>(write_data, 104);              
      adpbias_gamma       = nvhls::get_slc<spec::kAdpfloatBiasWidth>(write_data, 112);        
      adpbias_beta       = nvhls::get_slc<spec::kAdpfloatBiasWidth>(write_data, 120);        
    }
  }

  void ConfigRead(const NVUINT8 read_index, NVUINTW(write_width)& read_data) const {
    read_data = 0;
    if (read_index == 0x01) {
      read_data.set_slc<8>(48, num_vector);
      read_data.set_slc<16>(80, num_timestep);
      
      read_data.set_slc<spec::kAdpfloatBiasWidth>(96, adpbias_act1);  
      read_data.set_slc<spec::kAdpfloatBiasWidth>(104, adpbias_act2);            
      read_data.set_slc<spec::kAdpfloatBiasWidth>(112, adpbias_gamma);      
      read_data.set_slc<spec::kAdpfloatBiasWidth>(120, adpbias_beta);      
    }
  } */ //Put in Control.h

  void ResetCounter() {
    vector_counter      = 0;
    timestep_counter    = 0;  
  }

  NVUINT8 GetVectorIndex() const {
    return vector_counter;
  }
  NVUINT8 GetTimestepIndex() const {
    return timestep_counter;
  }
  
  
  void UpdateVectorCounter(bool& is_end) {
    if (vector_counter >= (num_vector - 1)) {
      is_end = 1;
      vector_counter = 0;
    }
    else {
      vector_counter += 1;
    }
  }
  
  void UpdateTimestepCounter(bool& is_end) {
    is_end = 0;
    if (timestep_counter >= (num_timestep - 1)) {
      is_end = 1;
      timestep_counter = 0;
    }
    else {
      timestep_counter += 1;
    }
  }
};
} // namespace spec 
#endif

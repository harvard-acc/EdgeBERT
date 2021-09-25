
#ifndef __SMAX_H__
#define __SMAX_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_connections.h>
#include <nvhls_vector.h>
#include <nvhls_module.h>
#include "../include/Spec.h"
#include "../DecodeTop/DecodeTop.h"
#include "../AuxMem/AuxMem.h"
#include "../include/PPU.h"

SC_MODULE(SMax)
{
 public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef DecodeTop MaskMemType; 
  typedef AuxMem AuxMemType;
  typedef spec::IndexType IndexType;
  IndexType base_input_reg;   
  static const int N = spec::N; 
  static const int kDebugLevel = 0;

  spec::ActScalarType sum_exp, maximum_value;
  spec::ActVectorType softmax_result, out_data;

  Connections::In<bool> start;
  Connections::In<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations
  Connections::In<spec::PeriphConfig> softmax_config;
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<spec::VectorType>  act_rsp; //Activation response from DecodeTop
  Connections::In<spec::VectorType>  attn_span_rsp;  // From AuxMem

  Connections::Out<AuxMemType::aux_req_t>  attn_span_req;  // To AuxMem for read request
  Connections::Out<spec::VectorType>  act_out_vec; // To be stored in DecodeTop
  Connections::Out<MaskMemType::mask_req_t>  mask_rd_req; // To DecodeTop for mask read requests
  Connections::Out<bool> done;

  // Constructor
  SC_HAS_PROCESS(SMax);
  SMax(sc_module_name name_) : sc_module(name_) {

    SC_THREAD(SMaxRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);  
  }

  bool is_start, input_ok, sm_config_ok, gb_config_ok;

  enum FSM {
    IDLE, FINDMAX, FINDMAX2, SFM1, SFM1b, SFM2, SFM2b, SPAN_MASK, SPAN_MASK2, OUT, NEXT
  };
  FSM state;

  spec::PeriphConfig softmax_config_tmp;
  spec::GBControlConfig gbcontrol_config_tmp;
  spec::InputBufferConfig input_buffer_config_tmp;
  
  NVUINT7  base_attn_span;
  spec::AdpfloatBiasType adpbias_attn_span;

  NVUINT8   num_vector;
  NVUINT8   num_timestep;
  spec::AdpfloatBiasType adpbias_act1;

  NVUINT8 vector_index;
  NVUINT8 timestep_index;

  void Reset() {
    state = IDLE;
    is_start = 0;
    input_ok = 0;
    sm_config_ok = 0;
    gb_config_ok = 0;
    base_input_reg = 0;
    base_attn_span = 0;
    adpbias_attn_span = 0;
    num_vector      = 1;
    num_timestep    = 1;
    vector_index = 0;
    timestep_index = 0;
    adpbias_act1    = 0;
    ResetPorts();
    ResetSoftmax();
  }
  
  void ResetPorts() { 
    start.Reset();
    done.Reset();
    act_rsp.Reset();
    input_buffer_config.Reset();
    attn_span_rsp.Reset();
    attn_span_req.Reset();
    softmax_config.Reset();
    gbcontrol_config.Reset();
    act_out_vec.Reset();
    mask_rd_req.Reset();
  }
 
  void ResetSoftmax() {
    sum_exp = 0;
    maximum_value = spec::kActWordMin;
  }
 
  void ResetCounter() {
    vector_index      = 0;
    timestep_index    = 0;  
  }

  NVUINT8 GetVectorIndex() const {
    return vector_index;
  }
  NVUINT8 GetTimestepIndex() const {
    return timestep_index;
  }
  
  void UpdateVectorCounter(bool& is_end) {
    if (vector_index >= (num_vector - 1)) {
      is_end = 1;
      vector_index = 0;
    }
    else {
      vector_index += 1;
    }
  }
  
  void UpdateTimestepCounter(bool& is_end) {
    is_end = 0;
    if (timestep_index >= (num_timestep - 1)) {
      is_end = 1;
      timestep_index = 0;
    }
    else {
      timestep_index += 1;
    }
  } 

  void UpdateMax(const spec::ActVectorType attention_vector) {
    spec::ActScalarType new_max = spec::kActWordMin;    // should not be a reg

    #pragma hls_unroll yes 
    for (int i=0; i< N; i++){
      if (attention_vector[i] > new_max) {
        new_max = attention_vector[i]; 
      }
    }
       
    if (new_max > maximum_value) {
      maximum_value = new_max;
    }
  }
 
  void CheckStart() {
    bool start_reg;
    if (start.PopNB(start_reg) && gb_config_ok && sm_config_ok && input_ok) {
      is_start = 1;
      CDCOUT(sc_time_stamp()  << name() << " SMax Start !!!" << endl, kDebugLevel);
    }
  }

  void RunFSM() {
    if (gbcontrol_config.PopNB(gbcontrol_config_tmp)) {
      num_vector = gbcontrol_config_tmp.num_vector;
      num_timestep = gbcontrol_config_tmp.num_timestep;
      adpbias_act1 = gbcontrol_config_tmp.adpbias_act1;
      CDCOUT(sc_time_stamp()  << name() << " DUT - num_vector is: " << num_vector << endl, kDebugLevel);
      CDCOUT(sc_time_stamp()  << name() << " DUT - num_timestep is: " << num_timestep << endl, kDebugLevel);
      gb_config_ok = 1;
    }
    vector_index = GetVectorIndex();
    timestep_index = GetTimestepIndex();
    //CDCOUT(sc_time_stamp()  << name() << " DUT - vector_index: " << vector_index << endl, kDebugLevel);
    //CDCOUT(sc_time_stamp()  << name() << " DUT - timesetp_Index: " << timestep_index << endl, kDebugLevel);

    if (softmax_config.PopNB(softmax_config_tmp)) {
      base_attn_span = softmax_config_tmp.base_attn_span;
      adpbias_attn_span = softmax_config_tmp.adpbias_attn_span;
      sm_config_ok = 1;
    }

    if (input_buffer_config.PopNB(input_buffer_config_tmp)) {
       base_input_reg = input_buffer_config_tmp.base_input[0];
       input_ok = 1;
    }

    switch (state) {
      case IDLE: {
        ResetSoftmax();
        break;
      }
      case FINDMAX: {
        CDCOUT(sc_time_stamp()  << name() << " case FINDMAX" << endl, kDebugLevel);
        MaskMemType::mask_req_t mask_reg;
        mask_reg.type.val = CLITYPE_T::LOAD;
        mask_reg.addr[0] = base_input_reg + vector_index + timestep_index*num_vector;
        mask_reg.valids[0] = 1;
        mask_rd_req.Push(mask_reg);                
        break;
      }
      case FINDMAX2: {
        CDCOUT(sc_time_stamp()  << name() << " case FINDMAX2" << endl, kDebugLevel);
        // Receive Rsp        
        spec::VectorType act_reg;
        act_reg = act_rsp.Pop();
        CDCOUT(sc_time_stamp()  << name() << " FINDMAX2 popped -----------" << endl, kDebugLevel);
        spec::ActVectorType x_vector;
        
        // Get activation vector in ActScalarType
        Adpfloat2Fixed(act_reg,  x_vector, adpbias_act1);       
        // Find max
        UpdateMax(x_vector);
        break;
      }
      case SFM1: {
        CDCOUT(sc_time_stamp()  << name() << " case SFM1" << endl, kDebugLevel);
        MaskMemType::mask_req_t mask_reg;
        mask_reg.type.val = CLITYPE_T::LOAD;
        mask_reg.addr[0] = base_input_reg + vector_index + timestep_index*num_vector;
        mask_reg.valids[0] = 1;
        mask_rd_req.Push(mask_reg);  
        break;
      }    
      case SFM1b: {
        CDCOUT(sc_time_stamp()  << name() << " case SFM1b" << endl, kDebugLevel);
        // Receive Rsp        
        spec::VectorType act_reg;
        act_reg = act_rsp.Pop();
        CDCOUT(sc_time_stamp()  << name() << " SFM1b popped -----------" << endl, kDebugLevel);

        spec::ActScalarType tmp_sum = 0;
        
        spec::ActVectorType exp_vector(act_reg.to_rawbits());
        
        #pragma hls_unroll yes
        for (int i = 0; i < N; i++) {
           exp_vector[i] -= maximum_value;
        }        
        Exponential(exp_vector, exp_vector);
        
        #pragma hls_unroll yes
        for (int i = 0; i < N; i++) {
          tmp_sum += exp_vector[i];
        }             
        sum_exp += tmp_sum;
        break;
      } 
      case SFM2: {
        CDCOUT(sc_time_stamp()  << name() << " case SFM2" << endl, kDebugLevel);
        MaskMemType::mask_req_t mask_reg;
        mask_reg.type.val = CLITYPE_T::LOAD;
        mask_reg.addr[0] = base_input_reg + vector_index + timestep_index*num_vector;
        mask_reg.valids[0] = 1;
        mask_rd_req.Push(mask_reg);  
        break;
      }    
      case SFM2b: {
        CDCOUT(sc_time_stamp()  << name() << " case SFM2b" << endl, kDebugLevel);
        // Receive Rsp  
        //spec::ActVectorType sum_exp_vector;      
        //spec::ActVectorType log_vector;      
        spec::VectorType act_reg;

        act_reg = act_rsp.Pop();
        CDCOUT(sc_time_stamp()  << name() << " SFM2b popped -----------" << endl, kDebugLevel);
        spec::ActVectorType exp_vector(act_reg.to_rawbits());

        /*#pragma hls_unroll yes
        for (int i = 0; i < N; i++) {
          sum_exp_vector[i] = sum_exp;
        } */    

        LogN(sum_exp, sum_exp);

        #pragma hls_unroll yes
        for (int i = 0; i < N; i++) {
          exp_vector[i] -= (maximum_value + sum_exp);
        } 

        Exponential(exp_vector, softmax_result);

        break;
      } 
      case SPAN_MASK: {
        CDCOUT(sc_time_stamp()  << name() << " case SPAN_MASK" << endl, kDebugLevel);
        AuxMemType::aux_req_t  attn_span_reg;

        #pragma hls_unroll yes
        for (unsigned i = 0; i < N; i++) { 
          attn_span_reg.addr[i] = base_attn_span + vector_index*N + i;
          attn_span_reg.valids[i] = 1;
        }
        attn_span_reg.type.val = CLITYPE_T::LOAD;
        attn_span_req.Push(attn_span_reg);
        break;
      }
      case SPAN_MASK2: {
        CDCOUT(sc_time_stamp()  << name() << " case SPAN_MASK2" << endl, kDebugLevel);
        spec::VectorType  attn_span_rsp_reg;
        attn_span_rsp_reg  = attn_span_rsp.Pop();   
       
        spec::ActVectorType attn_span_vector, vtmp;
        Adpfloat2Fixed(attn_span_rsp_reg, attn_span_vector, adpbias_attn_span);

        EMul (softmax_result, attn_span_vector, vtmp);
        out_data = vtmp; 
        break;
      }
      case OUT: {
        CDCOUT(sc_time_stamp()  << name() << " case OUT" << endl, kDebugLevel);
        CDCOUT(sc_time_stamp()  << name() << " DUT - vector_index: " << vector_index << endl, kDebugLevel);
        CDCOUT(sc_time_stamp()  << name() << " DUT - timesetp_Index: " << timestep_index << endl, kDebugLevel); 
        spec::VectorType out_reg; 
        Fixed2Adpfloat (out_data, out_reg, adpbias_act1);
        act_out_vec.Push(out_reg);               
        break;
      }
      case NEXT: {
        CDCOUT(sc_time_stamp()  << name() << " case NEXT" << endl, kDebugLevel);
        break;
      }
      default: {
        break;
      }
    }
  }
  
  void UpdateFSM() {
    FSM next_state;
    switch (state) {
      case IDLE: {
        if (is_start) {
          ResetCounter();
          next_state = FINDMAX;
        }
        else {
          next_state = IDLE;
        }
        break;
      }
      case FINDMAX: {
        next_state = FINDMAX2;
        break;
      }
      case FINDMAX2: {
        bool is_end = 0;
        UpdateVectorCounter(is_end);
        if (is_end) {
          next_state = SFM1;
        }
        else {
          next_state = FINDMAX;
        }
        break;
      }
      case SFM1: {
        next_state = SFM1b;
        break;        
      }
      case SFM1b: { 
        bool is_end = 0;
        UpdateVectorCounter(is_end);
        if (is_end) {
          next_state = SFM2;
        }
        else {
          next_state = SFM1;
        }
        break;
      }
      case SFM2: {
        next_state = SFM2b;
        break;        
      }
      case SFM2b: {
        next_state = SPAN_MASK;
        break;
      }
      case SPAN_MASK: {
        next_state = SPAN_MASK2;
        break;
      }
      case SPAN_MASK2: {
        next_state = OUT;
        break;
      }
      case OUT: { 
        bool is_end = 0;
        UpdateVectorCounter(is_end);
        if (is_end) {
          next_state = NEXT;
        }
        else {
          next_state = SFM2;
        }
        break;
      }
      case NEXT: {
        ResetSoftmax();
        // Move to next timestep
        bool is_end = 0;
        UpdateTimestepCounter(is_end);
        if (is_end) {
          is_start = 0;
          next_state = IDLE;
          CDCOUT(sc_time_stamp()  <<  name() << " SMax Finish" << endl, kDebugLevel);
          done.Push(1);    
        }
        else {
          next_state = FINDMAX;
        }
        break;
      }
      default: {
        next_state = IDLE;
        break;
      }
      
    }      
    state = next_state;
  }
  

  void SMaxRun() {
    Reset();

    #pragma hls_pipeline_init_interval 3 
    while(1) {
      RunFSM();
      if (is_start == 0) {
        CheckStart();
      }
      UpdateFSM();
      wait();
    }
  }
};
#endif


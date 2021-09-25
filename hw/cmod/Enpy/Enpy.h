#ifndef __ENPY_H__
#define __ENPY_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_connections.h>
#include <nvhls_vector.h>
#include <nvhls_module.h>
#include "../include/Spec.h"
#include "../DecodeTop/DecodeTop.h"
#include "../include/PPU.h"

SC_MODULE(Enpy)
{
 public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef DecodeTop MaskMemType; 
  typedef spec::IndexType IndexType;
  IndexType base_input_reg;   
  static const int N = spec::N; 
  static const int kDebugLevel = 0;

  spec::ActScalarType sum_exp, sum_xexp, maximum_value, enpy_result, enpy_threshold;

  Connections::In<bool> start;
  Connections::In<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<spec::EnpyConfig> enpy_config;
  Connections::In<spec::VectorType>  act_rsp; //Activation response from DecodeTop

  Connections::Out<MaskMemType::mask_req_t>  mask_rd_req; // To DecodeTop for mask read requests
  Connections::Out<NVUINT2> enpy_status;
  Connections::Out<spec::ActScalarType> enpy_val_out;
  Connections::Out<bool> done;

  // Constructor
  SC_HAS_PROCESS(Enpy);
  Enpy(sc_module_name name_) : sc_module(name_) {

    SC_THREAD(EnpyRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);  
  }

  bool is_start, en_config_ok, input_ok, gb_config_ok;

  enum FSM {
    IDLE, FINDMAX, FINDMAX2, RSUM, RSUMb, EVAL, NEXT
  };
  FSM state;

  spec::GBControlConfig gbcontrol_config_tmp;
  spec::EnpyConfig enpy_config_tmp;
  spec::InputBufferConfig input_buffer_config_tmp;
 
  NVUINT8   num_vector;
  spec::AdpfloatBiasType adpbias_act1;

  NVUINT8 vector_index;

  void Reset() {
    state = IDLE;
    is_start = 0;
    input_ok = 0;
    gb_config_ok = 0;
    en_config_ok = 0;
    base_input_reg = 0;
    num_vector      = 1;
    vector_index = 0;
    adpbias_act1    = 0;
    ResetPorts();
    ResetEnpy();
  }
  
  void ResetPorts() { 
    start.Reset();
    done.Reset();
    act_rsp.Reset();
    input_buffer_config.Reset();
    gbcontrol_config.Reset();
    enpy_config.Reset();
    enpy_status.Reset();
    enpy_val_out.Reset();
    mask_rd_req.Reset();
  }
 
  void ResetEnpy() {
    sum_exp = 0;
    sum_xexp = 0;
    maximum_value = spec::kActWordMin;
  }
 
  void ResetCounter() {
    vector_index      = 0;
  }

  NVUINT8 GetVectorIndex() const {
    return vector_index;
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

  void UpdateMax(const spec::ActVectorType activation_vector) {
    spec::ActScalarType new_max = spec::kActWordMin;    // should not be a reg

    #pragma hls_unroll yes 
    for (int i=0; i< N; i++){
      if (activation_vector[i] > new_max) {
        new_max = activation_vector[i]; 
      }
    }
       
    if (new_max > maximum_value) {
      maximum_value = new_max;
    }
  }
 
  void CheckStart() {
    bool start_reg;
    if (start.PopNB(start_reg) && gb_config_ok && en_config_ok && input_ok) {
      is_start = 1;
      CDCOUT(sc_time_stamp()  << name() << " Enpy Start !!!" << endl, kDebugLevel);
    }
  }

  void RunFSM() {

    if (gbcontrol_config.PopNB(gbcontrol_config_tmp)) {
      num_vector = gbcontrol_config_tmp.num_vector;
      adpbias_act1 = gbcontrol_config_tmp.adpbias_act1;
      CDCOUT(sc_time_stamp()  << name() << " DUT - num_vector is: " << num_vector << endl, kDebugLevel);
      gb_config_ok = 1;
    }
    vector_index = GetVectorIndex();

    if (enpy_config.PopNB(enpy_config_tmp)) {
      enpy_threshold = enpy_config_tmp.enpy_threshold;
      en_config_ok = 1;
    }

    if (input_buffer_config.PopNB(input_buffer_config_tmp)) {
       base_input_reg = input_buffer_config_tmp.base_input[0];
       input_ok = 1;
    }

    switch (state) {
      case IDLE: {
        ResetEnpy();
        break;
      }
      case FINDMAX: {
        CDCOUT(sc_time_stamp()  << name() << " case FINDMAX" << endl, kDebugLevel);
        MaskMemType::mask_req_t mask_reg;
        mask_reg.type.val = CLITYPE_T::LOAD;
        mask_reg.addr[0] = base_input_reg + vector_index;
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
      case RSUM: {
        CDCOUT(sc_time_stamp()  << name() << " case RSUM" << endl, kDebugLevel);
        MaskMemType::mask_req_t mask_reg;
        mask_reg.type.val = CLITYPE_T::LOAD;
        mask_reg.addr[0] = base_input_reg + vector_index;
        mask_reg.valids[0] = 1;
        mask_rd_req.Push(mask_reg);  
        break;
      }    
      case RSUMb: {
        CDCOUT(sc_time_stamp()  << name() << " case RSUMb" << endl, kDebugLevel);
        // Receive Rsp        
        spec::VectorType act_reg;
        act_reg = act_rsp.Pop();
        CDCOUT(sc_time_stamp()  << name() << " RSUMb popped -----------" << endl, kDebugLevel);

        spec::ActScalarType tmp_sum = 0;
        spec::ActScalarType tmp_xsum = 0;
 
        spec::ActVectorType exp_vector(act_reg.to_rawbits());
        spec::ActVectorType exp_vector_max, xexp_vector_max;
 
        #pragma hls_unroll yes
        for (int i = 0; i < N; i++) {
           exp_vector_max[i] = exp_vector[i] - maximum_value;
           xexp_vector_max[i] = exp_vector[i] * exp_vector_max[i];
        }        
        Exponential(exp_vector_max, exp_vector_max);
        Exponential(xexp_vector_max, xexp_vector_max);
        
        #pragma hls_unroll yes
        for (int i = 0; i < N; i++) {
          tmp_sum += exp_vector_max[i];
          tmp_xsum += xexp_vector_max[i];
        }             
        sum_exp += tmp_sum;
        sum_xexp += tmp_xsum;
        break;
      } 
      case EVAL: { 
        CDCOUT(sc_time_stamp()  << name() << " case EVAL" << endl, kDebugLevel);

        spec::ActScalarType tmp_div;
        spec::ActScalarType log_tmp;
        NVUINT2 enpy_status_reg;

        LogN(sum_exp, log_tmp); 
        tmp_div = sum_xexp / sum_exp;
        enpy_result = log_tmp + maximum_value - tmp_div;       
 
        //Scaling up in case enpy result is very small 
        //CDCOUT(sc_time_stamp()  << name() << " DUT - enpy value1 is: " << enpy_result << endl, kDebugLevel);
        //enpy_result = enpy_result << 10;    
        enpy_val_out.Push(enpy_result);
        //CDCOUT(sc_time_stamp()  << name() << " DUT - enpy value2 is: " << enpy_result << endl, kDebugLevel);

        if   (enpy_result <= enpy_threshold) 
             enpy_status.Push(3);
        else enpy_status.Push(2); 

        break;
      }
      case NEXT: {
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
          next_state = RSUM;
        }
        else {
          next_state = FINDMAX;
        }
        break;
      }
      case RSUM: {
        next_state = RSUMb;
        break;        
      }
      case RSUMb: { 
        bool is_end = 0;
        UpdateVectorCounter(is_end);
        if (is_end) {
          next_state = EVAL;
        }
        else {
          next_state = RSUM;
        }
        break;
      }
      case EVAL: {
        next_state = NEXT;
        break;        
      }
      case NEXT: {
        is_start = 0;
        next_state = IDLE;
        CDCOUT(sc_time_stamp()  <<  name() << " Enpy Finish" << endl, kDebugLevel);
        done.Push(1);    
        break;
      }
      default: {
        next_state = IDLE;
        break;
      }
      
    }      
    state = next_state;
  }
  

  void EnpyRun() {
    Reset();

    #pragma hls_pipeline_init_interval 4 
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


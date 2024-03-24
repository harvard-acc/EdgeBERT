
#ifndef __ELEMADD_H__
#define __ELEMADD_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_connections.h>
#include <nvhls_vector.h>
#include <nvhls_module.h>
#include "../include/Spec.h"
#include "../DecodeTop/DecodeTop.h"
#include "../include/PPU.h"

SC_MODULE(ElemAdd)
{
 public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef DecodeTop MaskMemType; 
  typedef spec::IndexType IndexType;
  IndexType base_input_reg[2];   
  static const int N = spec::N; 
  static const int kDebugLevel = 0;

  spec::ActVectorType out_vector, x_vector[2];

  Connections::In<bool> start;
  Connections::In<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<spec::VectorType>  act_rsp; //Activation response from DecodeTop

  Connections::Out<MaskMemType::mask_req_t>  mask_rd_req; // To DecodeTop for mask read requests
  Connections::Out<spec::VectorType>  act_out_vec; // To be stored in DecodeTop
  Connections::Out<bool> done;

  // Constructor
  SC_HAS_PROCESS(ElemAdd);
  ElemAdd(sc_module_name name_) : sc_module(name_) {

    SC_THREAD(ElemAddRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);  
  }

  bool is_start, input_ok, gb_config_ok;

  enum FSM {
    IDLE, REQ, RSP, REQ2, RSP2, EADD, SEND, NEXT
  };
  FSM state;

  spec::GBControlConfig gbcontrol_config_tmp;
  spec::InputBufferConfig input_buffer_config_tmp;
 
  NVUINT8   num_vector;
  NVUINT8   num_timestep;
  spec::AdpfloatBiasType adpbias_act[2], adpbias_out;

  NVUINT8 vector_index;
  NVUINT8 timestep_index;

  void Reset() {
    state = IDLE;
    is_start = 0;
    input_ok = 0;
    gb_config_ok = 0;
    base_input_reg[0] = 0;
    base_input_reg[1] = 0;
    num_vector      = 1;
    num_timestep    = 1;
    vector_index = 0;
    timestep_index = 0;
    adpbias_act[0]   = 0;
    adpbias_act[1]   = 0;
    adpbias_out    = 0;
    ResetPorts();
  }
  
  void ResetPorts() { 
    start.Reset();
    done.Reset();
    act_rsp.Reset();
    input_buffer_config.Reset();
    gbcontrol_config.Reset();
    mask_rd_req.Reset();
    act_out_vec.Reset();
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

  void CheckStart() {
    bool start_reg;
    if (start.PopNB(start_reg) && gb_config_ok && input_ok) {
      is_start = 1;
      CDCOUT(sc_time_stamp()  << name() << " ElemAdd Start !!!" << endl, kDebugLevel);
    }
  }

  void RunFSM() {
    if (gbcontrol_config.PopNB(gbcontrol_config_tmp)) {
      num_vector = gbcontrol_config_tmp.num_vector;
      num_timestep = gbcontrol_config_tmp.num_timestep;
      adpbias_act[0] = gbcontrol_config_tmp.adpbias_act1;
      adpbias_act[1] = gbcontrol_config_tmp.adpbias_act2;
      adpbias_out    = gbcontrol_config_tmp.adpbias_act3;
      CDCOUT(sc_time_stamp()  << name() << " DUT - num_vector is: " << num_vector << endl, kDebugLevel);
      CDCOUT(sc_time_stamp()  << name() << " DUT - num_timestep is: " << num_timestep << endl, kDebugLevel);
      gb_config_ok = 1;
      CDCOUT(sc_time_stamp()  << name() << " gb_config_ok " << gb_config_ok << endl, kDebugLevel);
    }
    vector_index = GetVectorIndex();
    timestep_index = GetTimestepIndex();
    //CDCOUT(sc_time_stamp()  << name() << " DUT - vector_index: " << vector_index << endl, kDebugLevel);

    if (input_buffer_config.PopNB(input_buffer_config_tmp)) {
       base_input_reg[0] = input_buffer_config_tmp.base_input[0];
       base_input_reg[1] = input_buffer_config_tmp.base_input[1];
       input_ok = 1;
       CDCOUT(sc_time_stamp()  << name() << " input_ok " << input_ok << " and base_input_reg[0]: " << base_input_reg[0] << " and base_input_reg[1]: " << base_input_reg[1] <<  endl, kDebugLevel);
    }

    switch (state) {
      case IDLE: {
        break;
      }
      case REQ: {
        CDCOUT(sc_time_stamp()  << name() << " case REQ" << endl, kDebugLevel);
        MaskMemType::mask_req_t mask_reg;

        mask_reg.type.val = CLITYPE_T::LOAD;
        mask_reg.addr[0] = base_input_reg[0] + vector_index + timestep_index*num_vector;
        //CDCOUT(sc_time_stamp()  << name() << " ElemAdd mask_reg.addr[0]: " << mask_reg.addr[0] << endl, 0);
        mask_reg.valids[0] = 1;
        mask_rd_req.Push(mask_reg);  
        break;
      }
      case RSP: {
        CDCOUT(sc_time_stamp()  << name() << " case RSP" << endl, kDebugLevel);
        spec::VectorType act_reg;

        act_reg = act_rsp.Pop(); 
        Adpfloat2Fixed(act_reg,  x_vector[0], adpbias_act[0]);
 
        break; 
      }
      case REQ2: {
        CDCOUT(sc_time_stamp()  << name() << " case REQ2" << endl, kDebugLevel);
        MaskMemType::mask_req_t mask_reg;

        mask_reg.type.val = CLITYPE_T::LOAD;
        mask_reg.addr[0] = base_input_reg[1] + vector_index + timestep_index*num_vector;
        mask_reg.valids[0] = 1;
        mask_rd_req.Push(mask_reg);  
        break;
      }
      case RSP2: {
        CDCOUT(sc_time_stamp()  << name() << " case RSP2" << endl, kDebugLevel);
        spec::VectorType act_reg;

        act_reg = act_rsp.Pop(); 
        Adpfloat2Fixed(act_reg,  x_vector[1], adpbias_act[1]);
 
        break; 
      }
      case EADD: {
        CDCOUT(sc_time_stamp()  << name() << " case EADD" << endl, kDebugLevel);
        EAdd(x_vector[0], x_vector[1], out_vector);
        break;
      }
      case SEND: {
        CDCOUT(sc_time_stamp()  << name() << " case SEND" << endl, kDebugLevel);
        spec::VectorType out_reg; 
        Fixed2Adpfloat (out_vector, out_reg, adpbias_out);
        act_out_vec.Push(out_reg);  
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
          next_state = REQ;
        }
        else {
          next_state = IDLE;
        }
        break;
      }
      case REQ: {
        next_state = RSP;
        break;
      }
      case RSP: {
        next_state = REQ2;
        break;
      }
      case REQ2: {
        next_state = RSP2;
        break;
      }
      case RSP2: {
        next_state = EADD;
        break;
      }
     case EADD: {
        next_state = SEND:
        break;
     }
     case SEND: { 
        bool is_end = 0;
        UpdateVectorCounter(is_end);
        if (is_end) {
          next_state = NEXT;
        }
        else {
          next_state = REQ;
        }
        break;
      }
      case NEXT: {
        // Move to next timestep
        bool is_end = 0;
        UpdateTimestepCounter(is_end);
        if (is_end) {
          is_start = 0;
          next_state = IDLE;
          CDCOUT(sc_time_stamp()  <<  name() << " ElemAdd Finish" << endl, kDebugLevel);
          done.Push(1);    
        }
        else {
          next_state = REQ;
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
  

  void ElemAddRun() {
    Reset();

    #pragma hls_pipeline_init_interval 1 
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


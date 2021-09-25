
#ifndef __DECODE_ENPY_H__
#define __DECODE_ENPY_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>
#include "../include/Spec.h"

SC_MODULE(Decode_Enpy)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  static const int N = spec::N;   
  
  typedef spec::IndexType IndexType;
  IndexType base_addr;
  IndexType offset;
  NVUINT8   num_vector;

  NVUINT12 bank_ctrs[N];

  bool gbcontrolconfig_ready, base_input_ready, base_offset_ready;

  static const int kDebugLevel = 0;

  Connections::In<IndexType> base_input;
  Connections::In<IndexType> base_offset;
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<spec::mask_rsp_t>   mask_rsp;
  Connections::Out<spec::input_req_t> out_req;

  SC_HAS_PROCESS(Decode_Enpy);
  Decode_Enpy(sc_module_name name_) : sc_module(name_) {

    SC_THREAD (RunInputAddrGen);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

  }
  
  void ResetVars() {
    gbcontrolconfig_ready = 0;
    base_input_ready = 0;
    base_offset_ready = 0;
    //basic_ready=0;
  }

  void ResetCtrs() {
    #pragma hls_unroll yes    
    for (int i = 0; i < N; i++) {
      bank_ctrs[i] = 0;
    }   
  }

  void RunInputAddrGen() {
    mask_rsp.Reset();
    base_input.Reset();
    base_offset.Reset();
    gbcontrol_config.Reset();
    out_req.Reset();
   
    base_addr = 0;
    offset = 0;
    NVUINT12 count = 0;

    ResetCtrs();
    ResetVars();

    wait();
    #pragma hls_pipeline_init_interval 1
    while (1) { 

      spec::GBControlConfig gbcontrol_config_reg;
      if (gbcontrol_config.PopNB(gbcontrol_config_reg)) {
         num_vector = gbcontrol_config_reg.num_vector;
         gbcontrolconfig_ready = 1;
      }

      IndexType  base_input_reg;
      if (base_input.PopNB(base_input_reg)) {
          base_addr = base_input_reg;
          base_input_ready = 1;
      } 

      IndexType offset_reg;
      if (base_offset.PopNB(offset_reg)) {
         offset = offset_reg;
         base_offset_ready = 1;
      }

      //if (gbcontrolconfig_ready && base_input_ready && base_offset_ready) {
      //   basic_ready = 1;
      //}

      //if (basic_ready) {
        //CDCOUT(sc_time_stamp()  << name() << " - DUT: Decode: Enpy mode" << endl, 0);
      NVUINT10  M_Enpy = 2*num_vector;
      NVUINT8 num_vector_1 = num_vector-1;

      ResetCtrs();

      count = 0;

      #pragma hls_pipeline_init_interval 1
      for (int i = 0; i < M_Enpy; i++) {
         
        spec::mask_rsp_t in_mask_reg = mask_rsp.Pop();

        spec::input_req_t req_reg;
        req_reg.type.val = CLITYPE_T::LOAD;

        #pragma hls_unroll yes    
        for (int i = 0; i < N; i++) {
          req_reg.addr[i] = N*bank_ctrs[i] + i + base_addr - offset;
          cout << sc_time_stamp() << name() <<  " - DUT: Decode/enpy - input req address from decoder: " << req_reg.addr[i] << endl;
          if (in_mask_reg.data[0][i] == 1) {
            bank_ctrs[i] += 1;
            req_reg.valids[i] = 1;
          } else {
            req_reg.valids[i] = 0;
          }
        }
        out_req.Push(req_reg);
       
        if (count == num_vector_1) {
           count = 0;
           ResetCtrs();
        }
        else count += 1;

        wait();
      } // for loop sequential         
      ResetVars();
      //}
      wait(); 
    } // while(1)

  }

};


#endif

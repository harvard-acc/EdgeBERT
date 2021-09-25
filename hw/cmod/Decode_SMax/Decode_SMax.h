
#ifndef __DECODE_SMAX_H__
#define __DECODE_SMAX_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>
#include "../include/Spec.h"

SC_MODULE(Decode_SMax)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  static const int N = spec::N;   
  
  typedef spec::IndexType IndexType;
  IndexType base_addr;
  IndexType offset;
  NVUINT8   num_vector;
  NVUINT8   num_timestep;

  NVUINT12 bank_ctrs[N];
  NVUINT12 record_ctrs[N];

  bool gbcontrolconfig_ready, base_input_ready, base_offset_ready, basic_ready;

  static const int kDebugLevel = 0;

  Connections::In<IndexType> base_input;
  Connections::In<IndexType> base_offset;
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<spec::mask_rsp_t>   mask_rsp;
  Connections::Out<spec::input_req_t> out_req;

  SC_HAS_PROCESS(Decode_SMax);
  Decode_SMax(sc_module_name name_) : sc_module(name_) {

    SC_THREAD (RunInputAddrGen);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

  }
  
  void ResetVars() {
    gbcontrolconfig_ready = 0;
    base_input_ready = 0;
    base_offset_ready = 0;
    basic_ready =0;
  }

  void ResetCtrs() {
    #pragma hls_unroll yes    
    for (int i = 0; i < N; i++) {
      bank_ctrs[i] = 0;
      record_ctrs[i] = 0;
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
    NVUINT2 k = 0;

    ResetCtrs();
    ResetVars();

    wait();
    #pragma hls_pipeline_init_interval 1
    while (1) { 

      spec::GBControlConfig gbcontrol_config_reg;
      if (gbcontrol_config.PopNB(gbcontrol_config_reg)) {
         num_vector = gbcontrol_config_reg.num_vector;
         num_timestep = gbcontrol_config_reg.num_timestep;
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


      if (base_input_ready && base_offset_ready && gbcontrolconfig_ready) {
         basic_ready =1; 
      }

      if (basic_ready)  { 
         CDCOUT(sc_time_stamp()  << name() << " - DUT: Decode: SoftMax mode" << endl, 0);
         NVUINT18 M_SMax = 3*num_vector*num_timestep;         
         NVUINT8 count_num = num_vector-1;

         ResetCtrs();

         count = 0;
         k = 0;
         #pragma hls_pipeline_init_interval 1
         for (int i = 0; i < M_SMax; i++) {
  
           spec::mask_rsp_t in_mask_reg = mask_rsp.Pop();

           spec::input_req_t req_reg;
           req_reg.type.val = CLITYPE_T::LOAD;

           #pragma hls_unroll yes    
           for (int i = 0; i < N; i++) {
             req_reg.addr[i] = N*bank_ctrs[i] + i + base_addr - offset;
             //cout << sc_time_stamp() << name() <<  " - DUT: Decode/SMax - input req address from decoder: " << req_reg.addr[i] << " with base_addr: " << base_addr << " with offset: " << offset << " with bank_ctrs[i]: " << bank_ctrs[i] << endl;
             if (in_mask_reg.data[0][i] == 1) {
               bank_ctrs[i] += 1;
               req_reg.valids[i] = 1;
             } else {
               req_reg.valids[i] = 0;
             }
           }
           out_req.Push(req_reg);

           if (count == count_num) {
              count = 0;
              k += 1;
              if (k == 3) {
                 k == 0;

                 #pragma hls_unroll yes    
                 for (int i = 0; i < N; i++) {
                     record_ctrs[i] = bank_ctrs[i];
                 }
              }
              #pragma hls_unroll yes 
              for (int i = 0; i < N; i++) {
                  bank_ctrs[i] = record_ctrs[i];
              }
           }
           else count += 1;

           wait();
         } // for loop sequential         
         ResetVars();
      } 

      wait(); 
    } // while(1)

  }

};


#endif

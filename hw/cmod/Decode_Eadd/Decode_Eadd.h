
#ifndef __DECODE_EADD_H__
#define __DECODE_EADD_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>
#include "../include/Spec.h"

SC_MODULE(Decode_Eadd)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  static const int N = spec::N;   
  
  typedef spec::IndexType IndexType;
  //IndexType base_addr;
  IndexType base_addr_eadd[2];
  IndexType offset;
  NVUINT8   num_vector;
  NVUINT8   num_timestep;
  NVUINT18  M_Eadd;
  NVUINT12 bank_ctrs[N];
  NVUINT12 bank_ctrs2[N];

  bool gbcontrolconfig_ready, inputbufferconfig_ready, base_offset_ready, basic_ready;

  static const int kDebugLevel = 0;

  //Connections::In<IndexType> base_input;
  Connections::In<IndexType> base_offset;
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations used only during ElemAdd
  Connections::In<spec::mask_rsp_t>   mask_rsp;
  Connections::Out<spec::input_req_t> out_req;

  SC_HAS_PROCESS(Decode_Eadd);
  Decode_Eadd(sc_module_name name_) : sc_module(name_) {

    SC_THREAD (RunInputAddrGen);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

  }
  
  void ResetVars() {
    gbcontrolconfig_ready = 0;
    inputbufferconfig_ready = 0;
    //base_input_ready = 0;
    base_offset_ready = 0;
    basic_ready =0;
  }

  void ResetCtrs() {
    #pragma hls_unroll yes    
    for (int i = 0; i < N; i++) {
      bank_ctrs[i] = 0;
      bank_ctrs2[i] = 0;
    }   
  }

  void RunInputAddrGen() {
    mask_rsp.Reset();
    //base_input.Reset();
    base_offset.Reset();
    gbcontrol_config.Reset();
    input_buffer_config.Reset();
    out_req.Reset();
  
    #pragma hls_unroll yes    
    for (int i = 0; i < 2; i++) base_addr_eadd[i] = 0;
 
    //base_addr = 0;
    offset = 0;

    ResetCtrs();
    ResetVars();

    wait();
    #pragma hls_pipeline_init_interval 1
    while (1) { 

      spec::GBControlConfig gbcontrol_config_reg;
      if (gbcontrol_config.PopNB(gbcontrol_config_reg)) {
         num_vector = gbcontrol_config_reg.num_vector;
         num_timestep = gbcontrol_config_reg.num_timestep;
         M_Eadd = 2*num_vector*num_timestep;
         gbcontrolconfig_ready = 1;
         cout << "check gbcontrol" << endl;
      }

      spec::InputBufferConfig input_buffer_config_reg;
      if (input_buffer_config.PopNB(input_buffer_config_reg)) {
         #pragma hls_unroll yes    
         for (int i = 0; i < 2; i++) {
           base_addr_eadd[i] = input_buffer_config_reg.base_input[i];
           CDCOUT(sc_time_stamp()  << name() << " base_addr_eadd: " << base_addr_eadd[i] << endl, 0);
         }  
         inputbufferconfig_ready = 1;
         cout << "check input_buffer_config" << endl;
      }

      /*IndexType  base_input_reg;
      if (base_input.PopNB(base_input_reg)) {
          base_addr = base_input_reg;
          base_input_ready = 1;
          cout << "check base_input" << endl;
      } */ 

      IndexType offset_reg;
      if (base_offset.PopNB(offset_reg)) {
         offset = offset_reg;
         base_offset_ready = 1;
         cout << "check offset" << endl;
      }


      if (inputbufferconfig_ready && base_offset_ready && gbcontrolconfig_ready) {
         basic_ready =1; 
      }

      if (basic_ready)  { 
         CDCOUT(sc_time_stamp()  << name() << " - DUT ElemAdd mode" << endl, 0);
     
         ResetCtrs();

         #pragma hls_pipeline_init_interval 1
         for (unsigned int i = 0; i < M_Eadd; i++) {

           NVUINT18  M_Eadd_reg = i;
           NVUINT1 lsb = nvhls::get_slc<1>(M_Eadd_reg, 0); 

           spec::mask_rsp_t in_mask_reg = mask_rsp.Pop();

           spec::input_req_t req_reg;
           req_reg.type.val = CLITYPE_T::LOAD;

           if (lsb == 0) {
             #pragma hls_unroll yes    
             for (int i = 0; i < N; i++) {
               req_reg.addr[i] = N*bank_ctrs[i] + i + base_addr_eadd[0] - offset;
               //cout << sc_time_stamp() << name() <<  " - DUT: Decode/ElemAdd0 - input req address from decoder: " << req_reg.addr[i] << " with offset: " << offset << " and bank_ctrs: " << bank_ctrs[i] << endl;
               if (in_mask_reg.data[0][i] == 1) {
                 bank_ctrs[i] += 1;
                 req_reg.valids[i] = 1;
               } else {
                 req_reg.valids[i] = 0;
               }
             } // for loop
             out_req.Push(req_reg);
           } else {
             #pragma hls_unroll yes    
             for (int i = 0; i < N; i++) {
               req_reg.addr[i] = N*bank_ctrs2[i] + i + base_addr_eadd[1] - offset;
               //cout << sc_time_stamp() << name() <<  " - DUT: Decode/ElemAdd1 - input req address from decoder: " << req_reg.addr[i] << endl;
               if (in_mask_reg.data[0][i] == 1) {
                 bank_ctrs2[i] += 1;
                 req_reg.valids[i] = 1;
               } else {
                 req_reg.valids[i] = 0;
               }
             } // for loop
             out_req.Push(req_reg);
           }
           wait();
         } // for loop sequential       
         ResetVars();
      } 

      wait(); 
    } // while(1)

  }

};


#endif

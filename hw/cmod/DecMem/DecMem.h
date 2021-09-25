/*
 * DecMem.h
 */

#ifndef __DECMEM_H__
#define __DECMEM_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>

#include <string>
#include <one_hot_to_bin.h>

#include "../include/Spec.h"
#include "../include/Xbar_Dec.h"
#include "../DecMemCore/DecMemCore.h"

SC_MODULE(DecMem)
{
 public: 
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;

  Connections::In<bool> flip_mem_input;
  Connections::In<bool> flip_mem_mask;
 
  Connections::In<mask_req_t>    mask_mem_req[5];
  Connections::In<input_req_t>   input_mem_req[4];
  Connections::Out<mask_rsp_t>   mask_mem_rsp[2];
  Connections::Out<input_rsp_t>  input_mem_rsp[2];

  Connections::Combinational<mask_req_t>  mask_req_inter;
  Connections::Combinational<input_req_t> input_req_inter;
  Connections::Combinational<mask_rsp_t>  mask_rsp_inter;
  Connections::Combinational<input_rsp_t> input_rsp_inter;

  Xbar_Dec<mask_req_t, 5, 1, 5, 1>  arbxbar_mask_inst;
  Xbar_Dec<input_req_t, 4, 1, 4, 1> arbxbar_input_inst;
  DecMemCore mem_inst;

  SC_HAS_PROCESS(DecMem);
  DecMem(sc_module_name name_) : 
    arbxbar_mask_inst("arbxbar_mask_inst"),
    arbxbar_input_inst("arbxbar_input_inst"),
    mem_inst("mem_inst")
  {
    arbxbar_mask_inst.clk(clk); 
    arbxbar_mask_inst.rst(rst);
    for (int i = 0; i < 5; i++) {     
      arbxbar_mask_inst.data_in[i](mask_mem_req[i]);
    }   
    arbxbar_mask_inst.data_out[0](mask_req_inter);

    arbxbar_input_inst.clk(clk); 
    arbxbar_input_inst.rst(rst);
    for (int i = 0; i < 4; i++) {     
      arbxbar_input_inst.data_in[i](input_mem_req[i]);
    }   
    arbxbar_input_inst.data_out[0](input_req_inter);

    mem_inst.clk(clk);
    mem_inst.rst(rst);
    mem_inst.mask_req_inter(mask_req_inter);
    mem_inst.mask_rsp_inter(mask_rsp_inter);    
    mem_inst.input_req_inter(input_req_inter);
    mem_inst.input_rsp_inter(input_rsp_inter); 

    SC_THREAD (MergeRspRun_Input);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 

    SC_THREAD (MergeRspRun_Mask);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 
  }

  void MergeRspRun_Input() {
    flip_mem_input.Reset();
    input_rsp_inter.ResetRead();

    #pragma hls_unroll yes
    for (int i=0; i<2; i++) input_mem_rsp[i].Reset();
    
    bool mem_id = 0;
    #pragma hls_pipeline_init_interval 1
    while(1) {
      input_rsp_t rsp_reg;
      
      if (input_rsp_inter.PopNB(rsp_reg)) {
        if (mem_id == 0)  input_mem_rsp[0].Push(rsp_reg);
        else              input_mem_rsp[1].Push(rsp_reg);
      }

      bool tmp;
      if (flip_mem_input.PopNB(tmp)) {
        //CDCOUT(sc_time_stamp() << name() << " flip_mem_input from DecMem: " << tmp << endl, 0);
        //cout << sc_time_stamp() << " flip_mem_input from DecMem: " << tmp << endl;
        mem_id = tmp;
      }       
      wait();
    }
  }

  void MergeRspRun_Mask() {
    flip_mem_mask.Reset();
    mask_rsp_inter.ResetRead();

    #pragma hls_unroll yes
    for (int i=0; i<2; i++) mask_mem_rsp[i].Reset();
    
    bool mem_id = 0;
    #pragma hls_pipeline_init_interval 1
    while(1) {
      mask_rsp_t rsp_reg;
      
      if (mask_rsp_inter.PopNB(rsp_reg)) {
        //cout << sc_time_stamp() << name() <<  " DecMem popped mask_rsp_inter : " << rsp_reg.data[0] << " with valid: " << rsp_reg.valids[0]  << endl;
        if (mem_id == 0)  mask_mem_rsp[0].Push(rsp_reg);
        else              mask_mem_rsp[1].Push(rsp_reg);
      }

      bool tmp;
      if (flip_mem_mask.PopNB(tmp)) {
        //cout << sc_time_stamp() << " flip_mem_mask from DecMem: " << tmp << endl;
        mem_id = tmp;
      }       
      wait();
    }
  } 

};

#endif


/*
 * DecodeTop.h
 */

#ifndef __DECODETOP_H__
#define __DECODETOP_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>

#include <string>
#include <one_hot_to_bin.h>

#include "../include/Spec.h"
#include "../DecMem/DecMem.h"
#include "../Decode/Decode.h"

SC_MODULE(DecodeTop)
{
 public: 
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;
  typedef spec::IndexType IndexType;

  Connections::In<NVUINT6> reset_mode;
  Connections::In<spec::MatrixConfig> mat_config;
  Connections::In<IndexType> base_input;
  Connections::In<IndexType> base_offset;
  Connections::In<bool> flip_mem_input;
  Connections::In<bool> flip_mem_mask;
  Connections::In<bool> flip_dec_out;
  Connections::In<mask_req_t>    mask_mem_req[5];
  Connections::In<input_req_t>   input_mem_req[3];
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<spec::InputBufferConfig> input_buffer_config;

  Connections::Out<mask_rsp_t>   axi_mask_rsp_out;
  Connections::Out<input_rsp_t>  axi_input_rsp_out;
  Connections::Out<spec::VectorType>  vec_out;
  Connections::Out<spec::VectorType>  vec_out_to_gb;

  Connections::Combinational<input_rsp_t>  input_rsp_inter;
  Connections::Combinational<mask_rsp_t>   mask_rsp_inter;
  Connections::Combinational<input_req_t> out_req_inter;
  Connections::Combinational<spec::VectorType>  vec_out_inter;


  Decode decode_inst;
  DecMem decmem_inst;

  SC_HAS_PROCESS(DecodeTop);
  DecodeTop(sc_module_name name_) : 
    decode_inst("decode_inst"),
    decmem_inst("decmem_inst")
  {
    decode_inst.clk(clk);
    decode_inst.rst(rst);
    decode_inst.reset_mode(reset_mode);
    decode_inst.mat_config(mat_config);
    decode_inst.gbcontrol_config(gbcontrol_config);
    decode_inst.input_buffer_config(input_buffer_config);
    decode_inst.base_input(base_input);
    decode_inst.base_offset(base_offset);
    decode_inst.input_rsp(input_rsp_inter);
    decode_inst.mask_rsp(mask_rsp_inter);    
    decode_inst.out_req(out_req_inter);
    decode_inst.vec_out(vec_out_inter);

    decmem_inst.clk(clk);
    decmem_inst.rst(rst);
    decmem_inst.flip_mem_input(flip_mem_input);
    decmem_inst.flip_mem_mask(flip_mem_mask);
    for (int i = 0; i < 5; i++) decmem_inst.mask_mem_req[i](mask_mem_req[i]);  
    for (int i = 0; i < 3; i++) decmem_inst.input_mem_req[i](input_mem_req[i]);
    decmem_inst.input_mem_req[3](out_req_inter);
    decmem_inst.mask_mem_rsp[0](mask_rsp_inter);    
    decmem_inst.mask_mem_rsp[1](axi_mask_rsp_out);
    decmem_inst.input_mem_rsp[0](input_rsp_inter);    
    decmem_inst.input_mem_rsp[1](axi_input_rsp_out);

    SC_THREAD (Vec_Out_Run);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 
  }

  void Vec_Out_Run() {
    flip_dec_out.Reset();
    vec_out_inter.ResetRead();
    vec_out.Reset();
    vec_out_to_gb.Reset();
    
    bool mem_id = 0;
    #pragma hls_pipeline_init_interval 1
    while(1) {
      spec::VectorType rsp_reg;
      
      if (vec_out_inter.PopNB(rsp_reg)) {
        if (mem_id == 0)  vec_out.Push(rsp_reg);
        else              vec_out_to_gb.Push(rsp_reg);
      }

      bool tmp;
      if (flip_dec_out.PopNB(tmp)) {
        mem_id = tmp;
      }       
      wait();
    }
  }

};

#endif


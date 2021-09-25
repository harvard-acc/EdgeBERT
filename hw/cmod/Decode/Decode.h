
#ifndef __DECODE_H__
#define __DECODE_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>
#include "../include/Spec.h"
#include "../include/Xbar.h"
#include "../Decode_N0/Decode_N0.h"
#include "../Decode_N1/Decode_N1.h"
#include "../Decode_LayerNorm/Decode_LayerNorm.h"
#include "../Decode_SMax/Decode_SMax.h"
#include "../Decode_Enpy/Decode_Enpy.h"
#include "../Decode_Eadd/Decode_Eadd.h"

SC_MODULE(Decode)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  static const int N = spec::N;   
  static const int log2v = nvhls::log2_floor<N>::val; 
  
  typedef spec::IndexType IndexType;

  static const int kDebugLevel = 0;

  Connections::In<IndexType> base_input;
  Connections::In<IndexType> base_offset;
  Connections::In<spec::input_rsp_t>  input_rsp;
  Connections::In<spec::mask_rsp_t>   mask_rsp;
  Connections::Out<spec::input_req_t> out_req;
  Connections::Out<spec::VectorType>  vec_out;
  Connections::In<NVUINT6> reset_mode; //1:N0, 2:N1, 4:LN, 8:SM, 16:EN, 32:EA 
  Connections::In<spec::MatrixConfig> mat_config;
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<spec::InputBufferConfig> input_buffer_config;  //used only during ElemAdd

  Connections::Combinational<IndexType> base_input_wr[5];
  Connections::Combinational<IndexType> base_offset_wr[6];
  Connections::Combinational<spec::MatrixConfig> mat_config_wr[2];
  Connections::Combinational<spec::mask_rsp_t>   mask_rsp_wr[6];
  Connections::Combinational<spec::input_req_t> out_req_wr[6];
  Connections::Combinational<spec::GBControlConfig> gbcontrol_config_wr[4];
  Connections::Combinational<spec::InputBufferConfig> input_buffer_config_wr;
  Connections::Combinational<NVUINT6> reset_mode_wr_mask;
  Connections::Combinational<NVUINT6> reset_mode_wr_wire;

  Decode_N0 decode_n0_inst;
  Decode_N1 decode_n1_inst;
  Decode_LayerNorm decode_ln_inst;
  Decode_SMax decode_sm_inst;
  Decode_Enpy decode_en_inst;
  Decode_Eadd decode_ea_inst;
  Xbar<spec::input_req_t, 6, 1, 6, 1> arbxbar_outreq_inst;

  SC_HAS_PROCESS(Decode);
  Decode(sc_module_name name_) : sc_module(name_), 
    decode_n0_inst("decode_n0_inst"),
    decode_n1_inst("decode_n1_inst"),
    decode_ln_inst("decode_ln_inst"),
    decode_sm_inst("decode_sm_inst"),
    decode_en_inst("decode_en_inst"),
    decode_ea_inst("decode_ea_inst"),
    arbxbar_outreq_inst("arbxbar_outreq_inst")
  {
    //Decode_N0
    decode_n0_inst.clk(clk);
    decode_n0_inst.rst(rst);
    decode_n0_inst.base_input(base_input_wr[0]);
    decode_n0_inst.base_offset(base_offset_wr[0]);
    decode_n0_inst.mask_rsp(mask_rsp_wr[0]);
    decode_n0_inst.out_req(out_req_wr[0]);
    decode_n0_inst.mat_config(mat_config_wr[0]);

    //Decode_N1
    decode_n1_inst.clk(clk);
    decode_n1_inst.rst(rst);
    decode_n1_inst.base_input(base_input_wr[1]);
    decode_n1_inst.base_offset(base_offset_wr[1]);
    decode_n1_inst.mask_rsp(mask_rsp_wr[1]);
    decode_n1_inst.out_req(out_req_wr[1]);
    decode_n1_inst.mat_config(mat_config_wr[1]);

    //Decode_LayerNorm
    decode_ln_inst.clk(clk);
    decode_ln_inst.rst(rst);
    decode_ln_inst.base_input(base_input_wr[2]);
    decode_ln_inst.base_offset(base_offset_wr[2]);
    decode_ln_inst.mask_rsp(mask_rsp_wr[2]);
    decode_ln_inst.out_req(out_req_wr[2]);
    decode_ln_inst.gbcontrol_config(gbcontrol_config_wr[0]);

    //Decode_SMax
    decode_sm_inst.clk(clk);
    decode_sm_inst.rst(rst);
    decode_sm_inst.base_input(base_input_wr[3]);
    decode_sm_inst.base_offset(base_offset_wr[3]);
    decode_sm_inst.mask_rsp(mask_rsp_wr[3]);
    decode_sm_inst.out_req(out_req_wr[3]);
    decode_sm_inst.gbcontrol_config(gbcontrol_config_wr[1]);

    //Decode_Enpy
    decode_en_inst.clk(clk);
    decode_en_inst.rst(rst);
    decode_en_inst.base_input(base_input_wr[4]);
    decode_en_inst.base_offset(base_offset_wr[4]);
    decode_en_inst.mask_rsp(mask_rsp_wr[4]);
    decode_en_inst.out_req(out_req_wr[4]);
    decode_en_inst.gbcontrol_config(gbcontrol_config_wr[2]);

    //Decode_Eadd
    decode_ea_inst.clk(clk);
    decode_ea_inst.rst(rst);
    decode_ea_inst.base_offset(base_offset_wr[5]);
    decode_ea_inst.mask_rsp(mask_rsp_wr[5]);
    decode_ea_inst.out_req(out_req_wr[5]);
    decode_ea_inst.gbcontrol_config(gbcontrol_config_wr[3]);
    decode_ea_inst.input_buffer_config(input_buffer_config_wr);


    //arbxbar_outreq_inst
    arbxbar_outreq_inst.clk(clk); 
    arbxbar_outreq_inst.rst(rst);
    for (int i = 0; i < 6; i++) {     
      arbxbar_outreq_inst.data_in[i](out_req_wr[i]);
    }   
    arbxbar_outreq_inst.data_out[0](out_req);

    SC_THREAD (RunRsp);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

    SC_THREAD (RunWire);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

    SC_THREAD (MaskRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

    SC_THREAD (ValidRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);
  }
 

  void ValidRun() {
    reset_mode.Reset();
    reset_mode_wr_mask.ResetWrite();
    reset_mode_wr_wire.ResetWrite();

    #pragma hls_pipeline_init_interval 1
    while(1) {
      NVUINT6 reset_reg;
      if (reset_mode.PopNB(reset_reg)) {
         reset_mode_wr_mask.Push(reset_reg);
         reset_mode_wr_wire.Push(reset_reg);
      }
      wait(); 
    }
  }
 
  void MaskRun() {
    mask_rsp.Reset();
    reset_mode_wr_mask.ResetRead();

    #pragma hls_unroll yes
    for (int i=0; i<6; i++) {
        mask_rsp_wr[i].ResetWrite();
    } 

    NVUINT6 valids = 0;
    bool in_ready = 0;

    //wait();
    #pragma hls_pipeline_init_interval 1
    while(1) {

      NVUINT6 reset_reg;
      if (reset_mode_wr_mask.PopNB(reset_reg)) {
         #pragma hls_unroll yes 
         for (unsigned int i = 0; i < 6; i++) {
             valids[i] = reset_reg[i];  
         }  
         in_ready = 1;
      }

      spec::mask_rsp_t mask_rsp_reg;
      if (mask_rsp.PopNB(mask_rsp_reg) && in_ready) {
        #pragma hls_unroll yes    
        for (unsigned int i = 0; i < 6; i++) {
          if (valids[i] != 0) mask_rsp_wr[i].Push(mask_rsp_reg);
        }
      } 

      wait();
    }
  }


  void RunWire() {
    //mask_rsp.Reset();
    //reset_mode.Reset();
    base_input.Reset();
    base_offset.Reset();
    mat_config.Reset();
    gbcontrol_config.Reset();
    input_buffer_config.Reset();
    reset_mode_wr_wire.ResetRead();

    #pragma hls_unroll yes
    for (int i=0; i<6; i++) {
        base_offset_wr[i].ResetWrite();
    } 

    #pragma hls_unroll yes
    for (int i=0; i<5; i++) {
        base_input_wr[i].ResetWrite();
    } 

    #pragma hls_unroll yes
    for (int i=0; i<2; i++) {
        mat_config_wr[i].ResetWrite();
    } 

    #pragma hls_unroll yes
    for (int i=0; i<4; i++) {
        gbcontrol_config_wr[i].ResetWrite();
    } 

    input_buffer_config_wr.ResetWrite();

    NVUINT6 valids = 0;
    bool in_ready = 0;

    //wait();
    #pragma hls_pipeline_init_interval 1
    while(1) {

      NVUINT6 reset_reg;
      if (reset_mode_wr_wire.PopNB(reset_reg)) {
         #pragma hls_unroll yes 
         for (unsigned int i = 0; i < 6; i++) { 
             valids[i] = reset_reg[i];   
         } 
         in_ready = 1;
      }

      IndexType base_input_reg;
      if (base_input.PopNB(base_input_reg) && in_ready) {
        #pragma hls_unroll yes    
        for (unsigned int i = 0; i < 5; i++) {
          if (valids[i] != 0) base_input_wr[i].Push(base_input_reg);
        }
      } 
 
      IndexType base_offset_reg;
      if (base_offset.PopNB(base_offset_reg) && in_ready) {
        #pragma hls_unroll yes    
        for (unsigned int i = 0; i < 6; i++) {
          if (valids[i] != 0) base_offset_wr[i].Push(base_offset_reg);
        }
      } 

      spec::MatrixConfig mat_config_reg;
      if (mat_config.PopNB(mat_config_reg) && in_ready) {
        #pragma hls_unroll yes    
        for (unsigned int i = 0; i < 2; i++) {
          if (valids[i] != 0) mat_config_wr[i].Push(mat_config_reg);
        }
      }

      spec::GBControlConfig gbcontrol_config_reg;
      if (gbcontrol_config.PopNB(gbcontrol_config_reg) && in_ready) {
        #pragma hls_unroll yes    
        for (unsigned int i = 2; i < 6; i++) {
          if (valids[i] != 0) gbcontrol_config_wr[i-2].Push(gbcontrol_config_reg);
        }
      }
    
      spec::InputBufferConfig input_buffer_config_reg;
      if (input_buffer_config.PopNB(input_buffer_config_reg) && in_ready) {
         if (valids[5] != 0) input_buffer_config_wr.Push(input_buffer_config_reg);
      }

      wait();
    }
  }

  void RunRsp() {
    input_rsp.Reset();
    vec_out.Reset();

    #pragma hls_pipeline_init_interval 1
    while(1) {
      // for each spec::kVectorSize cycles we push one matrix to output
      spec::VectorType vec_out_reg;

      spec::input_rsp_t rsp_reg;
      rsp_reg = input_rsp.Pop();
      //CDCOUT(sc_time_stamp()  << name() << " - DUT: Decode/RunRsp - Popped Input read response" << endl, 0);
      #pragma hls_unroll yes    
      for (int j = 0; j < spec::kVectorSize; j++) {
        if (rsp_reg.valids[j] == 1) {
          vec_out_reg[j] = rsp_reg.data[j];
        }
        else {
          vec_out_reg[j] = 0;
        }
      }
      vec_out.Push(vec_out_reg);
      //CDCOUT(sc_time_stamp()  << name() << " - DUT: Decode/RunRsp - Pushed Input read response" << endl, 0);
      wait();
    }
  }
};


#endif

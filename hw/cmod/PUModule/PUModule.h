
#ifndef __PUMODULE_H__
#define __PUMODULE_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_connections.h>

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_connections.h>
#include <nvhls_vector.h>
#include <nvhls_module.h>
#include "../include/Spec.h"
#include "../include/Xbar.h"
#include "../include/OnetoTwoMux.h"
#include "../DecodeTop/DecodeTop.h"
#include "../Encode/Encode.h"
#include "../InputSetup/InputSetup.h"
#include "../Datapath_Top/Datapath_Top.h"
#include "../Accum/Accum.h"

SC_MODULE(PUModule)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;
  typedef spec::IndexType IndexType;
  IndexType offset_reg[2];
  NVUINT6 reset_mode_reg[2];

  //Mux selectors
  Connections::In<bool> flip_mem;
  Connections::In<bool> use_axi;
  Connections::In<bool> use_gb;

  //From/To InputAXI and MaskAXI
  Connections::In<input_req_t> input_mem_wr_req;
  Connections::In<input_req_t> input_mem_rd_req;  
  Connections::In<mask_req_t> mask_mem_wr_req;
  Connections::In<mask_req_t> mask_mem_rd_req;
  Connections::Out<mask_rsp_t>   axi_mask_rsp_out;
  Connections::Out<input_rsp_t>  axi_input_rsp_out;

  //From/To GB
  Connections::In<spec::VectorType> activation_from_gb_to_encoder;
  Connections::In<mask_req_t> mask_req_out_from_gb_to_maskmem;
  Connections::Out<spec::VectorType> activation_from_mem_to_gb;

  Connections::In<IndexType> base_output;
  Connections::In<spec::AccelConfig> accel_config;
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<NVUINT12> reset_mode; // 6-bit for reset_mode[0], and 6-bit for reset_mode[1]

  //From/To InputSetup
  Connections::In<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations
  Connections::In<spec::MatrixConfig> mat_config;  // matrix configurations. Also used as start signal!!!
  Connections::Out<bool>  com_IRQ;      // computation IRQ

  Connections::In<spec::InputBufferBaseOffsetConfig> offset_config;

  //Connetions wires
  Connections::Combinational<mask_rsp_t>   axi_mask_rsp_out_wr[2];
  Connections::Combinational<input_rsp_t>  axi_input_rsp_out_wr[2];
  Connections::Combinational<spec::GBControlConfig> gbcontrol_config_wr[2];
  Connections::Combinational<spec::InputBufferConfig> input_buffer_config_wr[3];
  Connections::Combinational<IndexType> base_input_wr[2];
  Connections::Combinational<IndexType> base_offset_wr[2];
  Connections::Combinational<spec::VectorType> act_in_vec_wr[2];
  Connections::Combinational<spec::VectorType> accum_wr_req_wire;
  Connections::Combinational<bool> send_out_wr;
  Connections::Combinational<spec::AccumVectorType> vec_accum_in_wr;
  Connections::Combinational<spec::VectorType> accum_out_vec_wr;
  Connections::Combinational<spec::VectorType>  vec_out_wr[2];
  Connections::Combinational<spec::VectorType>  vec_out_to_gb_wr[2];
  Connections::Combinational<spec::VectorType>  encoder_input_wr;
  Connections::Combinational<mask_req_t> encoder_mask_req_out;
  Connections::Combinational<input_req_t> encoder_input_req_out;
  Connections::Combinational<mask_req_t> mask_mem0_req_wire[5]; 
  Connections::Combinational<mask_req_t> mask_mem1_req_wire[5]; 
  Connections::Combinational<input_req_t> input_mem0_req_wire[3]; 
  Connections::Combinational<input_req_t> input_mem1_req_wire[3]; 
  Connections::Combinational<spec::MatrixConfig> mat_config_wr[4]; //sc_thread 
  Connections::Combinational<bool> use_axi_wr[4]; //sc_thread
  Connections::Combinational<bool> use_gb_wr[2]; //sc_thread
  Connections::Combinational<bool> flip_mem_wr[7]; //sc_thread
  Connections::Combinational<NVUINT6> reset_mode_wr[2]; //sc_thread


  DecodeTop decode_inst0;
  DecodeTop decode_inst1;
  Encode encode_inst;
  InputSetup inputsetup_inst;
  Datapath_Top datapath_inst;
  Accum accum_inst;
  Xbar<spec::VectorType, 2, 1, 2, 1>  arbxbar_from_mem_to_gb_inst;
  Xbar<spec::VectorType, 2, 1, 2, 1>  arbxbar_from_gb_to_encoder_inst;
  OnetoTwoMux<mask_req_t> one_to_two_mux_encoder_to_maskmem_inst;
  OnetoTwoMux<input_req_t> one_to_two_mux_encoder_to_inputmem_inst;
  OnetoTwoMux<mask_req_t> one_to_two_mux_gb_to_mem_inst;
  Xbar<mask_rsp_t, 2, 1, 2, 1>  arbxbar_mask_rsp_to_axi_inst;
  Xbar<input_rsp_t, 2, 1, 2, 1>  arbxbar_input_rsp_to_axi_inst;
  OnetoTwoMux<input_req_t> one_to_two_mux_inputaxi_to_inputmem_rd_req_inst;
  OnetoTwoMux<input_req_t> one_to_two_mux_inputaxi_to_inputmem_wr_req_inst;
  OnetoTwoMux<mask_req_t> one_to_two_mux_maskaxi_to_maskmem_rd_req_inst;
  OnetoTwoMux<mask_req_t> one_to_two_mux_maskaxi_to_maskmem_wr_req_inst;

  SC_HAS_PROCESS(PUModule);
  PUModule(sc_module_name name_) : sc_module(name_),
    decode_inst0("decode_inst0"),
    decode_inst1("decode_inst1"),
    encode_inst("encode_inst"),
    inputsetup_inst("inputsetup_inst"),
    datapath_inst("datapath_inst"),
    accum_inst("accum_inst"),
    arbxbar_from_mem_to_gb_inst("arbxbar_from_mem_to_gb_inst"),
    arbxbar_from_gb_to_encoder_inst("arbxbar_from_gb_to_encoder_inst"),
    one_to_two_mux_encoder_to_maskmem_inst("one_to_two_mux_encoder_to_maskmem_inst"),
    one_to_two_mux_encoder_to_inputmem_inst("one_to_two_mux_encoder_to_inputmem_inst"),
    one_to_two_mux_gb_to_mem_inst("one_to_two_mux_gb_to_mem_inst"),
    arbxbar_mask_rsp_to_axi_inst("arbxbar_mask_rsp_to_axi_inst"),
    arbxbar_input_rsp_to_axi_inst("arbxbar_input_rsp_to_axi_inst"),
    one_to_two_mux_inputaxi_to_inputmem_rd_req_inst("one_to_two_mux_inputaxi_to_inputmem_rd_req_inst"),
    one_to_two_mux_inputaxi_to_inputmem_wr_req_inst("one_to_two_mux_inputaxi_to_inputmem_wr_req_inst"),
    one_to_two_mux_maskaxi_to_maskmem_rd_req_inst("one_to_two_mux_maskaxi_to_maskmem_rd_req_inst"),
    one_to_two_mux_maskaxi_to_maskmem_wr_req_inst("one_to_two_mux_maskaxi_to_maskmem_wr_req_inst")
  {
    //decode_inst0
    decode_inst0.clk(clk);
    decode_inst0.rst(rst);
    decode_inst0.base_input(base_input_wr[0]); 
    decode_inst0.base_offset(base_offset_wr[0]); 
    decode_inst0.flip_mem_input(use_axi_wr[0]);
    decode_inst0.flip_mem_mask(use_axi_wr[1]);
    decode_inst0.flip_dec_out(use_gb_wr[0]);
    for (int i = 0; i < 5; i++) decode_inst0.mask_mem_req[i](mask_mem0_req_wire[i]);
    for (int i = 0; i < 3; i++) decode_inst0.input_mem_req[i](input_mem0_req_wire[i]);
    decode_inst0.axi_mask_rsp_out(axi_mask_rsp_out_wr[0]);
    decode_inst0.axi_input_rsp_out(axi_input_rsp_out_wr[0]);
    decode_inst0.vec_out(vec_out_wr[0]);
    decode_inst0.vec_out_to_gb(vec_out_to_gb_wr[0]);
    decode_inst0.mat_config(mat_config_wr[2]);
    decode_inst0.reset_mode(reset_mode_wr[0]);
    decode_inst0.gbcontrol_config(gbcontrol_config_wr[0]);
    decode_inst0.input_buffer_config(input_buffer_config_wr[0]);

    //decode_inst1
    decode_inst1.clk(clk);
    decode_inst1.rst(rst);
    decode_inst1.base_input(base_input_wr[1]); 
    decode_inst1.base_offset(base_offset_wr[1]); 
    decode_inst1.flip_mem_input(use_axi_wr[2]);
    decode_inst1.flip_mem_mask(use_axi_wr[3]);
    decode_inst1.flip_dec_out(use_gb_wr[1]);
    for (int i = 0; i < 5; i++) decode_inst1.mask_mem_req[i](mask_mem1_req_wire[i]);
    for (int i = 0; i < 3; i++) decode_inst1.input_mem_req[i](input_mem1_req_wire[i]);
    decode_inst1.axi_mask_rsp_out(axi_mask_rsp_out_wr[1]);
    decode_inst1.axi_input_rsp_out(axi_input_rsp_out_wr[1]);
    decode_inst1.vec_out(vec_out_wr[1]);
    decode_inst1.vec_out_to_gb(vec_out_to_gb_wr[1]);
    decode_inst1.mat_config(mat_config_wr[3]);
    decode_inst1.reset_mode(reset_mode_wr[1]);
    decode_inst1.gbcontrol_config(gbcontrol_config_wr[1]);
    decode_inst1.input_buffer_config(input_buffer_config_wr[1]);

    //encode_inst
    encode_inst.clk(clk);
    encode_inst.rst(rst);
    encode_inst.base_output(base_output);
    encode_inst.vec_in(encoder_input_wr);
    encode_inst.out_req(encoder_input_req_out);
    encode_inst.out_mask_req(encoder_mask_req_out);

    //inputsetup_inst
    inputsetup_inst.clk(clk);
    inputsetup_inst.rst(rst);
    inputsetup_inst.com_IRQ(com_IRQ);
    inputsetup_inst.input_buffer_config(input_buffer_config_wr[2]);
    inputsetup_inst.start(mat_config_wr[0]);
    for (int i = 0; i < 2; i++) inputsetup_inst.base_input[i](base_input_wr[i]);
    inputsetup_inst.accum_out_vec(accum_out_vec_wr);
    for (int i = 0; i < 2; i++) inputsetup_inst.act_in_vec[i](act_in_vec_wr[i]);
    for (int i = 0; i < 2; i++) inputsetup_inst.act_dec_rsp[i](vec_out_wr[i]);
    inputsetup_inst.accum_wr_req(accum_wr_req_wire);
    inputsetup_inst.mask_rd_req[0](mask_mem0_req_wire[1]);
    inputsetup_inst.mask_rd_req[1](mask_mem1_req_wire[1]);
    //for (int i = 0; i < 2; i++) inputsetup_inst.reset_ctr[i](reset_ctr_wr[i]);
    //for (int i = 0; i < 2; i++) inputsetup_inst.record_ctr[i](record_ctr_wr[i]);

    //datapath_inst
    datapath_inst.clk(clk);
    datapath_inst.rst(rst);
    datapath_inst.mat_config(mat_config_wr[1]);
    datapath_inst.vec_in0(act_in_vec_wr[0]);
    datapath_inst.vec_in1(act_in_vec_wr[1]);
    datapath_inst.vec_out(vec_accum_in_wr);
    datapath_inst.send_out(send_out_wr);
    
    //accum_inst
    accum_inst.clk(clk);
    accum_inst.rst(rst);
    accum_inst.accel_config(accel_config); 
    accum_inst.vec_in(vec_accum_in_wr); 
    accum_inst.send_out(send_out_wr); 
    accum_inst.vec_out(accum_out_vec_wr); 

    //arbxbar_from_mem_to_gb_inst
    arbxbar_from_mem_to_gb_inst.clk(clk);
    arbxbar_from_mem_to_gb_inst.rst(rst);
    for (int i = 0; i < 2; i++) arbxbar_from_mem_to_gb_inst.data_in[i](vec_out_to_gb_wr[i]);
    arbxbar_from_mem_to_gb_inst.data_out[0](activation_from_mem_to_gb);

    //arbxbar_from_gb_to_encoder_inst
    arbxbar_from_gb_to_encoder_inst.clk(clk); 
    arbxbar_from_gb_to_encoder_inst.rst(rst);
    arbxbar_from_gb_to_encoder_inst.data_in[0](accum_wr_req_wire);
    arbxbar_from_gb_to_encoder_inst.data_in[1](activation_from_gb_to_encoder);
    arbxbar_from_gb_to_encoder_inst.data_out[0](encoder_input_wr);

    //one_to_two_mux_encoder_to_maskmem_inst
    one_to_two_mux_encoder_to_maskmem_inst.clk(clk);
    one_to_two_mux_encoder_to_maskmem_inst.rst(rst);
    one_to_two_mux_encoder_to_maskmem_inst.flip(flip_mem_wr[0]);
    one_to_two_mux_encoder_to_maskmem_inst.in(encoder_mask_req_out);
    one_to_two_mux_encoder_to_maskmem_inst.out[0](mask_mem0_req_wire[0]);
    one_to_two_mux_encoder_to_maskmem_inst.out[1](mask_mem1_req_wire[0]);

    //one_to_two_mux_encoder_to_inputmem_inst
    one_to_two_mux_encoder_to_inputmem_inst.clk(clk);
    one_to_two_mux_encoder_to_inputmem_inst.rst(rst);
    one_to_two_mux_encoder_to_inputmem_inst.flip(flip_mem_wr[1]);
    one_to_two_mux_encoder_to_inputmem_inst.in(encoder_input_req_out);
    one_to_two_mux_encoder_to_inputmem_inst.out[0](input_mem0_req_wire[0]);
    one_to_two_mux_encoder_to_inputmem_inst.out[1](input_mem1_req_wire[0]);

    //one_to_two_mux_gb_to_mem_inst
    one_to_two_mux_gb_to_mem_inst.clk(clk);
    one_to_two_mux_gb_to_mem_inst.rst(rst);
    one_to_two_mux_gb_to_mem_inst.flip(flip_mem_wr[2]);
    one_to_two_mux_gb_to_mem_inst.in(mask_req_out_from_gb_to_maskmem);
    one_to_two_mux_gb_to_mem_inst.out[0](mask_mem0_req_wire[2]);
    one_to_two_mux_gb_to_mem_inst.out[1](mask_mem1_req_wire[2]);

    //arbxbar_mask_rsp_to_axi_inst
    arbxbar_mask_rsp_to_axi_inst.clk(clk);
    arbxbar_mask_rsp_to_axi_inst.rst(rst);
    for (int i=0; i < 2; i++) arbxbar_mask_rsp_to_axi_inst.data_in[i](axi_mask_rsp_out_wr[i]);
    arbxbar_mask_rsp_to_axi_inst.data_out[0](axi_mask_rsp_out);

    //arbxbar_input_rsp_to_axi_inst
    arbxbar_input_rsp_to_axi_inst.clk(clk);
    arbxbar_input_rsp_to_axi_inst.rst(rst);
    for (int i=0; i < 2; i++) arbxbar_input_rsp_to_axi_inst.data_in[i](axi_input_rsp_out_wr[i]);
    arbxbar_input_rsp_to_axi_inst.data_out[0](axi_input_rsp_out);

    //one_to_two_mux_inputaxi_to_inputmem_rd_req_inst
    one_to_two_mux_inputaxi_to_inputmem_rd_req_inst.clk(clk);
    one_to_two_mux_inputaxi_to_inputmem_rd_req_inst.rst(rst);
    one_to_two_mux_inputaxi_to_inputmem_rd_req_inst.flip(flip_mem_wr[3]);
    one_to_two_mux_inputaxi_to_inputmem_rd_req_inst.in(input_mem_rd_req);
    one_to_two_mux_inputaxi_to_inputmem_rd_req_inst.out[0](input_mem0_req_wire[1]);
    one_to_two_mux_inputaxi_to_inputmem_rd_req_inst.out[1](input_mem1_req_wire[1]);

    //one_to_two_mux_inputaxi_to_inputmem_wr_req_inst
    one_to_two_mux_inputaxi_to_inputmem_wr_req_inst.clk(clk);
    one_to_two_mux_inputaxi_to_inputmem_wr_req_inst.rst(rst);
    one_to_two_mux_inputaxi_to_inputmem_wr_req_inst.flip(flip_mem_wr[4]);
    one_to_two_mux_inputaxi_to_inputmem_wr_req_inst.in(input_mem_wr_req);
    one_to_two_mux_inputaxi_to_inputmem_wr_req_inst.out[0](input_mem0_req_wire[2]);
    one_to_two_mux_inputaxi_to_inputmem_wr_req_inst.out[1](input_mem1_req_wire[2]);

    //one_to_two_mux_maskaxi_to_maskmem_rd_req_inst
    one_to_two_mux_maskaxi_to_maskmem_rd_req_inst.clk(clk);
    one_to_two_mux_maskaxi_to_maskmem_rd_req_inst.rst(rst);
    one_to_two_mux_maskaxi_to_maskmem_rd_req_inst.flip(flip_mem_wr[5]);
    one_to_two_mux_maskaxi_to_maskmem_rd_req_inst.in(mask_mem_rd_req);
    one_to_two_mux_maskaxi_to_maskmem_rd_req_inst.out[0](mask_mem0_req_wire[3]);
    one_to_two_mux_maskaxi_to_maskmem_rd_req_inst.out[1](mask_mem1_req_wire[3]);

    //one_to_two_mux_maskaxi_to_maskmem_wr_req_inst
    one_to_two_mux_maskaxi_to_maskmem_wr_req_inst.clk(clk);
    one_to_two_mux_maskaxi_to_maskmem_wr_req_inst.rst(rst);
    one_to_two_mux_maskaxi_to_maskmem_wr_req_inst.flip(flip_mem_wr[6]);
    one_to_two_mux_maskaxi_to_maskmem_wr_req_inst.in(mask_mem_wr_req);
    one_to_two_mux_maskaxi_to_maskmem_wr_req_inst.out[0](mask_mem0_req_wire[4]);
    one_to_two_mux_maskaxi_to_maskmem_wr_req_inst.out[1](mask_mem1_req_wire[4]);

    SC_THREAD (ResetModeRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 
 
    SC_THREAD (MatConfigRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 

    SC_THREAD (UseAxiRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 

    SC_THREAD (UseGbRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 

    SC_THREAD (FlipMemRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

    SC_THREAD (OffsetRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 

    SC_THREAD (GBControlRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 

    SC_THREAD (InputBufferConfigRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);  
  }

  void InputBufferConfigRun() {
    input_buffer_config.Reset();

    #pragma hls_unroll yes
    for (int i=0; i<3; i++) input_buffer_config_wr[i].ResetWrite();

    #pragma hls_pipeline_init_interval 1
    while(1) {
      spec::InputBufferConfig input_buffer_config_reg;
      if (input_buffer_config.PopNB(input_buffer_config_reg)) {
        #pragma hls_unroll yes
        for (int i=0; i<3; i++) input_buffer_config_wr[i].Push(input_buffer_config_reg);
      }
      wait();
    }
  }

  void GBControlRun() {
    gbcontrol_config.Reset();

    #pragma hls_unroll yes
    for (int i=0; i<2; i++) gbcontrol_config_wr[i].ResetWrite();

    #pragma hls_pipeline_init_interval 1
    while(1) {
      spec::GBControlConfig gbcontrol_config_reg;
      if (gbcontrol_config.PopNB(gbcontrol_config_reg)) {
        #pragma hls_unroll yes
        for (int i=0; i<2; i++) gbcontrol_config_wr[i].Push(gbcontrol_config_reg);
      }
      wait();
    }
  }

  //12-bit reset_mode split between 6-bit reset_mode_wr[0] and 6-bit reset_mode_wr[1]
  //6-bit reset_mode_wr config: 1:N0, 2:N1, 4:LayerNorm, 8:SMax, 16:Enpy, 32:EADD
  void ResetModeRun() {
    reset_mode.Reset();

    #pragma hls_unroll yes
    for (int i=0; i<2; i++) {
        reset_mode_wr[i].ResetWrite();
    }

    reset_mode_reg[0] = 0;
    reset_mode_reg[1] = 1;

    #pragma hls_pipeline_init_interval 1
    while(1) {
      NVUINT12 reset_mode_tmp;
      if (reset_mode.PopNB(reset_mode_tmp)) {
          #pragma hls_unroll yes
          for (int i=0; i<2; i++) {
             reset_mode_reg[i] = nvhls::get_slc<6>(reset_mode_tmp, 6*i); 
             reset_mode_wr[i].Push(reset_mode_reg[i]);
             CDCOUT(sc_time_stamp() << name() << "DUT - PUModule with reset_mode_reg[i]: " << reset_mode_reg[i] << endl, 0);
          }
      }
      wait();
    }
  }

  void OffsetRun() {
    #pragma hls_unroll yes
    for (int i=0; i<2; i++) {
      base_offset_wr[i].ResetWrite();
      offset_reg[i] = 0;
    }
    offset_config.Reset();

    //wait(); 
    #pragma hls_pipeline_init_interval 1
    while(1) {
      spec::InputBufferBaseOffsetConfig offset_config_tmp;
      if (offset_config.PopNB(offset_config_tmp)) {
          #pragma hls_unroll yes
          for (int i=0; i<2; i++) {
            offset_reg[i] = offset_config_tmp.base_input_offset[i];
            base_offset_wr[i].Push(offset_reg[i]);
          }
      }

      wait();
    }
  }

  void FlipMemRun() {
    flip_mem.Reset();

    #pragma hls_unroll yes
    for (int i=0; i<7; i++) flip_mem_wr[i].ResetWrite();

    #pragma hls_pipeline_init_interval 1
    while(1) {
      bool flip_mem_reg;
      if (flip_mem.PopNB(flip_mem_reg)) {
        #pragma hls_unroll yes
        for (int i=0; i<7; i++) flip_mem_wr[i].Push(flip_mem_reg);
      }
      wait();
    }
  }

  void UseGbRun() {
    use_gb.Reset();

    #pragma hls_unroll yes
    for (int i=0; i<2; i++) use_gb_wr[i].ResetWrite();

    #pragma hls_pipeline_init_interval 1
    while(1) {
      bool use_gb_reg;
      if (use_gb.PopNB(use_gb_reg)) {
        #pragma hls_unroll yes
        for (int i=0; i<2; i++) use_gb_wr[i].Push(use_gb_reg);
      }
      wait();
    }
  }

  void UseAxiRun() {
    use_axi.Reset();

    #pragma hls_unroll yes
    for (int i=0; i<4; i++) use_axi_wr[i].ResetWrite();

    #pragma hls_pipeline_init_interval 1
    while(1) {
      bool use_axi_reg;
      if (use_axi.PopNB(use_axi_reg)) {
        #pragma hls_unroll yes
        for (int i=0; i<4; i++) use_axi_wr[i].Push(use_axi_reg);
      }
      wait();
    }
  }

  void MatConfigRun() {
    mat_config.Reset();

    #pragma hls_unroll yes
    for (int i=0; i<4; i++) mat_config_wr[i].ResetWrite();

    #pragma hls_pipeline_init_interval 1
    while(1) {
      spec::MatrixConfig mat_config_reg;
      if (mat_config.PopNB(mat_config_reg)) {
        #pragma hls_unroll yes
        for (int i=0; i<4; i++) mat_config_wr[i].Push(mat_config_reg);
        CDCOUT(sc_time_stamp()  << name() << " - DUT: PUModule - Pushed mat_config" << endl, 0); 
      }
      wait();
    }
  }


};

#endif

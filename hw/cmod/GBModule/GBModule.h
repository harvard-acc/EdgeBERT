
#ifndef __GBMODULE_H__
#define __GBMODULE_H__

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
#include "../include/Broadcast.h"
#include "../include/GBDone.h"
#include "../DecodeTop/DecodeTop.h"
#include "../AuxMem/AuxMem.h"

#include "../LayerNorm/LayerNorm.h"
#include "../Enpy/Enpy.h"
#include "../SMax/SMax.h"
#include "../ElemAdd/ElemAdd.h"
#include "../Dvfs/Dvfs.h"

SC_MODULE(GBModule)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef DecodeTop MaskMemType;
  typedef AuxMem AuxMemType;

  //0: LayerNorm, 1: SMax, 2: ElemAdd, 3: Enpy, 4: Dvfs
  Connections::In<bool> start[5]; //added start of dvfs
  Connections::In<NVUINT4> mode_config; //1: LayerNorm, 2: SMax, 4:ElemAdd/Dvfs, 8: Enpy
  Connections::In<bool> use_axi;

  Connections::In<AuxMemType::aux_req_t>  aux_mem_req[2];
  Connections::In<spec::VectorType> activation_in;
  Connections::In<spec::EnpyConfig> enpy_config;
  Connections::In<spec::InputBufferConfig> input_buffer_config;
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<spec::PeriphConfig> periph_config;
  Connections::In<spec::DvfsConfig> dvfs_config; //dvfs 

  Connections::In<spec::DCOConfigA> dco_config_a;
  Connections::In<spec::DCOConfigB> dco_config_b;
  Connections::In<spec::DCOConfigC> dco_config_c;
  //  
  Connections::In<spec::LDOConfigA> ldo_config_a;
  Connections::In<spec::LDOConfigB> ldo_config_b;
  Connections::In<spec::LDOConfigC> ldo_config_c;
  Connections::In<spec::LDOConfigD> ldo_config_d;

  Connections::Out<spec::VectorType> activation_out;
  Connections::Out<MaskMemType::mask_req_t> mask_req_out;
  Connections::Out<AuxMemType::aux_rsp_t>  aux_mem_rsp;
  Connections::Out<NVUINT2> enpy_status;
  Connections::Out<bool> done_out;
  Connections::Out<NVUINT6> dco_sel_out;  //dvfs
  Connections::Out<NVUINT8> ldo_sel_out;   //dvfs

  Connections::Combinational<bool> done_wr[5];
  Connections::Combinational<NVUINT4> mode_config_wr[5];
  Connections::Combinational<MaskMemType::mask_req_t> mask_rd_req_wr[4];
  Connections::Combinational<AuxMemType::aux_req_t>  aux_mem_req_wr[2];
  Connections::Combinational<spec::VectorType>  act_out_vec_wr[3];
  Connections::Combinational<spec::VectorType>  aux_rsp_out_wr;
  Connections::Combinational<spec::VectorType>  aux_rsp_in_wr[2];
  Connections::Combinational<spec::InputBufferConfig> input_buffer_config_wr[4];
  Connections::Combinational<spec::GBControlConfig> gbcontrol_config_wr[4];
  Connections::Combinational<spec::PeriphConfig> periph_config_wr[2];
  Connections::Combinational<spec::VectorType> activation_in_wr[4];
  Connections::Combinational<spec::ActScalarType> enpy_val_out;

  AuxMem auxmem_inst;
  LayerNorm layernorm_inst;
  SMax smax_inst;
  ElemAdd elemadd_inst;
  Enpy enpy_inst;
  Dvfs dvfs_inst;
  GBDone gbdone_inst;
  Xbar<MaskMemType::mask_req_t, 4, 1, 4, 1> arbxbar_maskreq_inst;
  Xbar<spec::VectorType, 3, 1, 3, 1>  arbxbar_actout_inst;
  Broadcast<spec::VectorType, 4> broadcast_activation_inst;
  Broadcast<spec::GBControlConfig, 4> broadcast_gbcontrol_inst;
  Broadcast<spec::InputBufferConfig, 4> broadcast_inputbufferconfig_inst;
  Broadcast<spec::PeriphConfig, 2> broadcast_periphconfig_inst;
  Broadcast<spec::VectorType, 2> broadcast_auxbuffer_rsp_inst;


  SC_HAS_PROCESS(GBModule);
  GBModule(sc_module_name name_) : sc_module(name_),
    auxmem_inst("auxmem_inst"),
    layernorm_inst("layernorm_inst"),
    smax_inst("smax_inst"),
    elemadd_inst("elemadd_inst"),
    enpy_inst("enpy_inst"),
    dvfs_inst("dvfs_inst"),
    gbdone_inst("gbdone_inst"),
    arbxbar_maskreq_inst("arbxbar_maskreq_inst"),
    arbxbar_actout_inst("arbxbar_actout_inst"),
    broadcast_activation_inst("broadcast_activation_inst"),
    broadcast_gbcontrol_inst("broadcast_gbcontrol_inst"),
    broadcast_inputbufferconfig_inst("broadcast_inputbufferconfig_inst"),
    broadcast_periphconfig_inst("broadcast_periphconfig_inst"),
    broadcast_auxbuffer_rsp_inst("broadcast_auxbuffer_rsp_inst")
  {
    //Aux Buffer
    auxmem_inst.clk(clk);
    auxmem_inst.rst(rst);
    auxmem_inst.use_axi(use_axi);
    for (int i = 0; i < 2; i++) auxmem_inst.aux_mem_req[i](aux_mem_req_wr[i]);
    for (int i = 2; i < 4; i++) auxmem_inst.aux_mem_req[i](aux_mem_req[3-i]);
    auxmem_inst.aux_mem_rsp(aux_mem_rsp);
    auxmem_inst.aux_vec_out(aux_rsp_out_wr);

    //LayerNorm
    layernorm_inst.clk(clk);
    layernorm_inst.rst(rst);
    layernorm_inst.start(start[0]);
    layernorm_inst.input_buffer_config(input_buffer_config_wr[0]);
    layernorm_inst.layernorm_config(periph_config_wr[0]);
    layernorm_inst.gbcontrol_config(gbcontrol_config_wr[0]);
    layernorm_inst.act_rsp(activation_in_wr[0]);
    layernorm_inst.layernorm_param_rsp(aux_rsp_in_wr[0]);
    layernorm_inst.layernorm_param_req(aux_mem_req_wr[0]);
    layernorm_inst.act_out_vec(act_out_vec_wr[0]);
    layernorm_inst.mask_rd_req(mask_rd_req_wr[0]);
    layernorm_inst.done(done_wr[0]);

    //SMax
    smax_inst.clk(clk);
    smax_inst.rst(rst);
    smax_inst.start(start[1]);
    smax_inst.input_buffer_config(input_buffer_config_wr[1]);
    smax_inst.softmax_config(periph_config_wr[1]);
    smax_inst.gbcontrol_config(gbcontrol_config_wr[1]);
    smax_inst.act_rsp(activation_in_wr[1]);
    smax_inst.attn_span_rsp(aux_rsp_in_wr[1]);
    smax_inst.attn_span_req(aux_mem_req_wr[1]);
    smax_inst.act_out_vec(act_out_vec_wr[1]);
    smax_inst.mask_rd_req(mask_rd_req_wr[1]);
    smax_inst.done(done_wr[1]);

    //ElemAdd
    elemadd_inst.clk(clk);
    elemadd_inst.rst(rst);
    elemadd_inst.start(start[2]);
    elemadd_inst.input_buffer_config(input_buffer_config_wr[2]);
    elemadd_inst.gbcontrol_config(gbcontrol_config_wr[2]);
    elemadd_inst.act_rsp(activation_in_wr[2]);
    elemadd_inst.mask_rd_req(mask_rd_req_wr[2]);
    elemadd_inst.act_out_vec(act_out_vec_wr[2]);
    elemadd_inst.done(done_wr[2]);

    //Enpy
    enpy_inst.clk(clk);
    enpy_inst.rst(rst);
    enpy_inst.start(start[3]);
    enpy_inst.input_buffer_config(input_buffer_config_wr[3]);
    enpy_inst.gbcontrol_config(gbcontrol_config_wr[3]);
    enpy_inst.enpy_config(enpy_config);
    enpy_inst.act_rsp(activation_in_wr[3]);
    enpy_inst.mask_rd_req(mask_rd_req_wr[3]);
    enpy_inst.enpy_status(enpy_status);
    enpy_inst.enpy_val_out(enpy_val_out);
    enpy_inst.done(done_wr[3]);

    //Dvfs
    dvfs_inst.clk(clk);
    dvfs_inst.rst(rst);
    dvfs_inst.start(start[4]);
    dvfs_inst.dvfs_config(dvfs_config);
    dvfs_inst.enpy_val_in(enpy_val_out);
    dvfs_inst.dco_sel_out(dco_sel_out);
    dvfs_inst.ldo_sel_out(ldo_sel_out);
    dvfs_inst.dco_config_a(dco_config_a);
    dvfs_inst.dco_config_b(dco_config_b);
    dvfs_inst.dco_config_c(dco_config_c);
    dvfs_inst.ldo_config_a(ldo_config_a);
    dvfs_inst.ldo_config_b(ldo_config_b);
    dvfs_inst.ldo_config_c(ldo_config_c);
    dvfs_inst.ldo_config_d(ldo_config_d);
    dvfs_inst.done(done_wr[4]);

    //GBDone
    gbdone_inst.clk(clk);
    gbdone_inst.rst(rst);
    gbdone_inst.layernorm_done(done_wr[0]);
    gbdone_inst.smax_done(done_wr[1]);
    gbdone_inst.elemadd_done(done_wr[2]);
    gbdone_inst.enpy_done(done_wr[3]);
    gbdone_inst.dvfs_done(done_wr[4]);
    gbdone_inst.done(done_out); 

    //arbxbar_maskreq_inst
    arbxbar_maskreq_inst.clk(clk); 
    arbxbar_maskreq_inst.rst(rst);
    for (int i = 0; i < 4; i++) {     
      arbxbar_maskreq_inst.data_in[i](mask_rd_req_wr[i]);
    }   
    arbxbar_maskreq_inst.data_out[0](mask_req_out);

    //arbxbar_actout_inst
    arbxbar_actout_inst.clk(clk); 
    arbxbar_actout_inst.rst(rst);
    for (int i = 0; i < 3; i++) {     
      arbxbar_actout_inst.data_in[i](act_out_vec_wr[i]);
    }   
    arbxbar_actout_inst.data_out[0](activation_out);

    //broadcast_activation_inst
    broadcast_activation_inst.clk(clk);
    broadcast_activation_inst.rst(rst);
    broadcast_activation_inst.data_in(activation_in); 
    broadcast_activation_inst.mode_config(mode_config_wr[0]);
    for (int i = 0; i < 4; i++) broadcast_activation_inst.data_out[i](activation_in_wr[i]);

    //broadcast_gbcontrol_inst
    broadcast_gbcontrol_inst.clk(clk);
    broadcast_gbcontrol_inst.rst(rst);
    broadcast_gbcontrol_inst.data_in(gbcontrol_config);
    broadcast_gbcontrol_inst.mode_config(mode_config_wr[1]);
    for (int i = 0; i < 4; i++) broadcast_gbcontrol_inst.data_out[i](gbcontrol_config_wr[i]);

    //broadcast_inputbufferconfig_inst
    broadcast_inputbufferconfig_inst.clk(clk);
    broadcast_inputbufferconfig_inst.rst(rst);
    broadcast_inputbufferconfig_inst.data_in(input_buffer_config);
    broadcast_inputbufferconfig_inst.mode_config(mode_config_wr[2]);
    for (int i = 0; i < 4; i++) broadcast_inputbufferconfig_inst.data_out[i](input_buffer_config_wr[i]);

    //broadcast_periphconfig_inst
    broadcast_periphconfig_inst.clk(clk);
    broadcast_periphconfig_inst.rst(rst);
    broadcast_periphconfig_inst.data_in(periph_config);
    broadcast_periphconfig_inst.mode_config(mode_config_wr[3]);
    for (int i = 0; i < 2; i++) broadcast_periphconfig_inst.data_out[i](periph_config_wr[i]);

    //broadcast_auxbuffer_rsp_inst
    broadcast_auxbuffer_rsp_inst.clk(clk);
    broadcast_auxbuffer_rsp_inst.rst(rst);
    broadcast_auxbuffer_rsp_inst.data_in(aux_rsp_out_wr);
    broadcast_auxbuffer_rsp_inst.mode_config(mode_config_wr[4]);
    for (int i = 0; i < 2; i++) broadcast_auxbuffer_rsp_inst.data_out[i](aux_rsp_in_wr[i]);

    SC_THREAD (ModeRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 
  }

  void ModeRun() {
    mode_config.Reset();

    #pragma hls_unroll yes
    for (int i=0; i<5; i++) mode_config_wr[i].ResetWrite();

    #pragma hls_pipeline_init_interval 1
    while(1) {
      NVUINT4 mode_reg;
      
      if (mode_config.PopNB(mode_reg)) {
        //cout << "DUT - GBModule mode_config: " << mode_reg << endl;
        #pragma hls_unroll yes
        for (int i=0; i<5; i++) mode_config_wr[i].Push(mode_reg);
      }

      wait();
    }
  }

};

#endif

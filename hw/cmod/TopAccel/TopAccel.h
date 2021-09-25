/*
 * All rights reserved - Harvard University. 
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
 * EdgeBERT Accelerator TopAccel-level
 */

#ifndef __TOPACCEL_H__
#define __TOPACCEL_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_module.h>
#include <nvhls_connections.h>

#include "../include/Spec.h"
#include "../include/AxiSpec.h"

#include "../InputAxi/InputAxi.h"
#include "../MaskAxi/MaskAxi.h"
#include "../AuxAxi/AuxAxi.h"
#include "../PUModule/PUModule.h"
#include "../GBModule/GBModule.h"
#include "../Control/Control.h"

SC_MODULE(TopAccel)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;
  typedef spec::aux_req_t aux_req_t;
  typedef spec::aux_rsp_t aux_rsp_t;
  typedef spec::IndexType IndexType;
  const static int N = spec::N;

  // IRQ
  sc_out<bool> interrupt;

  // DCO & LDO settings
  sc_out<NVUINT6> edgebert_dco_cc_sel;
  sc_out<NVUINT8> edgebert_ldo_res_sel;

  // Axi Slave Config
  typename spec::AxiConf::axi4_conf::read::template slave<>   if_axi_rd;
  typename spec::AxiConf::axi4_conf::write::template slave<>  if_axi_wr;

  // Axi Master Data
  typename spec::AxiData::axi4_data::read::template master<>   if_data_rd;
  typename spec::AxiData::axi4_data::write::template master<>  if_data_wr;

  // Axi Master Mask
  typename spec::AxiData::axi4_data::read::template chan<>   if_mask_rd;
  typename spec::AxiData::axi4_data::write::template chan<>  if_mask_wr;

  // Axi Master Input
  typename spec::AxiData::axi4_data::read::template chan<>   if_input_rd;
  typename spec::AxiData::axi4_data::write::template chan<>  if_input_wr;

  // Axi Master Aux
  typename spec::AxiData::axi4_data::read::template chan<>   if_aux_rd;
  typename spec::AxiData::axi4_data::write::template chan<>  if_aux_wr;


  Connections::Combinational<bool> IRQs[8];
  Connections::Combinational<NVUINT2> enpy_status;
  Connections::Combinational<bool> start[5];
  Connections::Combinational<bool> flip_mem;
  Connections::Combinational<bool> use_axi[2];
  Connections::Combinational<bool> use_gb;
  Connections::Combinational<NVUINT4> gb_mode_config;
  Connections::Combinational<IndexType> base_output[3];
  Connections::Combinational<NVUINT12> reset_mode;
  Connections::Combinational<spec::AccelConfig> accel_config;
  Connections::Combinational<spec::MatrixConfig> mat_config;
  Connections::Combinational<spec::InputBufferConfig> input_buffer_config[2];
  Connections::Combinational<spec::InputBufferBaseOffsetConfig> offset_config;
  Connections::Combinational<spec::PeriphConfig> periph_config;
  Connections::Combinational<spec::GBControlConfig> gbcontrol_config[2];
  Connections::Combinational<spec::EnpyConfig> enpy_config;
  Connections::Combinational<InputAxi::MasterTrig> input_rd_axi_start;
  Connections::Combinational<InputAxi::MasterTrig> input_wr_axi_start;
  Connections::Combinational<MaskAxi::MasterTrig> mask_rd_axi_start;
  Connections::Combinational<MaskAxi::MasterTrig> mask_wr_axi_start;
  Connections::Combinational<AuxAxi::MasterTrig> aux_rd_axi_start;
  Connections::Combinational<AuxAxi::MasterTrig> aux_wr_axi_start;
  Connections::Combinational<spec::VectorType> activation_fom_pu_to_gb;
  Connections::Combinational<spec::VectorType> activation_from_gb_to_pu;
  Connections::Combinational<mask_req_t> gb_to_pu_mask_req_out;

  Connections::Combinational<spec::DvfsConfig> dvfs_config; //dvfs 
  Connections::Combinational<NVUINT6> dco_sel_out;  //dvfs
  Connections::Combinational<NVUINT8> ldo_sel_out;   //dvfs
  Connections::Combinational<spec::DCOConfigA> dco_config_a;
  Connections::Combinational<spec::DCOConfigB> dco_config_b;
  Connections::Combinational<spec::DCOConfigC> dco_config_c;
  Connections::Combinational<spec::LDOConfigA> ldo_config_a;
  Connections::Combinational<spec::LDOConfigB> ldo_config_b;
  Connections::Combinational<spec::LDOConfigC> ldo_config_c;
  Connections::Combinational<spec::LDOConfigD> ldo_config_d;

  Connections::Combinational<aux_req_t>  axi_aux_mem_req[2];
  Connections::Combinational<aux_rsp_t>  axi_aux_mem_rsp;

  Connections::Combinational<input_req_t>  axi_input_mem_req[2];
  Connections::Combinational<input_rsp_t>  axi_input_mem_rsp;

  Connections::Combinational<mask_req_t>  axi_mask_mem_req[2];
  Connections::Combinational<mask_rsp_t>  axi_mask_mem_rsp;

  PUModule    pu_inst;
  GBModule    gb_inst;
  Control     ct_inst;
  MaskAxi     ma_inst;
  InputAxi    ia_inst;
  AuxAxi      aa_inst;
  spec::AxiData::ArbiterData axi_arbiter;

  SC_HAS_PROCESS(TopAccel);
  TopAccel(sc_module_name name_) : sc_module(name_),
    if_axi_rd("if_axi_rd"),
    if_axi_wr("if_axi_wr"),
    if_data_rd("if_data_rd"),
    if_data_wr("if_data_wr"),
    if_mask_rd("if_mask_rd"),
    if_mask_wr("if_mask_wr"),
    if_input_rd("if_input_rd"),
    if_input_wr("if_input_wr"),
    if_aux_rd("if_aux_rd"),
    if_aux_wr("if_aux_wr"),
    pu_inst("pu_inst"),
    gb_inst("gb_inst"),
    ct_inst("ct_inst"),
    ma_inst("ma_inst"),
    ia_inst("ia_inst"),
    aa_inst("aa_inst"),
    axi_arbiter("axi_arbiter")
  {

    //PUModule unit
    pu_inst.clk(clk);
    pu_inst.rst(rst);
    pu_inst.flip_mem(flip_mem);
    pu_inst.use_axi(use_axi[0]); 
    pu_inst.use_gb(use_gb);
    pu_inst.input_mem_rd_req(axi_input_mem_req[0]);
    pu_inst.input_mem_wr_req(axi_input_mem_req[1]);
    pu_inst.mask_mem_rd_req(axi_mask_mem_req[0]);
    pu_inst.mask_mem_wr_req(axi_mask_mem_req[1]);
    pu_inst.axi_mask_rsp_out(axi_mask_mem_rsp);
    pu_inst.axi_input_rsp_out(axi_input_mem_rsp);
    pu_inst.activation_from_gb_to_encoder(activation_from_gb_to_pu);
    pu_inst.mask_req_out_from_gb_to_maskmem(gb_to_pu_mask_req_out);
    pu_inst.activation_from_mem_to_gb(activation_fom_pu_to_gb);
    pu_inst.base_output(base_output[0]);
    pu_inst.accel_config(accel_config);
    pu_inst.gbcontrol_config(gbcontrol_config[0]);
    pu_inst.reset_mode(reset_mode);
    pu_inst.input_buffer_config(input_buffer_config[0]);
    pu_inst.mat_config(mat_config);
    pu_inst.com_IRQ(IRQs[0]);
    pu_inst.offset_config(offset_config);

    // Mask Axi Master
    ma_inst.clk(clk);
    ma_inst.rst(rst); 
    ma_inst.master_read(mask_rd_axi_start);
    ma_inst.master_write(mask_wr_axi_start);
    ma_inst.base_output(base_output[1]);
    ma_inst.rd_IRQ(IRQs[2]);
    ma_inst.wr_IRQ(IRQs[3]);
    ma_inst.if_data_rd(if_mask_rd);
    ma_inst.if_data_wr(if_mask_wr);
    ma_inst.mem_rd_req(axi_mask_mem_req[0]);
    ma_inst.mem_wr_req(axi_mask_mem_req[1]);
    ma_inst.mem_rd_rsp(axi_mask_mem_rsp);

    // Input Axi Master
    ia_inst.clk(clk);
    ia_inst.rst(rst); 
    ia_inst.master_read(input_rd_axi_start);
    ia_inst.master_write(input_wr_axi_start);
    ia_inst.base_output(base_output[2]);
    ia_inst.rd_IRQ(IRQs[4]);
    ia_inst.wr_IRQ(IRQs[5]);
    ia_inst.if_data_rd(if_input_rd);
    ia_inst.if_data_wr(if_input_wr);
    ia_inst.mem_rd_req(axi_input_mem_req[0]);
    ia_inst.mem_wr_req(axi_input_mem_req[1]);
    ia_inst.mem_rd_rsp(axi_input_mem_rsp);
    
    // Auxiliary Axi Master
    aa_inst.clk(clk);
    aa_inst.rst(rst);
    aa_inst.master_read(aux_rd_axi_start);
    aa_inst.master_write(aux_wr_axi_start);
    aa_inst.rd_IRQ(IRQs[6]);
    aa_inst.wr_IRQ(IRQs[7]);
    aa_inst.if_data_rd(if_aux_rd);
    aa_inst.if_data_wr(if_aux_wr);
    aa_inst.mem_rd_req(axi_aux_mem_req[0]);
    aa_inst.mem_wr_req(axi_aux_mem_req[1]);
    aa_inst.mem_rd_rsp(axi_aux_mem_rsp);

    //GBmodule unit
    gb_inst.clk(clk);
    gb_inst.rst(rst);
    gb_inst.mode_config(gb_mode_config);
    for (int i = 0; i < 5; i++) gb_inst.start[i](start[i]);
    gb_inst.use_axi(use_axi[1]);
    gb_inst.done_out(IRQs[1]);
    gb_inst.dco_sel_out(dco_sel_out);
    gb_inst.ldo_sel_out(ldo_sel_out);
    gb_inst.enpy_status(enpy_status);
    gb_inst.enpy_config(enpy_config);
    gb_inst.dco_config_a(dco_config_a);
    gb_inst.dco_config_b(dco_config_b);
    gb_inst.dco_config_c(dco_config_c);
    gb_inst.ldo_config_a(ldo_config_a);
    gb_inst.ldo_config_b(ldo_config_b);
    gb_inst.ldo_config_c(ldo_config_c);
    gb_inst.ldo_config_d(ldo_config_d);
    gb_inst.input_buffer_config(input_buffer_config[1]);
    gb_inst.gbcontrol_config(gbcontrol_config[1]);  
    gb_inst.periph_config(periph_config);
    gb_inst.dvfs_config(dvfs_config);
    gb_inst.activation_out(activation_from_gb_to_pu);
    gb_inst.activation_in(activation_fom_pu_to_gb);
    gb_inst.mask_req_out(gb_to_pu_mask_req_out);
    gb_inst.aux_mem_rsp(axi_aux_mem_rsp);
    for (int i = 0; i < 2; i++) gb_inst.aux_mem_req[i](axi_aux_mem_req[i]);

    // Control unit 
    ct_inst.clk(clk);
    ct_inst.rst(rst); 
    ct_inst.interrupt(interrupt); // IRQ sc_out -
    ct_inst.edgebert_dco_cc_sel(edgebert_dco_cc_sel);
    ct_inst.edgebert_ldo_res_sel(edgebert_ldo_res_sel);
    ct_inst.if_axi_rd(if_axi_rd); // axi slave
    ct_inst.if_axi_wr(if_axi_wr); // axi slave
    ct_inst.flip_mem(flip_mem);       // flip mem -
    ct_inst.use_gb(use_gb);
    for (int i = 0; i < 3; i++) ct_inst.base_output[i](base_output[i]);
    ct_inst.reset_mode(reset_mode);
    ct_inst.enpy_status(enpy_status);
    ct_inst.accel_config(accel_config);                 // Accel Config
    ct_inst.mat_config(mat_config);                 // -
    ct_inst.gb_mode_config(gb_mode_config);                 // -
    ct_inst.enpy_config(enpy_config);                 // -
    ct_inst.periph_config(periph_config);                 // -
    ct_inst.offset_config(offset_config);                 // -
    ct_inst.dvfs_config(dvfs_config); //dvfs
    ct_inst.dco_sel_out(dco_sel_out); //dvfs
    ct_inst.ldo_sel_out(ldo_sel_out); //dvfs
    ct_inst.dco_config_a(dco_config_a);
    ct_inst.dco_config_b(dco_config_b);
    ct_inst.dco_config_c(dco_config_c);
    ct_inst.ldo_config_a(ldo_config_a);
    ct_inst.ldo_config_b(ldo_config_b);
    ct_inst.ldo_config_c(ldo_config_c);
    ct_inst.ldo_config_d(ldo_config_d);
    for (int i = 0; i < 2; i++) ct_inst.input_buffer_config[i](input_buffer_config[i]); // -
    for (int i = 0; i < 2; i++) ct_inst.gbcontrol_config[i](gbcontrol_config[i]); // -
    ct_inst.mask_rd_axi_start(mask_rd_axi_start);     // mask master read
    ct_inst.mask_wr_axi_start(mask_wr_axi_start);     // mask master write
    ct_inst.input_rd_axi_start(input_rd_axi_start); // input master read
    ct_inst.input_wr_axi_start(input_wr_axi_start); // input master write
    ct_inst.aux_rd_axi_start(aux_rd_axi_start); // aux master read
    ct_inst.aux_wr_axi_start(aux_wr_axi_start); // aux master write
    for (int i = 0; i < 2; i++) ct_inst.use_axi[i](use_axi[i]); //-
    for (int i = 0; i < 5; i++) ct_inst.start[i](start[i]); //-
    for (int i = 0; i < 8; i++) ct_inst.IRQs[i](IRQs[i]); //-


    axi_arbiter.clk(clk);
    axi_arbiter.reset_bar(rst);

    axi_arbiter.axi_rd_m_ar[0](if_mask_rd.ar);
    axi_arbiter.axi_rd_m_r [0](if_mask_rd.r);
    axi_arbiter.axi_wr_m_aw[0](if_mask_wr.aw);
    axi_arbiter.axi_wr_m_w [0](if_mask_wr.w);
    axi_arbiter.axi_wr_m_b [0](if_mask_wr.b);

    axi_arbiter.axi_rd_m_ar[1](if_input_rd.ar);
    axi_arbiter.axi_rd_m_r [1](if_input_rd.r);
    axi_arbiter.axi_wr_m_aw[1](if_input_wr.aw);
    axi_arbiter.axi_wr_m_w [1](if_input_wr.w);
    axi_arbiter.axi_wr_m_b [1](if_input_wr.b);

    axi_arbiter.axi_rd_m_ar[2](if_aux_rd.ar);
    axi_arbiter.axi_rd_m_r [2](if_aux_rd.r);
    axi_arbiter.axi_wr_m_aw[2](if_aux_wr.aw);
    axi_arbiter.axi_wr_m_w [2](if_aux_wr.w);
    axi_arbiter.axi_wr_m_b [2](if_aux_wr.b);


    axi_arbiter.axi_rd_s(if_data_rd);
    axi_arbiter.axi_wr_s(if_data_wr);

  }
};

#endif

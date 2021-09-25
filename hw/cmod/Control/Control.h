#ifndef __CONTROL_H__
#define __CONTROL_H__

/*
 * Controller SC Module for recieving configurations
 */

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_module.h>

#include "../include/Spec.h"
#include "../include/AxiSpec.h"

#include "../InputAxi/InputAxi.h"
#include "../MaskAxi/MaskAxi.h"
#include "../AuxAxi/AuxAxi.h"

SC_MODULE(Control) {
 public:
  sc_in<bool>  clk;
  sc_in<bool>  rst;

  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;
  typedef spec::aux_req_t aux_req_t;
  typedef spec::aux_rsp_t aux_rsp_t;
  typedef spec::IndexType IndexType;
  
  sc_out<bool> interrupt; 
  sc_out<NVUINT6> edgebert_dco_cc_sel;
  sc_out<NVUINT8> edgebert_ldo_res_sel;
   
  // AXI slave read/write ports
  typename spec::AxiConf::axi4_conf::read::template slave<>   if_axi_rd;
  typename spec::AxiConf::axi4_conf::write::template slave<>  if_axi_wr;


  // Input: IRQs (= #triggers)
  Connections::In<bool> IRQs[8]; 
  Connections::In<NVUINT2> enpy_status;
  Connections::In<NVUINT6> dco_sel_out;  //dvfs
  Connections::In<NVUINT8> ldo_sel_out;   //dvfs

  //0: LayerNorm, 1: SMax, 2: ElemAdd, 3: Enpy, 4: Dvfs
  Connections::Out<bool> start[5];

  Connections::Out<bool> flip_mem;
  Connections::Out<bool> use_axi[2];
  Connections::Out<bool> use_gb;
  Connections::Out<NVUINT4> gb_mode_config;
  Connections::Out<IndexType> base_output[3];
  Connections::Out<NVUINT12> reset_mode;

  // Output: Various Accelerator Config
  Connections::Out<spec::AccelConfig> accel_config;
  Connections::Out<spec::MatrixConfig> mat_config;
  Connections::Out<spec::InputBufferConfig> input_buffer_config[2];
  Connections::Out<spec::InputBufferBaseOffsetConfig> offset_config;
  Connections::Out<spec::PeriphConfig> periph_config;
  Connections::Out<spec::GBControlConfig> gbcontrol_config[2];
  Connections::Out<spec::EnpyConfig> enpy_config;
  Connections::Out<spec::DvfsConfig> dvfs_config; //dvfs 
  //Connections::Out<spec::VddConfig> vdd_config;   //dvfs
  Connections::Out<spec::DCOConfigA> dco_config_a;
  Connections::Out<spec::DCOConfigB> dco_config_b;
  Connections::Out<spec::DCOConfigC> dco_config_c;
  //
  Connections::Out<spec::LDOConfigA> ldo_config_a;
  Connections::Out<spec::LDOConfigB> ldo_config_b;
  Connections::Out<spec::LDOConfigC> ldo_config_c;
  Connections::Out<spec::LDOConfigD> ldo_config_d;



  // Output: Trigger
  Connections::Out<InputAxi::MasterTrig> input_rd_axi_start;
  Connections::Out<InputAxi::MasterTrig> input_wr_axi_start;
  Connections::Out<MaskAxi::MasterTrig> mask_rd_axi_start;
  Connections::Out<MaskAxi::MasterTrig> mask_wr_axi_start;
  Connections::Out<AuxAxi::MasterTrig> aux_rd_axi_start;
  Connections::Out<AuxAxi::MasterTrig> aux_wr_axi_start;

  //Connections::Combinational<spec::InputBufferConfig> input_buffer_config_wr[2]; //sc_thread
  //Connections::Combinational<spec::GBControlConfig> gbcontrol_config_wr[2]; //sc_thread
  //Connections::Combinational<bool> use_axi_wr[2]; //sc_thread
  

  // AXI rv channels
  Connections::Combinational<spec::AxiConf::SlaveToRV::Write> rv_in;
  Connections::Combinational<spec::AxiConf::SlaveToRV::Read>  rv_out;
  
  // AXI slave to ready/valid/address format
  spec::AxiConf::SlaveToRV   rv_inst;

  // 32-bit config regs, need 34 entries
  // 0--0x00: dummy
  // 1--0x04: start signal (write only), fire interrupt upon completion 
  // ----data=1 -> master mask read
  // ----data=2 -> master mask write
  // ----data=3 -> master input read
  // ----data=4 -> master input write
  // ----data=5 -> master aux read
  // ----data=6 -> master aux write
  // ----data=7 -> start PUModule
  // ----data=8 -> start LayerNorm / start[0]
  // ----data=9 -> start SMax / start[1]
  // ----data=10 -> start ElemAdd / start[2]
  // ----data=11 -> start Enpy / start[3]
  // ----data=12 -> start dvfs / start[4]
  // 2--0x08: flip_mem (1-bit)
  // 3--0x0C: AccelConfig
  // ----data[00-03]= is_relu (1 bit)
  // ----data[04-07]= is_bias (1 bit)
  // ----data[08-15]= weight_bias (8 bit)
  // ----data[16-19]= adf_accum_bias (3-bit)
  // ----data[20-27]= accum_right_shift (5-bit)
  // 4--0x10: MatrixConfig  // Size of Matrix Multiplications
  // ----data[00-09]= N0 (10-bit)
  // ----data[10-19]= N1 (10-bit)
  // ----data[20-31]= M (12-bit)
  // 5--0x14: InputBufferConfig
  // ----data[00-15]= base_input[0] (12-bit)
  // ----data[16-31]= base_input[1] (12-bit)
  // 6--0x18: InputBufferBaseOffsetConfig
  // ----data[00-15]= base_input_offset[0] (12-bit)
  // ----data[16-31]= base_input_offset[1] (12-bit)
  // 7--0x1C: PeriphConfig
  // ----data[00-06]= base_attn_span (7-bit)
  // ----data[07-14]= base_gamma (8-bit)
  // ----data[15-22]= base_beta (8-bit)
  // ----data[23-25]= adpbias_attn_span (3-bit)
  // ----data[26-28]= adpbias_gamma (3-bit)
  // ----data[29-31]= adpbias_beta (3-bit)
  // 8--0x20: GBControlConfig
  // ----data[00-07]= num_vector (8-bit)
  // ----data[08-15]= num_timestep (8-bit)
  // ----data[16-19]= adpbias_act1 (3-bit)
  // ----data[20-23]= adpbias_act2 (3-bit)
  // ----data[24-27]= adpbias_act3 (3-bit)
  // 9--0x24: EnpyConfig (16-bit), DvfsConfig (8-bit)
  // ----data[00-15]= enpy_threshold (16-bit)
  // 10--0x28:  mask_read_base 
  // 11--0x2C: mask_write_base 
  // 12--0x30: input_read_base 
  // 13--0x34: input_write_base 
  // 14--0x38: aux_read_base 
  // 15--0x3C: aux_write_base 
  // 16--0x40: num_words M_1 to read/write for Input/Mask AXI read/write 
  // ----data[00-15]= M_1 (15-bit)
  // 17--0x44: num_words M_1 to read/write for Aux AXI read/write 
  // ----data[00-07]= M_1 (8-bit)
  // 18--0x48: base_output 
  // 19--0x4C: use_axi (1-bit)
  // 20--0x50: use_gb (1-bit)
  // 21--0x54: gb_mode_config (4-bit)
  // 22--0x58: reset_mode (12-bit)
  // 23--0x5C: enpy_status (2-bit), dco_sel_out (6-bit), ldo_sel_out (8-bit)
  // 24--0x60: dummy
  // 25--0x64: DC01
  // 26--0x68: DC02
  // 27--0x6C: DC03
  //
  // 28--0x70: LD01
  // 29--0x74: LD02
  // 30--0x78: LD03
  // 31--0x7C: LD04

  NVUINT32 config_regs[32]; 

  SC_HAS_PROCESS(Control);
  Control(sc_module_name name)
     : sc_module(name),
       clk("clk"),
       rst("rst"),
       if_axi_rd("if_axi_rd"),
       if_axi_wr("if_axi_wr"),
       rv_inst("rv_inst")
  { 
    rv_inst.clk(clk);
    rv_inst.reset_bar(rst);
    rv_inst.if_axi_rd(if_axi_rd);
    rv_inst.if_axi_wr(if_axi_wr);
    rv_inst.if_rv_rd(rv_out);
    rv_inst.if_rv_wr(rv_in);

    SC_THREAD(ControlRun);
      sensitive << clk.pos();
      async_reset_signal_is(rst, false);
    
    SC_THREAD(InterruptRun);
      sensitive << clk.pos();
      async_reset_signal_is(rst, false);
  }

  void ControlRun() {
    // Reset 
    #pragma hls_unroll yes 
    for (unsigned i = 0; i < 32; i++) {
      config_regs[i] = 0;
    }

    // Reset AXI alave RV
    rv_out.ResetWrite(); // Output  
    rv_in.ResetRead();   // Input
    
    // Reset AXI slave 
    //if_axi_rd.reset();   // read config, debug
    //if_axi_wr.reset();   // write config, command

    // Reset 
    flip_mem.Reset();
    use_gb.Reset();

    #pragma hls_unroll yes     
    for (unsigned i = 0; i < 3; i++) base_output[i].Reset();

    gb_mode_config.Reset();

    reset_mode.Reset();
    accel_config.Reset();
    mat_config.Reset();  

    #pragma hls_unroll yes 
    for (unsigned i = 0; i < 2; i++) {
        input_buffer_config[i].Reset();
        gbcontrol_config[i].Reset();
        use_axi[i].Reset();
    }
    #pragma hls_unroll yes 
    for (unsigned i = 0; i < 5; i++) start[i].Reset(); 
    
    offset_config.Reset();
    periph_config.Reset();
    enpy_config.Reset();
    dvfs_config.Reset();
    //
    dco_config_a.Reset();
    dco_config_b.Reset();
    dco_config_c.Reset();
    //
    ldo_config_a.Reset();
    ldo_config_b.Reset();
    ldo_config_c.Reset();
    ldo_config_d.Reset();
    //vdd_config.Reset();
    enpy_status.Reset();
    dco_sel_out.Reset();
    ldo_sel_out.Reset();

    input_rd_axi_start.Reset();
    input_wr_axi_start.Reset();
    mask_rd_axi_start.Reset();
    mask_wr_axi_start.Reset();
    aux_rd_axi_start.Reset();
    aux_wr_axi_start.Reset();
   
    edgebert_dco_cc_sel.write(63);
    edgebert_ldo_res_sel.write(0);

    // This is controller module, no need II=1
    while(1) {
      spec::AxiConf::SlaveToRV::Write rv_in_reg;
      spec::AxiConf::SlaveToRV::Read  rv_out_reg;
      NVUINT5 index; // 0-31
      bool is_read = 0;
      bool is_write = 0;
      
      // TODO: implement hardware to block config when acc is running
      if (rv_in.PopNB(rv_in_reg)) {
        // use bit [6-2] for 32-bit = 4-byte word size
        index = nvhls::get_slc<5>(rv_in_reg.addr, 2);
         
        //cout << "debug index, data: " << index << "\t" << rv_in_reg.data << endl;
	      
        if (rv_in_reg.rw == 1) { // Write mode
          is_write = 1;
          config_regs[index] = rv_in_reg.data;
        } 
	      else { // Read mode
          is_read = 1;
          rv_out_reg.data = config_regs[index];
	      }
        wait();
      }

      NVUINT2 enpy_status_reg;
      if (enpy_status.PopNB(enpy_status_reg)) {
         config_regs[23].set_slc<2>(0, enpy_status_reg);
      }
       
      NVUINT6 freq_opt_reg;
      if (dco_sel_out.PopNB(freq_opt_reg)) {
         config_regs[23].set_slc<6>(2, freq_opt_reg); 
         edgebert_dco_cc_sel.write(nvhls::get_slc<6>(freq_opt_reg, 0));
      }

      NVUINT8 vdd_opt_reg;
      if (ldo_sel_out.PopNB(vdd_opt_reg)) {
         config_regs[23].set_slc<8>(10, vdd_opt_reg); 
         edgebert_ldo_res_sel.write(nvhls::get_slc<8>(vdd_opt_reg, 0));
      }      

      // flip_mem 
      if ((is_write==1) && (index==2)) {
        bool flip_mem_reg = (config_regs[2] > 0); 
        flip_mem.Push(flip_mem_reg);       
      } 
      // use_axi
      if ((is_write==1) && (index==19)) {
        bool use_axi_reg = (config_regs[19] > 0); 
        #pragma hls_unroll yes 
        for (unsigned i = 0; i < 2; i++) use_axi[i].Push(use_axi_reg);
      }
      // use_gb
      if ((is_write==1) && (index==20)) {
        bool use_gb_reg = (config_regs[20] > 0); 
        use_gb.Push(use_gb_reg);
      }
      // gb_mode_config
      if ((is_write==1) && (index==21)) {
        NVUINT4 gb_mode_config_reg = nvhls::get_slc<4>(config_regs[21], 0);
        gb_mode_config.Push(gb_mode_config_reg);
      }
      // reset_mode
      if ((is_write==1) && (index==22)) {
         NVUINT12 reset_mode_reg = nvhls::get_slc<12>(config_regs[22], 0);
         reset_mode.Push(reset_mode_reg);   
      }
      // AccelConfig
      if ((is_write==1) && (index==3)) {
         spec::AccelConfig accel_config_reg;
         accel_config_reg.is_relu = nvhls::get_slc<1>(config_regs[3], 0);
         accel_config_reg.is_bias = nvhls::get_slc<1>(config_regs[3], 4);
         accel_config_reg.weight_bias = nvhls::get_slc<8>(config_regs[3], 8);
         accel_config_reg.adf_accum_bias = nvhls::get_slc<3>(config_regs[3], 16);
         accel_config_reg.accum_right_shift = nvhls::get_slc<5>(config_regs[3], 20);
         accel_config.Push(accel_config_reg);    
      }
      // InputBufferConfig
      if ((is_write==1) && (index==5)) {
         spec::InputBufferConfig input_buffer_config_reg;
         input_buffer_config_reg.base_input[0] = nvhls::get_slc<12>(config_regs[5], 0);
         input_buffer_config_reg.base_input[1] = nvhls::get_slc<12>(config_regs[5], 16);
         #pragma hls_unroll yes 
         for (unsigned i = 0; i < 2; i++) input_buffer_config[i].Push(input_buffer_config_reg); 
      }
      // OffsetConfig
      if ((is_write==1) && (index==6)) {
         spec::InputBufferBaseOffsetConfig offsetconfig_reg;
         offsetconfig_reg.base_input_offset[0] = nvhls::get_slc<12>(config_regs[6], 0);
         offsetconfig_reg.base_input_offset[1] = nvhls::get_slc<12>(config_regs[6], 16);
         offset_config.Push(offsetconfig_reg);
      }
      // PeriphConfig
      if ((is_write==1) && (index==7)) {
         spec::PeriphConfig perifconfig_reg;
         perifconfig_reg.base_attn_span = nvhls::get_slc<7>(config_regs[7], 0);
         perifconfig_reg.base_gamma = nvhls::get_slc<8>(config_regs[7], 7);
         perifconfig_reg.base_beta = nvhls::get_slc<8>(config_regs[7], 15);
         perifconfig_reg.adpbias_attn_span = nvhls::get_slc<3>(config_regs[7], 23);
         perifconfig_reg.adpbias_gamma = nvhls::get_slc<3>(config_regs[7], 26); 
         perifconfig_reg.adpbias_beta = nvhls::get_slc<3>(config_regs[7], 29); 
         periph_config.Push(perifconfig_reg);
      }
      // GBControlConfig
      if ((is_write==1) && (index==8)) {
         spec::GBControlConfig gbcontrol_config_reg;
         gbcontrol_config_reg.num_vector   = nvhls::get_slc<8>(config_regs[8], 0);
         gbcontrol_config_reg.num_timestep = nvhls::get_slc<8>(config_regs[8], 8);
         gbcontrol_config_reg.adpbias_act1 = nvhls::get_slc<3>(config_regs[8], 16);
         gbcontrol_config_reg.adpbias_act2 = nvhls::get_slc<3>(config_regs[8], 20);
         gbcontrol_config_reg.adpbias_act3 = nvhls::get_slc<3>(config_regs[8], 24);
         #pragma hls_unroll yes
         for (unsigned i = 0; i < 2; i++) gbcontrol_config[i].Push(gbcontrol_config_reg);
      } 
      // EnpyConfig && DVFSConfig
      if ((is_write==1) && (index==9)) {
         spec::EnpyConfig enpy_config_reg;
         enpy_config_reg.enpy_threshold = nvhls::get_slc<16>(config_regs[9], 0);
 
         spec::DvfsConfig dvfs_config_reg;
         dvfs_config_reg.enpy_scale  = nvhls::get_slc<8>(config_regs[9], 16);

         enpy_config.Push(enpy_config_reg);
         dvfs_config.Push(dvfs_config_reg);
      }
      // base_output
      if ((is_write==1) && (index==18)) {
         IndexType base_output_reg;
         base_output_reg = nvhls::get_slc<12>(config_regs[18], 0);
         #pragma hls_unroll yes
         for (unsigned i = 0; i < 3; i++) base_output[i].Push(base_output_reg);
      }
      // DvfsConfig
      /*if ((is_write==1) && (index==24)) {
         spec::DvfsConfig dvfs_config_reg;
         dvfs_config_reg.enpy_scale  = nvhls::get_slc<8>(config_regs[24], 0);
         dvfs_config.Push(dvfs_config_reg);
      }*/ 
      // VddConfig
      /*if ((is_write==1) && (index==25)) {
         spec::VddConfig vdd_config_reg;
         vdd_config_reg.vdd_scale  = nvhls::get_slc<8>(config_regs[25], 0);
         vdd_config_reg.base_vdd = nvhls::get_slc<8>(config_regs[25], 8);
         vdd_config.Push(vdd_config_reg);
      } */

      // DCOConfigA
      if ((is_write==1) && (index==25)) {
         spec::DCOConfigA dco_config_a_config_reg;
         dco_config_a_config_reg.dco_val0 =  nvhls::get_slc<6>(config_regs[25], 0);
         dco_config_a_config_reg.dco_val1 =  nvhls::get_slc<6>(config_regs[25], 6);
         dco_config_a_config_reg.dco_val2 =  nvhls::get_slc<6>(config_regs[25], 12);
         dco_config_a_config_reg.dco_val3 =  nvhls::get_slc<6>(config_regs[25], 18);
         dco_config_a_config_reg.dco_val4 =  nvhls::get_slc<6>(config_regs[25], 24);
         dco_config_a.Push(dco_config_a_config_reg);
      } 


      // DCOConfigB
      if ((is_write==1) && (index==26)) {
         spec::DCOConfigB dco_config_b_config_reg;
         dco_config_b_config_reg.dco_val5 =  nvhls::get_slc<6>(config_regs[26], 0);
         dco_config_b_config_reg.dco_val6 =  nvhls::get_slc<6>(config_regs[26], 6);
         dco_config_b_config_reg.dco_val7 =  nvhls::get_slc<6>(config_regs[26], 12);
         dco_config_b_config_reg.dco_val8 =  nvhls::get_slc<6>(config_regs[26], 18);
         dco_config_b_config_reg.dco_val9 =  nvhls::get_slc<6>(config_regs[26], 24);
         dco_config_b.Push(dco_config_b_config_reg);
      } 

      // DCOConfigC
      if ((is_write==1) && (index==27)) {
         spec::DCOConfigC dco_config_c_config_reg;
         dco_config_c_config_reg.dco_val10 =  nvhls::get_slc<6>(config_regs[27], 0);
         dco_config_c_config_reg.dco_val11 =  nvhls::get_slc<6>(config_regs[27], 6);
         dco_config_c_config_reg.dco_val12 =  nvhls::get_slc<6>(config_regs[27], 12);
         dco_config_c_config_reg.dco_val13 =  nvhls::get_slc<6>(config_regs[27], 18);
         dco_config_c_config_reg.dco_val14 =  nvhls::get_slc<6>(config_regs[27], 24);
         dco_config_c.Push(dco_config_c_config_reg);
      } 

      // LDOConfigA
      if ((is_write==1) && (index==28)) {
         spec::LDOConfigA ldo_config_a_config_reg;
         ldo_config_a_config_reg.ldo_val0 =  nvhls::get_slc<8>(config_regs[28], 0);
         ldo_config_a_config_reg.ldo_val1 =  nvhls::get_slc<8>(config_regs[28], 8);
         ldo_config_a_config_reg.ldo_val2 =  nvhls::get_slc<8>(config_regs[28], 16);
         ldo_config_a_config_reg.ldo_val3 =  nvhls::get_slc<8>(config_regs[28], 24);
         ldo_config_a.Push(ldo_config_a_config_reg);
      } 

      // LDOConfigB
      if ((is_write==1) && (index==29)) {
         spec::LDOConfigB ldo_config_b_config_reg;
         ldo_config_b_config_reg.ldo_val4 =  nvhls::get_slc<8>(config_regs[29], 0);
         ldo_config_b_config_reg.ldo_val5 =  nvhls::get_slc<8>(config_regs[29], 8);
         ldo_config_b_config_reg.ldo_val6 =  nvhls::get_slc<8>(config_regs[29], 16);
         ldo_config_b_config_reg.ldo_val7 =  nvhls::get_slc<8>(config_regs[29], 24);
         ldo_config_b.Push(ldo_config_b_config_reg);
      } 

      // LDOConfigC
      if ((is_write==1) && (index==30)) {
         spec::LDOConfigC ldo_config_c_config_reg;
         ldo_config_c_config_reg.ldo_val8 =  nvhls::get_slc<8>(config_regs[30], 0);
         ldo_config_c_config_reg.ldo_val9 =  nvhls::get_slc<8>(config_regs[30], 8);
         ldo_config_c_config_reg.ldo_val10 =  nvhls::get_slc<8>(config_regs[30], 16);
         ldo_config_c_config_reg.ldo_val11 =  nvhls::get_slc<8>(config_regs[30], 24);
         ldo_config_c.Push(ldo_config_c_config_reg);
      } 

      // LDOConfigD
      if ((is_write==1) && (index==31)) {
         spec::LDOConfigD ldo_config_d_config_reg;
         ldo_config_d_config_reg.ldo_val12 =  nvhls::get_slc<8>(config_regs[31], 0);
         ldo_config_d_config_reg.ldo_val13 =  nvhls::get_slc<8>(config_regs[31], 8);
         ldo_config_d_config_reg.ldo_val14 =  nvhls::get_slc<8>(config_regs[31], 16);
         ldo_config_d_config_reg.ldo_val15 =  nvhls::get_slc<8>(config_regs[31], 24);
         ldo_config_d.Push(ldo_config_d_config_reg);
      } 

      // Start command, write only 
      if ((is_write==1) && (index==1)) {
        switch (config_regs[1]) {
          case 1: { // Mask Master Read (load from outside)
            MaskAxi::MasterTrig trig_reg;
            trig_reg.M_1  = nvhls::get_slc<12>(config_regs[16], 0);
            trig_reg.base_addr = config_regs[10]; // base address of mask read 
            mask_rd_axi_start.Push(trig_reg);
            break;
          }  
          case 2: { // Mask Master Write (store to outside)
            MaskAxi::MasterTrig trig_reg;
            trig_reg.M_1  = nvhls::get_slc<12>(config_regs[16], 0);
            trig_reg.base_addr = config_regs[11]; // base address of mask write 
            mask_wr_axi_start.Push(trig_reg);
            break;
          }  
          case 3: { // Input Master Read (load from outside)
            InputAxi::MasterTrig trig_reg;
            trig_reg.M_1  = nvhls::get_slc<12>(config_regs[16], 0);
            trig_reg.base_addr = config_regs[12]; // base address of input read
            input_rd_axi_start.Push(trig_reg);
            break;
          }  
          case 4: { // Input Master Write (store to outside)
            InputAxi::MasterTrig trig_reg;
            trig_reg.M_1  = nvhls::get_slc<12>(config_regs[16], 0);
            trig_reg.base_addr = config_regs[13]; // base address of input write
            input_wr_axi_start.Push(trig_reg);
            break;
          }
          case 5: { // Aux Master Read (load from outside)
            AuxAxi::MasterTrig trig_reg;
            trig_reg.M_1  = nvhls::get_slc<8>(config_regs[17], 0);
            trig_reg.base_addr = config_regs[14]; // base address of aux read
            aux_rd_axi_start.Push(trig_reg);
            break;
          }  
          case 6: { // Aux Master Write (store to outside)
            AuxAxi::MasterTrig trig_reg;
            trig_reg.M_1  = nvhls::get_slc<8>(config_regs[17], 0);
            trig_reg.base_addr = config_regs[15]; // base address of aux write
            aux_wr_axi_start.Push(trig_reg);
            break;
          }
          case 7: { // Start PUModule
            spec::MatrixConfig matrixconfig_reg;
            matrixconfig_reg.N0 = nvhls::get_slc<10>(config_regs[4], 0);
            matrixconfig_reg.N1 = nvhls::get_slc<10>(config_regs[4], 10);
            matrixconfig_reg.M = nvhls::get_slc<12>(config_regs[4], 20);
            mat_config.Push(matrixconfig_reg);
            break;
          }
          case 8: { // Start LayerNorm
             start[0].Push(1);
             break;
          }
          case 9: { // Start SMax
             start[1].Push(1);
             break;
          }
          case 10: { // Start ElemAdd
             start[2].Push(1);
             break;
          }
          case 11: { // Start Enpy
             start[3].Push(1);
             break;
          }
          case 12: { // Start Dvfs
             start[4].Push(1);
             break;
          }
          default: 
            break; // no IRQ, may lead to error
        }
      }
      wait();
   
      // Read response
      if (is_read) {
        //cout << "rv_out.data: " << rv_out_reg.data << endl;
        rv_out.Push(rv_out_reg);
      }
      wait(); 
    }
  }

  // Simple IRQ handler for sending IRQ trigger
  // IRQ[0]: master mask read   
  // IRQ[1]: master mask write   
  // IRQ[2]: master input read
  // IRQ[3]: master input write
  // IRQ[4]: master aux read 
  // IRQ[5]: master aux write 
  // IRQ[6]: com_IRQ / PUModule done 
  // IRQ[7]: done_out / GBModule done 
  void InterruptRun() {
    #pragma hls_unroll yes 
    for (unsigned i = 0; i < 8; i++) {
      IRQs[i].Reset();
    }
    // Reset IRO
    interrupt.write(false);

    while(1) {
      NVUINTW(8) irq_valids = 0;
      #pragma hls_unroll yes
      for (int i = 0; i < 8; i++) {
        bool tmp;
        irq_valids[i] = IRQs[i].PopNB(tmp);
      }
      if (irq_valids != 0) {
        CDCOUT(sc_time_stamp()  << name() << " Interrupt Received in Control! " << endl, 0);
        interrupt.write(true);
        #pragma hls_pipeline_init_interval 1
        for (unsigned i = 0; i < 10; i++) {
          wait();
        }
        interrupt.write(false);
        wait();
      }
      wait();
    }
  }
};

#endif

/*
 * EdgeBERT TopAccel testbench
 */

#include "TopAccel.h"
#include "../include/Spec.h"
#include "../include/AxiSpec.h"
#include "../include/utils.h"
//#include "testbench/Master.h"

#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>

#include <vector>

#define NVHLS_VERIFY_BLOCKS (TopAccel)
#include <nvhls_verify.h>
using namespace::std;

#include <testbench/nvhls_rand.h>
#include <testbench/Slave.h>
//#include "testbench/Slave.h"
//#include <axi/testbench/Slave.h>

// W/I/O dimensions
//const static int N = spec::N;
const static int N = 64;
const static int M = N;

// bias left shift, 
const static bool  is_relu = 1;
const static bool  is_bias = 1;
const static int   weight_bias = 3;
const static int   adf_accum_bias = 2;
const static int   accum_right_shift = 8;

const static int base_input0 = 0;
const static int base_input1 = 0;

const static int base_attn_span = 0;
const static int base_gamma = 8;
const static int base_beta = 56;
const static int adpbias_attn_span = 2;
const static int adpbias_gamma = 2;
const static int adpbias_beta = 2;

const static int num_vector = 32;
const static int num_timestep = 2;
const static int adpbias_act1 = 2;
const static int adpbias_act2 = 2;
const static int adpbias_act3 = 2;

const static int base_output = 1024;

const static int axi_on = 1;
const static int axi_off = 0;

const static int enpy_scale = 126;
const static int target_time = 16;
const static int base_dvfs = 0;

const static int vdd_scale = 16;
const static int base_vdd = 8;

// base address
const static unsigned mask_rd_base = 0x4000;
const static unsigned mask_wr_base = 0x8000;
const static unsigned input_rd_base = 0xC000;
const static unsigned input_wr_base = 0x10000;
const static unsigned aux_rd_base = 0x14000;
const static unsigned aux_wr_base = 0x18000;

// Matrix configurations
const static unsigned N0 = 32;
const static unsigned N1 = 32;
const static unsigned M_mat = 32;

const static int enpy_threshold1 = 3957;
const static int enpy_threshold2 = 3959;

int ERROR = 0;

vector<int>   Count(N, 0);
SC_MODULE (Master) {
  sc_in<bool> clk;
  sc_in<bool> rst;
  sc_in<bool> interrupt;
  sc_in<NVUINT6> edgebert_dco_cc_sel;
  sc_in<NVUINT8> edgebert_ldo_res_sel;

  typename spec::AxiConf::axi4_conf::read::template master<> if_rd;
  typename spec::AxiConf::axi4_conf::write::template master<> if_wr;

  SC_CTOR(Master)
        : if_rd("if_rd"), 
          if_wr("if_wr")
  {
    SC_THREAD(Run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void MasterAccess(const bool rw, NVUINT32 addr, NVUINT32& data) {
    typename spec::AxiConf::axi4_conf::AddrPayload    a_pld;
    typename spec::AxiConf::axi4_conf::ReadPayload   rd_pld;
    typename spec::AxiConf::axi4_conf::WritePayload  wr_pld;
    typename spec::AxiConf::axi4_conf::WRespPayload   b_pld;

    a_pld.len = 0;
    a_pld.addr = addr;

    if (rw == 0) {  // read and set data 
      if_rd.ar.Push(a_pld);
      rd_pld = if_rd.r.Pop();
      data = rd_pld.data;
      wait();
    }
    else {          // write
      if_wr.aw.Push(a_pld);
      wait();
      wr_pld.data = data;
      wr_pld.wstrb = ~0;
      wr_pld.last = 1;
      if_wr.w.Push(wr_pld);
      wait();
      if_wr.b.Pop();
      wait();
    }
  }
  void Run() {
    if_rd.reset();
    if_wr.reset();

    wait(100);

    NVUINT32 data=0, data_read=0;
   
    cout << "@" << sc_time_stamp() << " Start base_output 0x48 and use_gb 0x50 and reset_mode 0x58" << endl;
    data = 0;
    data += base_output;
    MasterAccess(1, 0x48, data); 
    MasterAccess(0, 0x48, data_read); 
    assert (data == data_read);

    data = 0;
    MasterAccess(1, 0x50, data); 
    MasterAccess(0, 0x50, data_read); 
    assert (data == data_read);

    data = 0x81; //dec0=N0, dec1=N1;
    MasterAccess(1, 0x58, data); 
    MasterAccess(0, 0x58, data_read); 
    assert (data == data_read);

    cout << "@" << sc_time_stamp() << " Finish base_output 0x48 and use_gb 0x50 and reset_mode 0x58" << endl;
    
    cout << "@" << sc_time_stamp() << " Start AccelConfig 0x0c " << endl;
    data = 0;
    data += is_relu;
    data += is_bias << 4;
    data += weight_bias << 8;
    data += adf_accum_bias << 16;
    data += accum_right_shift << 20;
    MasterAccess(1, 0x0C, data);   
    MasterAccess(0, 0x0C, data_read);
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish AccelConfig 0x0c " << endl;

    cout << "@" << sc_time_stamp() << " Start InputBufferConfig 0x14 and BufferOffsetConfig 0x18" << endl;
    data = 0;
    data += base_input0;
    data += base_input1 << 16;
    MasterAccess(1, 0x14, data); 
    MasterAccess(0, 0x14, data_read); 
    assert (data == data_read);

    MasterAccess(1, 0x18, data); 
    MasterAccess(0, 0x18, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish InputBufferConfig 0x14 and BufferOffsetConfig 0x18" << endl;
    
    cout << "@" << sc_time_stamp() << " Start num_words M_1 for Input/Mask AXI 0x40" << endl;
    data = 0;
    data += (M-1);
    MasterAccess(1, 0x40, data); 
    MasterAccess(0, 0x40, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish num_words M_1 for Input/Mask AXI 0x40" << endl;       

    cout << "@" << sc_time_stamp() << " Start mask_read_base (0x28) and input_read_base (0x30)" << endl;
    data = mask_rd_base;
    MasterAccess(1, 0x28, data); 
    MasterAccess(0, 0x28, data_read); 
    assert (data == data_read);
    
    data = input_rd_base;
    MasterAccess(1, 0x30, data); 
    MasterAccess(0, 0x30, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish mask_read_base (0x28) and input_read_base (0x30)" << endl;

    cout << "@" << sc_time_stamp() << " Start InputMem0/1 Master Read 0x04-data=3 " << endl;  
    data = 0x0;
    MasterAccess(1, 0x4C, data); //use_axi = 0;
    MasterAccess(0, 0x4C, data_read); 
    assert (data == data_read);    
 
    data = 0x0;
    MasterAccess(1, 0x08, data); //flip_mem = 0;
    MasterAccess(0, 0x08, data_read); 
    assert (data == data_read);      

    data = 0x03;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    assert (data == data_read); 
    // wait for IRQ
    while (interrupt.read() == 0) wait();

    data = 0x1;
    MasterAccess(1, 0x08, data); //flip_mem = 1;
    MasterAccess(0, 0x08, data_read); 
    assert (data == data_read);      

    data = 0x03;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    // wait for IRQ
    while (interrupt.read() == 0) wait();
    cout << "@" << sc_time_stamp() << " Finish InputMem0/1 Master Read 0x04-data=3 " << endl;  

    cout << "@" << sc_time_stamp() << " Start MaskMem0/1 Master Read 0x04-data=1 " << endl;  
    data = 0x0;
    MasterAccess(1, 0x08, data); //flip_mem = 0;
    MasterAccess(0, 0x08, data_read); 
    assert (data == data_read);      

    data = 0x01;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    assert (data == data_read); 
    // wait for IRQ
    while (interrupt.read() == 0) wait();

    data = 0x1;
    MasterAccess(1, 0x08, data); //flip_mem = 1;
    MasterAccess(0, 0x08, data_read); 
    assert (data == data_read);      

    data = 0x01;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    // wait for IRQ
    while (interrupt.read() == 0) wait();
    cout << "@" << sc_time_stamp() << " Finish MaskMem0/1 Master Read 0x04-data=1 " << endl;      

    wait(10);
    cout << "@" << sc_time_stamp() << " Start num_words M_1 for Aux AXI 0x44" << endl;
    data = 0;
    data += (M-1);
    MasterAccess(1, 0x44, data); 
    MasterAccess(0, 0x44, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish num_words M_1 for Aux AXI 0x44" << endl;       

    cout << "@" << sc_time_stamp() << " Start aux_read_base (0x38)" << endl;
    data = aux_rd_base;
    MasterAccess(1, 0x38, data); 
    MasterAccess(0, 0x38, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish aux_read_base (0x38)" << endl;

    cout << "@" << sc_time_stamp() << " Start Aux Master Read 0x04-data=5 " << endl;  
    data = 0x05;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    // wait for IRQ
    while (interrupt.read() == 0) wait();
    cout << "@" << sc_time_stamp() << " Finish Aux Master Read 0x04-data=5 " << endl;  
 
    cout << "@" << sc_time_stamp() << " Start MatrixConfig 0x10" << endl;     
    data = 0x0;
    data += N0;
    data += N1 << 10;
    data += M_mat << 20;
    MasterAccess(1, 0x10, data);   
    MasterAccess(0, 0x10, data_read);
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish MatrixConfig 0x10" << endl;     


    // start computation 
    //cout << "@" << sc_time_stamp() << " Start PUModule Computation" << endl;
    /*data = 0x0;
    MasterAccess(1, 0x4C, data); //use_axi = 0;
    MasterAccess(0, 0x4C, data_read); 
    assert (data == data_read); */ 

    wait();
    cout << "@" << sc_time_stamp() << " Start PUModule Computation" << endl;
    data = 0x07;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    assert (data == data_read); 
    // wait for IRQ
    while (interrupt.read() == 0) wait();
    cout << "@" << sc_time_stamp() << " Finish PUModule Computation" << endl;
    wait(100); 

    /*cout << "@" << sc_time_stamp() << " Start mask_write_base (0x2C) and input_write_base (0x34)" << endl;
    data = mask_wr_base;
    MasterAccess(1, 0x2C, data); 
    MasterAccess(0, 0x2C, data_read); 
    assert (data == data_read);
    
    data = input_wr_base;
    MasterAccess(1, 0x34, data); 
    MasterAccess(0, 0x34, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish mask_write_base (0x2C) and input_write_base (0x34)" << endl;

    cout << "@" << sc_time_stamp() << " Start MaskMem1 Master Write" << endl;
    data = 0x1;
    MasterAccess(1, 0x4C, data); //use_axi = 1;
    MasterAccess(0, 0x4C, data_read); 
    assert (data == data_read);  

    data = 0x02;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    cout << "@" << sc_time_stamp() << " Finish MaskMem1 Master Write" << endl;*/ 


    cout << "@" << sc_time_stamp() << " Start use_gb 0x50 and reset_mode 0x58" << endl;
    wait(10);
    data = 0x0;
    MasterAccess(1, 0x08, data); //flip_mem = 0;
    MasterAccess(0, 0x08, data_read); 
    assert (data == data_read);    

    data = 0x1;
    MasterAccess(1, 0x50, data); //use_gb = 1
    MasterAccess(0, 0x50, data_read); 
    assert (data == data_read);

    data = 0x04;
    MasterAccess(1, 0x58, data); //reset_mode = 0x02 layernorm pu mode
    MasterAccess(0, 0x58, data_read); 
    assert (data == data_read);

    /*data = 0;
    cout << "check1a" << endl;
    MasterAccess(1, 0x4C, data); //use_axi = 0;
    MasterAccess(0, 0x4C, data_read); 
    assert (data == data_read); */  
    cout << "@" << sc_time_stamp() << " Finish use_gb 0x50 and reset_mode 0x58" << endl;

    cout << "@" << sc_time_stamp() << " Start gbmode_config" << endl;
    data = 0x1;
    MasterAccess(1, 0x54, data); //gb_mode_config = 1 // layernorm gb mode
    MasterAccess(0, 0x54, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Start gbmode_config" << endl;

    cout << "@" << sc_time_stamp() << " Start PeriphConfig 0x1C" << endl;
    data = 0;
    data += base_attn_span;
    data += base_gamma << 7; 
    data += base_beta << 15;
    data += adpbias_attn_span << 23;
    data += adpbias_gamma << 26;
    data += adpbias_beta << 29;
    MasterAccess(1, 0x1C, data); 
    MasterAccess(0, 0x1C, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish PeriphConfig 0x1C" << endl;

    cout << "@" << sc_time_stamp() << " Start GBControlConfig 0x20" << endl;
    data = 0;
    data += num_vector;
    data += num_timestep << 8; 
    data += adpbias_act1 << 16;
    data += adpbias_act2 << 20;
    data += adpbias_act3 << 24;
    MasterAccess(1, 0x20, data); 
    MasterAccess(0, 0x20, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish GBControlConfig 0x20" << endl;

    cout << "@" << sc_time_stamp() << " Start InputBufferConfig 0x14" << endl;
    data = 0;
    data += base_input0;
    data += base_input1 << 16;
    MasterAccess(1, 0x14, data); 
    MasterAccess(0, 0x14, data_read); 
    assert (data == data_read); 

    MasterAccess(1, 0x18, data); 
    MasterAccess(0, 0x18, data_read); 
    assert (data == data_read); 

    cout << "@" << sc_time_stamp() << " Finish InputBufferConfig 0x14" << endl;


    wait();
    cout << "@" << sc_time_stamp() << " Start LayerNorm Computation" << endl;
    data = 0x8;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    while (interrupt.read() == 0) wait();
    cout << "@" << sc_time_stamp() << " End LayerNorm Computation" << endl;  


    cout << "@" << sc_time_stamp() << " Start config for SMax" << endl;
    data = 0x08; //XXX: make sure flip_mem = 0
    MasterAccess(1, 0x58, data); //reset_mode = 0x03 smax pu mode, XXX: make sure flip_mem = 0
    MasterAccess(0, 0x58, data_read); 
    assert (data == data_read);

    data = 0x02;
    MasterAccess(1, 0x54, data); //gb_mode_config = 2 // SMax gb mode
    MasterAccess(0, 0x54, data_read); 
    assert (data == data_read);

    data = 0;
    data += base_attn_span;
    data += base_gamma << 7; 
    data += base_beta << 15;
    data += adpbias_attn_span << 23;
    data += adpbias_gamma << 26;
    data += adpbias_beta << 29;
    MasterAccess(1, 0x1C, data); 
    MasterAccess(0, 0x1C, data_read); 
    assert (data == data_read);

    data = 0;
    data += num_vector;
    data += num_timestep << 8; 
    data += adpbias_act1 << 16;
    data += adpbias_act2 << 20;
    data += adpbias_act3 << 24;
    MasterAccess(1, 0x20, data); 
    MasterAccess(0, 0x20, data_read); 
    assert (data == data_read);

    data = 0;
    data += base_input0;
    data += base_input1 << 16;
    MasterAccess(1, 0x14, data); 
    MasterAccess(0, 0x14, data_read); 
    assert (data == data_read);

    MasterAccess(1, 0x18, data); 
    MasterAccess(0, 0x18, data_read); 
    assert (data == data_read); 

    cout << "@" << sc_time_stamp() << " Finish config for SMax" << endl;

    wait();
    cout << "@" << sc_time_stamp() << " Start SMax Computation" << endl;
    data = 0x0;
    MasterAccess(1, 0x08, data); //flip_mem = 0;
    MasterAccess(0, 0x08, data_read); 
    assert (data == data_read);    

    data = 0x9;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    while (interrupt.read() == 0) wait();
    cout << "@" << sc_time_stamp() << " End SMax Computation" << endl;  

    wait();
    cout << "@" << sc_time_stamp() << " Start config for Enpy" << endl;
    //wait(10);
    data = 0x10;
    MasterAccess(1, 0x58, data); 
    MasterAccess(0, 0x58, data_read); 
    assert (data == data_read);

    data = 0x08;
    MasterAccess(1, 0x54, data); //gb_mode_config = 8 // Enpy gb mode
    MasterAccess(0, 0x54, data_read); 
    assert (data == data_read); 

    //cout << "check0a" << endl;

    /*data = 0x0;
    MasterAccess(1, 0x4C, data); //use_axi = 0;
    MasterAccess(0, 0x4C, data_read); 
    assert (data == data_read);  
    cout << "check0b" << endl; */

    //data = 0x0;
    //MasterAccess(1, 0x08, data); //flip_mem = 0;
    //MasterAccess(0, 0x08, data_read); 
    //assert (data == data_read);  
    //cout << "check0c" << endl;

    //data = 0x1;
    //MasterAccess(1, 0x50, data); //use_gb = 1
    //MasterAccess(0, 0x50, data_read); 
    //assert (data == data_read);
    //cout << "check0d" << endl;

    data = 0;
    data += enpy_threshold1;
    data += enpy_scale << 16;
    MasterAccess(1, 0x24, data); //enpy_config;
    MasterAccess(0, 0x24, data_read); 
    assert (data == data_read);  
    //cout << "check0e" << endl;

    data = 0;
    data += num_vector;
    data += num_timestep << 8; 
    data += adpbias_act1 << 16;
    data += adpbias_act2 << 20;
    data += adpbias_act3 << 24;
    MasterAccess(1, 0x20, data); 
    MasterAccess(0, 0x20, data_read); 
    assert (data == data_read);

    //wait(10);
    data = 0;
    data += base_input0;
    data += base_input1 << 16;
    MasterAccess(1, 0x14, data); 
    MasterAccess(0, 0x14, data_read); 
    assert (data == data_read); 

    MasterAccess(1, 0x18, data); 
    MasterAccess(0, 0x18, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish config for Enpy" << endl;

    wait();
    cout << "@" << sc_time_stamp() << " Start Enpy Computation" << endl;
    data = 0xB;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    while (interrupt.read() == 0) wait();
    cout << "@" << sc_time_stamp() << " End Enpy Computation" << endl; 

    cout << "@" << sc_time_stamp() << " Start reading enpy status" << endl;
    MasterAccess(0, 0x5C, data_read);
    cout << "@" << sc_time_stamp() << " Finished Reading enpy status: " << data_read << endl;

    wait();
    cout << "@" << sc_time_stamp() << " Start config for Dvfs" << endl;

    data = 0x04;
    MasterAccess(1, 0x54, data); //gb_mode_config = 4 // DVFS gb mode
    MasterAccess(0, 0x54, data_read); 
    assert (data == data_read);

    /*cout << "@" << sc_time_stamp() << " Start Dvfsconfig 0x60" << endl;
    data = 0;
    data += enpy_scale;
    //data += target_time << 8; 
    //data += base_dvfs << 24;
    MasterAccess(1, 0x60, data); 
    MasterAccess(0, 0x60, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish DvfsConfig 0x60" << endl; */

    cout << "@" << sc_time_stamp() << " Start DCOConfig 0x64 - Ox68 - 0x6C" << endl;
    data = 0;
    data += 0xAFAFAFAF;
    MasterAccess(1, 0x64, data); 
    MasterAccess(0, 0x64, data_read);  
    assert (data == data_read);

    data= 0;
    data += 0x1D1D1D1D;
    MasterAccess(1, 0x68, data); 
    MasterAccess(0, 0x68, data_read);  
    assert (data == data_read);

    data= 0;
    data += 0xB7B7B7B7;
    MasterAccess(1, 0x6C, data); 
    MasterAccess(0, 0x6C, data_read);  
    assert (data == data_read);

    cout << "@" << sc_time_stamp() << " Finish DCOConfig 0x64 - Ox68 - 0x6C" << endl;


    cout << "@" << sc_time_stamp() << " Start LDOConfig 0x70 - Ox74 - 0x78 - 0x7C" << endl;
    data = 0;
    data += 0xAEAEAEAE;
    MasterAccess(1, 0x70, data); 
    MasterAccess(0, 0x70, data_read);  
    assert (data == data_read);
    wait(5);

    data= 0;
    data += 0x1C1C1A1A;
    MasterAccess(1, 0x74, data); 
    MasterAccess(0, 0x74, data_read);  
    assert (data == data_read);
    wait(5);

    data= 0;
    //data += 0x5252C1C1;
    data += 0x1C1C1A1A;
    MasterAccess(1, 0x78, data); 
    MasterAccess(0, 0x78, data_read);  
    assert (data == data_read);
    wait(5);

    data= 0;
    data += 0x11FF446F;
    MasterAccess(1, 0x7C, data); 
    MasterAccess(0, 0x7C, data_read);  
    assert (data == data_read);
    wait(5);
    cout << "@" << sc_time_stamp() << " Finish LDOConfig 0x70 - Ox74 - 0x78 - 0x7C" << endl;

    
    cout << "@" << sc_time_stamp() << " Finish config for Dvfs" << endl;

    wait();
    cout << "@" << sc_time_stamp() << " Start Dvfs Computation" << endl;
    data = 0xC;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    while (interrupt.read() == 0) wait();
    cout << "@" << sc_time_stamp() << " End Dvfs Computation" << endl; 
 
    cout << "@" << sc_time_stamp() << " Start reading dvfs status" << endl;
    MasterAccess(0, 0x5C, data_read);
    cout << "@" << sc_time_stamp() << " Finished Reading dvfs status: " << data_read << endl;
  
    /*cout << "@" << sc_time_stamp() << " Start config for ElemAdd" << endl;
    data = 0x800;
    MasterAccess(1, 0x58, data); //reset_mode = 0x800 (6'100000-000000) EADD using PU DecMem1, XXX: flip_mem should be 1
    MasterAccess(0, 0x58, data_read); 
    assert (data == data_read);

    data = 0x04;
    MasterAccess(1, 0x54, data); //gb_mode_config = 4 // EADD gb mode
    MasterAccess(0, 0x54, data_read); 
    assert (data == data_read);

    data = 0x1;
    MasterAccess(1, 0x08, data); //flip_mem = 1;
    MasterAccess(0, 0x08, data_read); 
    assert (data == data_read);  */

    /*data = 0x0;
    MasterAccess(1, 0x4C, data); //use_axi = 0;
    MasterAccess(0, 0x4C, data_read); 
    assert (data == data_read); */ 

    /*data = 0x1;
    MasterAccess(1, 0x50, data); //use_gb = 1
    MasterAccess(0, 0x50, data_read); 
    assert (data == data_read);

    data = 0;
    data += num_vector;
    data += num_timestep << 8; 
    data += adpbias_act1 << 16;
    data += adpbias_act2 << 20;
    data += adpbias_act3 << 24;
    MasterAccess(1, 0x20, data); 
    MasterAccess(0, 0x20, data_read); 
    assert (data == data_read);

    data = 0;
    data += base_input0;
    data += base_input1 << 16;
    MasterAccess(1, 0x14, data); 
    MasterAccess(0, 0x14, data_read); 
    assert (data == data_read);

    data = 0;
    MasterAccess(1, 0x18, data); 
    MasterAccess(0, 0x18, data_read); 
    assert (data == data_read);
    cout << "@" << sc_time_stamp() << " Finish config for ElemAdd" << endl;


    wait(50);
    cout << "@" << sc_time_stamp() << " Start ElemAdd Computation" << endl;
    data = 0xA;
    MasterAccess(1, 0x04, data);
    MasterAccess(0, 0x04, data_read);
    while (interrupt.read() == 0) wait();
    cout << "@" << sc_time_stamp() << " End ElemAdd Computation" << endl; */


    wait(20);
    cout << "@" << sc_time_stamp() << " sc_stop " << endl ;
    sc_stop();
  }
};

SC_MODULE (testbench) {
  sc_clock clk;
  sc_signal<bool> rst;
  
  sc_signal<bool> interrupt;
  sc_signal<NVUINT6> edgebert_dco_cc_sel;
  sc_signal<NVUINT8> edgebert_ldo_res_sel;

  // Testbench master, Control.h Slave 
  typename spec::AxiConf::axi4_conf::read::template chan<> axi_conf_rd;
  typename spec::AxiConf::axi4_conf::write::template chan<> axi_conf_wr;

  // Slave slave, InputAxi WeightAxi Master
  typename spec::AxiData::axi4_data::read::template chan<> axi_data_rd;
  typename spec::AxiData::axi4_data::write::template chan<> axi_data_wr;
  
  //vector<vector<spec::InputType>> W_mat;
  //vector<vector<spec::InputType>> I_mat;
  //vector<vector<spec::InputType>> B_mat;
  //vector<vector<spec::AccumType>> O_mat;
  
  NVHLS_DESIGN(TopAccel) top;
  Slave<spec::AxiData::axiCfg> slave;
  Master master;
  SC_HAS_PROCESS(testbench);
  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
    rst("rst"),
    axi_conf_rd("axi_conf_rd"),
    axi_conf_wr("axi_conf_wr"),
    axi_data_rd("axi_data_rd"),
    axi_data_wr("axi_data_wr"),
    top("top"),
    slave("slave"),
    master("master")
  {
    top.clk(clk);
    top.rst(rst);
    top.if_axi_rd(axi_conf_rd);
    top.if_axi_wr(axi_conf_wr);
    top.if_data_rd(axi_data_rd);
    top.if_data_wr(axi_data_wr);
    top.interrupt(interrupt);
    top.edgebert_dco_cc_sel(edgebert_dco_cc_sel);
    top.edgebert_ldo_res_sel(edgebert_ldo_res_sel);
    
    slave.clk(clk);
    slave.reset_bar(rst);
    slave.if_rd(axi_data_rd);
    slave.if_wr(axi_data_wr);

    master.clk(clk);
    master.rst(rst);
    master.if_rd(axi_conf_rd);
    master.if_wr(axi_conf_wr);
    master.interrupt(interrupt);
    master.edgebert_dco_cc_sel(edgebert_dco_cc_sel);
    master.edgebert_ldo_res_sel(edgebert_ldo_res_sel);

    SC_THREAD(Run);
  }


  void Run() {
    //printf("check check\n");
    wait(2, SC_NS );
    cout << "@" << sc_time_stamp() << " Asserting Reset " << endl;
    rst.write(false);
    wait(2, SC_NS );
    rst.write(true);
    cout << "@" << sc_time_stamp() << " Deasserting Reset " << endl;
    wait(250000,SC_NS);
    cout << "@" << sc_time_stamp() << " sc_stop " << endl ;
    sc_stop();
  }
};

int sc_main(int argc, char *argv[])
{
  //nvhls::set_random_seed();


  // Weight N*N 
  // Input N*M
  // Output N*M
  
  vector<vector<spec::InputType>> W_mat = GetMat<spec::InputType>(N, N); 
  /*vector<vector<spec::InputType>> I_mat = GetMat<spec::InputType>(N, M);  
  vector<vector<spec::InputType>> B_mat = GetMat<spec::InputType>(N, 1);

  vector<vector<spec::AccumType>> O_mat;
  O_mat = MatMul<spec::InputType, spec::AccumType>(W_mat, I_mat); 
 
  // Add bias/mul shift  for each column output
  // 
  for (int j=0; j<M; j++) {    // for each column
    for (int i=0; i<N; i++) {  // for each element in column
      //cout << O_mat[i][j] << "\t";
      spec::AccumType tmp = B_mat[i][0]; 
      O_mat[i][j] = O_mat[i][j] + (tmp << BiasShift);
      //cout << O_mat[i][j] << "\t";
      if (IsRelu && O_mat[i][j] < 0) O_mat[i][j] = 0;
      O_mat[i][j] = O_mat[i][j] * AccumMul;
      
      //cout << O_mat[i][j] << "\t";
      
      //cout << O_mat[i][j] << endl;
      O_mat[i][j] = O_mat[i][j] >> AccumShift;
      if (O_mat[i][j] > 127) O_mat[i][j] = 127;
      if (O_mat[i][j] < -128) O_mat[i][j] = -128;
      //cout << O_mat[i][j] << endl;
    }
    //cout << endl;
  } */

  //cout << "Weight Matrix " << endl; 
  //PrintMat(W_mat);
  //cout << "Input Matrix " << endl; 
  //PrintMat(I_mat);
  //cout << "Reference Output Matrix " << endl; 
  //PrintMat(O_mat);

  testbench my_testbench("my_testbench");

  
  //my_testbench.W_mat = W_mat;
  //my_testbench.I_mat = I_mat;
  //my_testbench.B_mat = B_mat;
  //my_testbench.O_mat = O_mat;
  
  //cout << "Weight Matrix " << endl; 
  //PrintMat(W_mat);
  //cout << "Input Matrix " << endl; 
  //PrintMat(I_mat);
  //cout << "Bias Matrix " << endl; 
  //PrintMat(B_mat);
  //cout << "Reference Output Matrix " << endl; 
  //PrintMat(O_mat);
  
  // write weight to source (make sure data pattern is correct)
  // bias,  row N-1 -> 0 of weight,
  // need to follow this format to store bias/weight into slave memory
  // b00 b10 b20 b30
  // w03 w13 w23 w33
  // w02 w12 w22 w32 
  // w01 w11 w21 w31
  // w00 w10 w20 w30
  // 
  /*int addr = w_rd_base;
  // store bias
  for (int j = 0; j < N; j+=8) {
    NVUINT64 data = 0;
    for (int k = 0; k < 8; k++) {
      data.set_slc<8>(k*8, B_mat[j+k][0]);
    }
    my_testbench.slave.localMem[addr] = data;
    //cout << hex << "slave weight: " << data << endl; 
    addr += 8;
  } */

  // store weight
  int addr = input_rd_base;
  for (int j = N-1; j >= 0; j--) {
    for (int i = 0; i < N; i+=8) {
      NVUINT64 data = 0;
      for (int k = 0; k < 8; k++) {
        data.set_slc<8>(k*8, W_mat[i+k][j]);
      }
      my_testbench.slave.localMem[addr] = data;
      //cout << hex << "slave weight: " << data << endl; 
      addr += 8;
    }   
  }

  // store mask
  NVUINT8 allone = 255;
  addr = mask_rd_base;
  for (int j = 0; j < M; j++) {    // each col
    for (int i = 0; i < N; i+=8) {  // each row
      NVUINT64 data = 0;
      for (int k = 0; k < 8; k++) {
        //data.set_slc<8>(k*8, I_mat[i+k][j]);
        data.set_slc<8>(k*8, allone);
      }
      my_testbench.slave.localMem[addr] = data;
      //cout << addr << "\t" << my_testbench.slave.localMem[addr] << endl;
      addr += 8;
    } 
  }   

  cout << "SC_START" << endl;
  sc_start();

  cout << "CMODEL PASS" << endl;
  return 0;
};


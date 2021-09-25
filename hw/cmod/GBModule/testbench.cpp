/*
 * GBModule testbench
 */

#include "GBModule.h"
#include "../DecodeTop/DecodeTop.h"
#include "../AuxMem/AuxMem.h"

#include "../include/Spec.h"
#include "../include/utils.h"
#include "../include/helper.h"

#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>
#include <vector>

#define NVHLS_VERIFY_BLOCKS (GBModule)
#include <nvhls_verify.h>
using namespace::std;
const int TestNumTiles = 12;

SC_MODULE (Source) {
  sc_in<bool> clk;
  sc_in<bool> rst;

  typedef DecodeTop MaskMemType;
  typedef AuxMem AuxMemType;

  Connections::Out<bool> use_axi;
  //0: LayerNorm, 1: SMax, 2: ElemAdd, 3: Enpy, 4: Dvfs
  Connections::Out<bool> start[5];
  Connections::Out<NVUINT4> mode_config;
  Connections::Out<AuxMemType::aux_req_t>  aux_mem_req[2];
  Connections::Out<spec::VectorType> activation_in;
  Connections::Out<spec::EnpyConfig> enpy_config;
  Connections::Out<spec::InputBufferConfig> input_buffer_config;
  Connections::Out<spec::GBControlConfig> gbcontrol_config;
  Connections::Out<spec::PeriphConfig> periph_config;
  Connections::Out<spec::DvfsConfig> dvfs_config; //dvfs 

  Connections::Out<spec::DCOConfigA> dco_config_a;
  Connections::Out<spec::DCOConfigB> dco_config_b;
  Connections::Out<spec::DCOConfigC> dco_config_c;
  //  
  Connections::Out<spec::LDOConfigA> ldo_config_a;
  Connections::Out<spec::LDOConfigB> ldo_config_b;
  Connections::Out<spec::LDOConfigC> ldo_config_c;
  Connections::Out<spec::LDOConfigD> ldo_config_d;

  Connections::In<spec::VectorType> activation_out;
  Connections::In<MaskMemType::mask_req_t> mask_req_out;
  Connections::In<AuxMemType::aux_rsp_t>  aux_mem_rsp;
  Connections::In<NVUINT2> enpy_status;
  Connections::In<bool> done_out;
  Connections::In<NVUINT6> dco_sel_out;  //dvfs
  Connections::In<NVUINT8> ldo_sel_out;   //dvfs

  spec::VectorType  act_rsp_reg;

  Connections::Combinational<spec::VectorType>  act_out_wr;
  Connections::Combinational<MaskMemType::mask_req_t>  mask_rd_wr;
  //Connections::Combinational<NVUINT4>  mode_config_wr;

  NVUINT8 test_matrix[TestNumTiles][spec::kVectorSize];
  NVUINT8 test_matrix_1[TestNumTiles][spec::kVectorSize];
  NVUINT1 test_mask[TestNumTiles][spec::kVectorSize];
  NVUINTW(spec::kVectorSize) maskdata;

  SC_CTOR(Source) {
    SC_THREAD(InRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(OutRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(ModeRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void InRun() {
 
    use_axi.Reset();
    //mode_config.Reset();
    for (int i=0; i<5; i++) start[i].Reset();
    for (int i=0; i<2; i++) aux_mem_req[i].Reset();
    activation_in.Reset();
    enpy_config.Reset();
    input_buffer_config.Reset();
    gbcontrol_config.Reset();
    periph_config.Reset();
    dvfs_config.Reset();
    dco_config_a.Reset();
    dco_config_b.Reset();
    dco_config_c.Reset();
    ldo_config_a.Reset();
    ldo_config_b.Reset();
    ldo_config_c.Reset();
    ldo_config_d.Reset();

    act_out_wr.ResetRead();
    mask_rd_wr.ResetRead();   
    //mode_config_wr.ResetRead();

    spec::VectorType  act_reg; 
    MaskMemType::mask_req_t mask_reg;
    //NVUINT4 mode_reg;

    wait(10);    
    //mode_reg = mode_config_wr.Pop();
    spec::InputBufferConfig input_buffer_config_reg;
    input_buffer_config_reg.base_input[0] = 0;
    input_buffer_config_reg.base_input[1] = 0;
    input_buffer_config.Push(input_buffer_config_reg);

    wait();
    spec::PeriphConfig periph_config_reg;
    periph_config_reg.base_attn_span = 0;
    periph_config_reg.base_gamma = 4;
    periph_config_reg.base_beta = 8;
    periph_config_reg.adpbias_attn_span = 0;
    periph_config_reg.adpbias_beta = 2;
    periph_config_reg.adpbias_gamma = 2;
    periph_config.Push(periph_config_reg);

    wait();
    spec::GBControlConfig gbcontrol_config_reg;
    gbcontrol_config_reg.num_vector = 1;
    gbcontrol_config_reg.num_timestep = 1;
    gbcontrol_config_reg.adpbias_act1 = -1;
    gbcontrol_config_reg.adpbias_act2 = 2;
    gbcontrol_config_reg.adpbias_act3 = 2;
    gbcontrol_config.Push(gbcontrol_config_reg); 

    wait();
    spec::DCOConfigA dco_config_a_reg;
    dco_config_a_reg.dco_val0 = 0;
    dco_config_a_reg.dco_val1 = 2;
    dco_config_a_reg.dco_val2 = 4;
    dco_config_a_reg.dco_val3 = 6;
    dco_config_a_reg.dco_val4 = 8;
    dco_config_a.Push(dco_config_a_reg);
    cout << "TB - pushed dco_config_a" << endl;
 
    wait();
    spec::DCOConfigB dco_config_b_reg;
    dco_config_b_reg.dco_val5 = 10;
    dco_config_b_reg.dco_val6 = 12;
    dco_config_b_reg.dco_val7 = 14;
    dco_config_b_reg.dco_val8 = 16;
    dco_config_b_reg.dco_val9 = 18;
    dco_config_b.Push(dco_config_b_reg);
    cout << "TB - pushed dco_config_b" << endl;

    wait();
    spec::DCOConfigC dco_config_c_reg;
    dco_config_c_reg.dco_val10 = 20;
    dco_config_c_reg.dco_val11 = 22;
    dco_config_c_reg.dco_val12 = 24;
    dco_config_c_reg.dco_val13 = 26;
    dco_config_c_reg.dco_val14 = 28;
    dco_config_c.Push(dco_config_c_reg);
    cout << "TB - pushed dco_config_c" << endl;

    wait();
    spec::LDOConfigA ldo_config_a_reg;
    ldo_config_a_reg.ldo_val0 = 0;
    ldo_config_a_reg.ldo_val1 = 4;
    ldo_config_a_reg.ldo_val2 = 8;
    ldo_config_a_reg.ldo_val3 = 12;
    ldo_config_a.Push(ldo_config_a_reg);
    cout << "TB - pushed ldo_config_a" << endl;

    wait();
    spec::LDOConfigB ldo_config_b_reg;
    ldo_config_b_reg.ldo_val4 = 16;
    ldo_config_b_reg.ldo_val5 = 20;
    ldo_config_b_reg.ldo_val6 = 24;
    ldo_config_b_reg.ldo_val7 = 28;
    ldo_config_b.Push(ldo_config_b_reg);
    cout << "TB - pushed ldo_config_b" << endl;

    wait();
    spec::LDOConfigC ldo_config_c_reg;
    ldo_config_c_reg.ldo_val8 = 32;
    ldo_config_c_reg.ldo_val9 = 36;
    ldo_config_c_reg.ldo_val10 = 40;
    ldo_config_c_reg.ldo_val11 = 44;
    ldo_config_c.Push(ldo_config_c_reg);
    cout << "TB - pushed ldo_config_c" << endl;
   
    wait();
    spec::LDOConfigD ldo_config_d_reg;
    ldo_config_d_reg.ldo_val12 = 48;
    ldo_config_d_reg.ldo_val13 = 52;
    ldo_config_d_reg.ldo_val14 = 56;
    ldo_config_d_reg.ldo_val15 = 60;
    ldo_config_d.Push(ldo_config_d_reg);
    cout << "TB - pushed ldo_config_d" << endl;

    wait();
    spec::EnpyConfig enpy_config_reg;
    enpy_config_reg.enpy_threshold = 3957;
    enpy_config.Push(enpy_config_reg);

    wait();
    use_axi.Push(0);

    wait(10);
    cout  << sc_time_stamp() << "--------Storing Data via aux_mem_req[0] into Auxiliary Mem--------" << endl;
    for (int i = 0; i < TestNumTiles; i++) {
      AuxMemType::aux_req_t aux_mem_reg;
      aux_mem_reg.type.val = CLITYPE_T::STORE;
      for (int j = 0; j < spec::kVectorSize; j++) {
        aux_mem_reg.data[j] = test_matrix[i][j];
        aux_mem_reg.addr[j] = j + i*(spec::kVectorSize) + 0;
        aux_mem_reg.valids[j] = 1;
        cout << sc_time_stamp() << " Testbench Storing aux_mem_reg.data[i] = " << aux_mem_reg.data[j] << " at address: " << aux_mem_reg.addr[j] << endl;
      }
      aux_mem_req[0].Push(aux_mem_reg);
      wait(10);
    }

    start[3].Push(1);  
    wait(5);
    

    //Timestep 1
    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg= set_bytes<16>("5F_8B_FA_6F_4F_AC_6F_A1_5A_6A_5F_6F_3F_FB_3F_FB");
    activation_in.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg= set_bytes<16>("5F_8B_FA_6F_4F_AC_6F_A1_5A_6A_5F_6F_3F_FB_3F_FB");
    activation_in.Push(act_rsp_reg);
    
    /*mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<32>("00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01_00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    activation_in.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<32>("00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01_00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    activation_in.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<32>("00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01_00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    activation_in.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<32>("00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01_00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    activation_in.Push(act_rsp_reg); */
    
  } 


  void ModeRun() {
     mode_config.Reset();
     //mode_config_wr.ResetWrite();
     //NVUINT4 input = 1;
     wait();
     while(1) {
       mode_config.Push(8);
       //mode_config_wr.Push(1);
       wait();
     }
  } 

  void OutRun() {

    activation_out.Reset();
    mask_req_out.Reset();
    aux_mem_rsp.Reset();
    enpy_status.Reset();
    done_out.Reset(); 
    dco_sel_out.Reset(); 
    ldo_sel_out.Reset(); 

    mask_rd_wr.ResetWrite();
    act_out_wr.ResetWrite();

    wait();

    while(1) {
      spec::VectorType activation_out_reg;
      MaskMemType::mask_req_t mask_reg;
      AuxMemType::aux_rsp_t aux_mem_reg;
      NVUINT2 enpy_status_reg;
      bool done_out_reg;

      if (activation_out.PopNB (activation_out_reg)) {
         //cout << sc_time_stamp() << " Testbench - vec_out data = " << "\t" ;
         //for (int j = 0; j < spec::kVectorSize; j++) {
         //cout << activation_out_reg.data[j] << "\t";
         cout << hex << sc_time_stamp() << " TB ------------------ Popped activation_out_reg data"  << endl;
         //cout << sc_time_stamp() << " Testbench - vec_out data = " << activation_out_reg.data[j] << "\t" << endl;
         //}
         //cout << endl;
         act_out_wr.Push(activation_out_reg);
      }

      if (mask_req_out.PopNB(mask_reg)) {
         cout << hex << sc_time_stamp() << " TB - Popped mask_reg data!" << endl;
         mask_rd_wr.Push(mask_reg);
      }

      if (enpy_status.PopNB(enpy_status_reg)) {
         cout << hex << sc_time_stamp() << " TB - Popped enpy signal = " << enpy_status_reg << endl;
      } 

      if (aux_mem_rsp.PopNB(aux_mem_reg)) {
         cout << hex << sc_time_stamp() << " TB - Popped aux_mem_rsp data = " << endl;
      }

      if (done_out.PopNB(done_out_reg)) {
         cout << hex << sc_time_stamp() << " TB - Popped done signal = " << done_out_reg << endl;
         sc_stop();
      }

      wait();
    }
  }

};

SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);
	sc_clock clk;
  sc_signal<bool> rst;


  typedef DecodeTop MaskMemType;
  typedef AuxMem AuxMemType;

  Connections::Combinational<bool> use_axi;
  Connections::Combinational<bool> start[5];
  Connections::Combinational<NVUINT4> mode_config;
  Connections::Combinational<AuxMemType::aux_req_t>  aux_mem_req[2];
  Connections::Combinational<spec::VectorType> activation_in;
  Connections::Combinational<spec::EnpyConfig> enpy_config;
  Connections::Combinational<spec::InputBufferConfig> input_buffer_config;
  Connections::Combinational<spec::GBControlConfig> gbcontrol_config;
  Connections::Combinational<spec::PeriphConfig> periph_config;
  Connections::Combinational<spec::DvfsConfig> dvfs_config; //dvfs 

  Connections::Combinational<spec::VectorType> activation_out;
  Connections::Combinational<MaskMemType::mask_req_t> mask_req_out;
  Connections::Combinational<AuxMemType::aux_rsp_t>  aux_mem_rsp;
  Connections::Combinational<NVUINT2> enpy_status;
  Connections::Combinational<bool> done_out;
  Connections::Combinational<NVUINT6> dco_sel_out;  //dvfs
  Connections::Combinational<NVUINT8> ldo_sel_out;   //dvfs

  Connections::Combinational<spec::DCOConfigA> dco_config_a;
  Connections::Combinational<spec::DCOConfigB> dco_config_b;
  Connections::Combinational<spec::DCOConfigC> dco_config_c;
  //
  Connections::Combinational<spec::LDOConfigA> ldo_config_a;
  Connections::Combinational<spec::LDOConfigB> ldo_config_b;
  Connections::Combinational<spec::LDOConfigC> ldo_config_c;
  Connections::Combinational<spec::LDOConfigD> ldo_config_d;

  Source src;
  NVHLS_DESIGN(GBModule) dut;

  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
    rst("rst"),
    src("src"),
    dut("dut")
  {
    dut.clk(clk);
    dut.rst(rst);
    dut.use_axi(use_axi);
    for (int i = 0; i < 5; i++) dut.start[i](start[i]);
    dut.mode_config(mode_config);
    for (int i = 0; i < 2; i++) dut.aux_mem_req[i](aux_mem_req[i]);
    dut.activation_in(activation_in);
    dut.enpy_config(enpy_config);
    dut.input_buffer_config(input_buffer_config);
    dut.gbcontrol_config(gbcontrol_config);
    dut.periph_config(periph_config);
    dut.dvfs_config(dvfs_config);
    dut.activation_out(activation_out);
    dut.mask_req_out(mask_req_out);
    dut.aux_mem_rsp(aux_mem_rsp);
    dut.enpy_status(enpy_status);
    dut.done_out(done_out);
    dut.dco_sel_out(dco_sel_out);
    dut.ldo_sel_out(ldo_sel_out);
    dut.dco_config_a(dco_config_a);
    dut.dco_config_b(dco_config_b);
    dut.dco_config_c(dco_config_c);
    dut.ldo_config_a(ldo_config_a);
    dut.ldo_config_b(ldo_config_b);
    dut.ldo_config_c(ldo_config_c);
    dut.ldo_config_d(ldo_config_d);

    src.clk(clk);
    src.rst(rst);
    src.use_axi(use_axi);
    for (int i = 0; i < 5; i++) src.start[i](start[i]);
    src.mode_config(mode_config);
    for (int i = 0; i < 2; i++) src.aux_mem_req[i](aux_mem_req[i]);
    src.activation_in(activation_in);
    src.enpy_config(enpy_config);
    src.input_buffer_config(input_buffer_config);
    src.gbcontrol_config(gbcontrol_config);
    src.periph_config(periph_config);
    src.dvfs_config(dvfs_config);
    src.activation_out(activation_out);
    src.mask_req_out(mask_req_out);
    src.aux_mem_rsp(aux_mem_rsp);
    src.enpy_status(enpy_status);
    src.done_out(done_out);
    src.dco_sel_out(dco_sel_out);
    src.ldo_sel_out(ldo_sel_out);
    src.dco_config_a(dco_config_a);
    src.dco_config_b(dco_config_b);
    src.dco_config_c(dco_config_c);
    src.ldo_config_a(ldo_config_a);
    src.ldo_config_b(ldo_config_b);
    src.ldo_config_c(ldo_config_c);
    src.ldo_config_d(ldo_config_d);

    SC_THREAD(run); 
  }
  
  void run(){
    for (int i = 0; i < TestNumTiles; i++) {
      for (int j = 0; j < spec::kVectorSize; j++) {
        src.test_matrix[i][j] = j + i*(spec::kVectorSize);
        src.test_matrix_1[i][j] = j + i*(spec::kVectorSize) + 1;
        src.test_mask[i][j] = 1;
      }
    }
    /*src.test_mask[0][0] = 1;//15
    src.test_mask[1][0] = 0;//14 
    src.test_mask[2][1] = 0;//13 
    src.test_mask[3][2] = 0;//11
    src.test_mask[4][3] = 0;//7
    src.test_mask[5][0] = 0;
    src.test_mask[5][3] = 0;//6
    src.test_mask[6][1] = 0;
    src.test_mask[6][3] = 0;//5
    src.test_mask[7][3] = 0;
    src.test_mask[7][1] = 0;
    src.test_mask[7][0] = 0;//4 */ 

    
	  wait(2, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" Asserting reset" << std::endl;
    rst.write(false);
    wait(2, SC_NS );
    rst.write(true);
    std::cout << "@" << sc_time_stamp() <<" De-Asserting reset" << std::endl;
    wait(3000, SC_NS);
    std::cout << "@" << sc_time_stamp() <<" sc_stop" << std::endl;
    sc_stop();
  }
};


int sc_main(int argc, char *argv[]) {
  
  cout << "Vector Size = " << spec::kVectorSize << endl;
  testbench tb("tb");
  sc_report_handler::set_actions(SC_ERROR, SC_DISPLAY);
  sc_start();

  bool rc = (sc_report_handler::get_count(SC_ERROR) > 0);
  if (rc)
    DCOUT("TESTBENCH FAIL" << endl);
  else
    DCOUT("TESTBENCH PASS" << endl);
  return rc;
}


 /*
  */

#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>

#include <vector>
#include "Enpy.h"
#include "../include/Spec.h"
#include "../include/utils.h"
#include "../include/helper.h"

#define NVHLS_VERIFY_BLOCKS (Enpy)
#include <nvhls_verify.h>

using namespace::std;

SC_MODULE(Source) {
  sc_in<bool> clk;
  sc_in<bool> rst; 

  typedef DecodeTop MaskMemType;

  Connections::Out<bool> start;
  Connections::Out<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations
  Connections::Out<spec::GBControlConfig> gbcontrol_config;
  Connections::Out<spec::EnpyConfig> enpy_config;
  Connections::Out<spec::VectorType>  act_rsp; //Activation response from DecodeTop

  Connections::In<MaskMemType::mask_req_t>  mask_rd_req; // To DecodeTop for mask read requests
  Connections::In<bool> done;
  Connections::In<NVUINT2> enpy_status;
  Connections::In<spec::ActScalarType> enpy_val_out;

  Connections::Combinational<MaskMemType::mask_req_t>  mask_rd_wr; // To DecodeTop for mask read requests

  bool start_src; 
  
  spec::VectorType  act_rsp_reg;
  spec::EnpyConfig enpy_config_reg;
  spec::GBControlConfig gbcontrol_config_reg;
    
  SC_CTOR(Source) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

    SC_THREAD(runOut);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }


  void runOut(){
    mask_rd_req.Reset();
    done.Reset();
    enpy_status.Reset();
    enpy_val_out.Reset();
    mask_rd_wr.ResetWrite();

    wait();
 
    while (1) {
      bool done_dest;
      MaskMemType::mask_req_t  mask_reg;
      NVUINT2 enpy_status_reg;
      spec::ActScalarType enpy_val_reg;

      if (enpy_val_out.PopNB(enpy_val_reg)) {
        cout << hex << sc_time_stamp() << " TB - Received enpy_val_reg: " << enpy_val_reg << endl;
      }

      if (enpy_status.PopNB(enpy_status_reg)) {
        cout << hex << sc_time_stamp() << " TB - Received enpy status: " << enpy_status_reg << endl;
      }

      if (mask_rd_req.PopNB(mask_reg)) {
        cout << hex << sc_time_stamp() << " TB - Popped mask_reg data" << endl;
        mask_rd_wr.Push(mask_reg);
      }

      if (done.PopNB(done_dest)) {
        cout << hex << sc_time_stamp() << " TB - Done signal issued !!!!" << endl;
        //sc_stop();
      }
      
      wait();    
    }
  }

  void run(){
    input_buffer_config.Reset();
    enpy_config.Reset();
    gbcontrol_config.Reset();
    act_rsp.Reset();
    start.Reset();

    mask_rd_wr.ResetRead();

    MaskMemType::mask_req_t  mask_reg; // To DecodeTop for mask read requests

    //Test1
    wait();
    spec::InputBufferConfig input_buffer_config_reg;
    input_buffer_config_reg.base_input[0] = 0;
    input_buffer_config_reg.base_input[1] = 0;
    input_buffer_config.Push(input_buffer_config_reg);

    wait(20);
    enpy_config_reg.enpy_threshold = 3957;
    enpy_config.Push(enpy_config_reg);
    wait(20);

    gbcontrol_config_reg.num_vector = 1;
    gbcontrol_config_reg.num_timestep = 1;
    gbcontrol_config_reg.adpbias_act1 = -1;
    gbcontrol_config_reg.adpbias_act2 = 1;
    gbcontrol_config_reg.adpbias_act3 = 1;
    gbcontrol_config.Push(gbcontrol_config_reg); 
    wait(20);

    start_src = 1;
    start.Push(start_src);
    wait(4);

    mask_reg = mask_rd_wr.Pop();
    //act_rsp_reg = set_bytes<16>("01_01_01_01_01_01_01_01_01_01_01_01_01_01_01_00");
    //act_rsp_reg = set_bytes<16>("01_00_00_00_04_04_00_CC_00_00_00_00_00_00_AA_F4");
    //act_rsp_reg = set_bytes<16>("00_00_FF_FF_FF_11_11_FF_FF_FF_FF_FF_FF_FF_00_00");
    act_rsp_reg= set_bytes<16>("5F_8B_FA_6F_4F_AC_6F_A1_5A_6A_5F_6F_3F_FB_3F_FB");
    act_rsp.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    //act_rsp_reg = set_bytes<16>("11_00_2A_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    //act_rsp_reg = set_bytes<16>("01_01_01_01_01_01_01_01_01_01_01_01_01_01_01_00");
    //act_rsp_reg = set_bytes<16>("01_00_00_00_04_04_00_CC_00_00_00_00_00_00_AA_F4");
    //act_rsp_reg = set_bytes<16>("00_00_00_00_00_00_00_FF_00_00_00_00_00_00_AA_DD");
    act_rsp_reg= set_bytes<16>("5F_8B_FA_6F_4F_AC_6F_A1_5A_6A_5F_6F_3F_FB_3F_FB");
    //act_rsp_reg = set_bytes<16>("00_00_00_00_FF_FF_00_FF_FF_00_00_00_00_00_00_00");
    //act_rsp_reg = set_bytes<16>("00_00_FF_FF_FF_11_11_FF_FF_FF_FF_FF_FF_FF_00_00");
    act_rsp.Push(act_rsp_reg);

    /*mask_reg = mask_rd_wr.Pop();
    //act_rsp_reg = set_bytes<16>("AF_00_11_02_00_00_00_B0_00_10_00_01_00_00_FF_F1");
    //act_rsp_reg= set_bytes<16>("F0_00_00_00_00_00_00_FF_00_00_00_00_00_00_00_FF");
    act_rsp_reg = set_bytes<16>("2F_01_01_01_01_01_01_01_01_01_01_01_01_01_01_00");
    //act_rsp_reg = set_bytes<16>("FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF");
    act_rsp.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    //act_rsp_reg = set_bytes<16>("F0_00_00_00_00_00_00_FF_00_00_00_00_00_00_00_FF");
    //act_rsp_reg = set_bytes<16>("AF_00_11_02_00_00_00_B0_00_10_00_01_00_00_FF_F1");
    act_rsp_reg = set_bytes<16>("2F_01_01_01_01_01_01_01_01_01_01_01_01_01_01_00");
    //act_rsp_reg = set_bytes<16>("FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF");
    //act_rsp_reg = set_bytes<16>("BB_00_A0_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    act_rsp.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<16>("1F_01_01_01_01_01_01_01_01_01_01_01_01_01_01_00");
    //act_rsp_reg = set_bytes<16>("AF_AF_00_00_00_00_00_00_00_00_00_00_00_00_00_00");
    //act_rsp_reg = set_bytes<16>("FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF");
    //act_rsp_reg = set_bytes<16>("FF_F4_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    act_rsp.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<16>("1F_01_01_01_01_01_01_01_01_01_01_01_01_01_01_00");
    //act_rsp_reg = set_bytes<16>("AF_AF_00_00_00_00_00_00_00_00_00_00_00_00_00_00");
    //act_rsp_reg = set_bytes<16>("FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF_FF");
    //act_rsp_reg = set_bytes<16>("FF_50_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    act_rsp.Push(act_rsp_reg); */
    

    //Test2
    /*wait();
    input_buffer_config_reg.base_input[0] = 0;
    input_buffer_config_reg.base_input[1] = 0;
    input_buffer_config.Push(input_buffer_config_reg);

    wait(20);
    enpy_config_reg.enpy_threshold = 3959;
    enpy_config.Push(enpy_config_reg);
    wait(20);

    gbcontrol_config_reg.num_vector = 3;
    gbcontrol_config_reg.num_timestep = 1;
    gbcontrol_config_reg.adpbias_act1 = 2;
    gbcontrol_config_reg.adpbias_act2 = 2;
    gbcontrol_config.Push(gbcontrol_config_reg); 
    wait(20);

    start_src = 1;
    start.Push(start_src);
    wait(4);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<16>("00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    act_rsp.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<16>("00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    act_rsp.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<16>("00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    act_rsp.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<16>("00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    act_rsp.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<16>("00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    act_rsp.Push(act_rsp_reg);

    mask_reg = mask_rd_wr.Pop();
    act_rsp_reg = set_bytes<16>("00_00_00_02_00_00_00_B0_00_10_00_01_00_00_00_01");
    act_rsp.Push(act_rsp_reg); */

    wait(20);
    sc_stop();
  }
};

SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);
	sc_clock clk;
  sc_signal<bool> rst;

  typedef DecodeTop MaskMemType;
 
  Connections::Combinational<bool> start;
  Connections::Combinational<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations
  Connections::Combinational<spec::EnpyConfig> enpy_config;
  Connections::Combinational<spec::GBControlConfig> gbcontrol_config;
  Connections::Combinational<spec::VectorType>  act_rsp; //Activation response from DecodeTop

  Connections::Combinational<MaskMemType::mask_req_t>  mask_rd_req; // To DecodeTop for mask read requests
  Connections::Combinational<NVUINT2> enpy_status;
  Connections::Combinational<bool> done;
  Connections::Combinational<spec::ActScalarType> enpy_val_out;
 
  NVHLS_DESIGN(Enpy) dut;
  Source  source;
  
  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
    rst("rst"),
    dut("dut"),
    source("source")
  {
    dut.clk(clk);
    dut.rst(rst);
    dut.start(start);
    dut.input_buffer_config(input_buffer_config);
    dut.enpy_config(enpy_config);
    dut.gbcontrol_config(gbcontrol_config);
    dut.act_rsp(act_rsp);
    dut.mask_rd_req(mask_rd_req);
    dut.enpy_status(enpy_status);
    dut.enpy_val_out(enpy_val_out);
    dut.done(done);
    
    source.clk(clk);
    source.rst(rst);
    source.start(start);
    source.input_buffer_config(input_buffer_config);
    source.enpy_config(enpy_config);
    source.gbcontrol_config(gbcontrol_config);
    source.act_rsp(act_rsp);
    source.mask_rd_req(mask_rd_req);
    source.enpy_status(enpy_status);
    source.enpy_val_out(enpy_val_out);
    source.done(done);
	  
    SC_THREAD(run);
  }
  

  void run(){
	  wait(2, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" Asserting reset" << std::endl;
    rst.write(false);
    wait(2, SC_NS );
    rst.write(true);
    std::cout << "@" << sc_time_stamp() <<" De-Asserting reset" << std::endl;
    wait(10000, SC_NS );
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

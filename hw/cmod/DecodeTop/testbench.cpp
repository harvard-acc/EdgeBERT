/*
 * DecodeTop testbench
 */

#include "DecodeTop.h"
#include "../include/Spec.h"
#include "../include/utils.h"

#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>
#include <vector>

#define NVHLS_VERIFY_BLOCKS (DecodeTop)
#include <nvhls_verify.h>
using namespace::std;
const int TestNumTiles = 32;

SC_MODULE (Source) {
  sc_in<bool> clk;
  sc_in<bool> rst;

  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;


  Connections::Out<spec::InputBufferConfig> input_buffer_config;
  Connections::Out<spec::GBControlConfig> gbcontrol_config;
  Connections::Out<spec::IndexType>   base_offset;  //
  Connections::Out<spec::IndexType>   base_input; //
  Connections::Out<bool> flip_mem_input;
  Connections::Out<bool> flip_mem_mask;
  Connections::Out<bool> flip_dec_out;
  Connections::Out<mask_req_t>    mask_mem_req[5];
  Connections::Out<input_req_t>   input_mem_req[3];
  Connections::In<mask_rsp_t>   axi_mask_rsp_out;
  Connections::In<input_rsp_t>  axi_input_rsp_out;
  Connections::In<spec::VectorType>  vec_out; 
  Connections::In<spec::VectorType>  vec_out_to_gb; 
  Connections::Out<NVUINT6> reset_mode;
  Connections::Out<spec::MatrixConfig> mat_config;

  Connections::Combinational<bool> trigger_wr;
  Connections::Combinational<bool> trigger_wr2;

  NVUINT8 test_matrix[TestNumTiles][spec::kVectorSize];
  NVUINT8 test_matrix_1[TestNumTiles][spec::kVectorSize];
  NVUINT1 test_mask[TestNumTiles][spec::kVectorSize];
  NVUINTW(spec::kVectorSize) maskdata;

  SC_CTOR(Source) {
    SC_THREAD(BaseRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(Mask_Input_Run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(OutRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }


  void BaseRun() {
    trigger_wr.ResetRead();
    trigger_wr2.ResetRead();
    reset_mode.Reset();
    base_input.Reset();
    base_offset.Reset();
    mat_config.Reset();
    gbcontrol_config.Reset();
    input_buffer_config.Reset();

    spec::InputBufferConfig input_buffer_config_reg;
    input_buffer_config_reg.base_input[0] = 0;
    input_buffer_config_reg.base_input[1] = 0;

    spec::GBControlConfig gbcontrol_config_reg;
    gbcontrol_config_reg.num_vector = 2;
    gbcontrol_config_reg.num_timestep = 2;
    gbcontrol_config_reg.adpbias_act1 = 2; 
    gbcontrol_config_reg.adpbias_act2 = 2; 
    gbcontrol_config_reg.adpbias_act3 = 2; 

    spec::MatrixConfig mat_config_reg; 
    mat_config_reg.N0 = 32;
    mat_config_reg.N1 = 32;
    mat_config_reg.M = 32;
 
    wait();
    while(1) {
      bool reg;
      if (trigger_wr.PopNB(reg)) {
         reset_mode.Push(16);
      }

      bool reg2;
      if (trigger_wr2.PopNB(reg2)) {
         base_input.Push(0);
         base_offset.Push(0);
         input_buffer_config.Push(input_buffer_config_reg);
         gbcontrol_config.Push(gbcontrol_config_reg);
         mat_config.Push(mat_config_reg);
      }
      
      wait();
    }
  }

  void Mask_Input_Run() {
    trigger_wr.ResetWrite();
    trigger_wr2.ResetWrite();

    NVUINT12 base = 0; 
    cout << "Testbench base: " << base << endl;
    for (int i=0; i<5; i++) mask_mem_req[i].Reset();
    for (int i=0; i<3; i++) input_mem_req[i].Reset();
    // store
    wait(10);
    cout  << sc_time_stamp() << "  ------Storing Mask into Mask Mem--------" << endl;
    for (int i = 0; i < TestNumTiles; i++) {
      spec::mask_req_t mask_reg;
      mask_reg.type.val = CLITYPE_T::STORE;
      for (int j = 0; j < spec::kVectorSize; j++) {
        maskdata.set_slc(j, test_mask[i][j]);
        //cout << "test_mask: " << test_mask[i][j] << endl; 
      }
      mask_reg.data[0] = maskdata;
      mask_reg.addr[0] = i + base;
      mask_reg.valids[0] = 1;
      cout << sc_time_stamp() << " Testbench Storing mask_reg = " << mask_reg.data[0] << " at address: " << mask_reg.addr[0] << endl;
      mask_mem_req[i%5].Push(mask_reg);
      wait(10);
    }


    wait(10);
    cout  << sc_time_stamp() << "--------Storing Data via input_mem_req[0] into Input Mem--------" << endl;
    // store request via input_mem_req[0]
    for (int i = 0; i < TestNumTiles; i++) {
      spec::input_req_t input_reg;
      input_reg.type.val = CLITYPE_T::STORE;
      for (int j = 0; j < spec::kVectorSize; j++) {
        input_reg.data[j] = test_matrix[i][j];
        input_reg.addr[j] = j + i*(spec::kVectorSize) + base;
        input_reg.valids[j] = 1;
        cout << sc_time_stamp() << " Testbench Storing input_reg.data[i] = " << input_reg.data[j] << " at address: " << input_reg.addr[j] << endl;
        //cout << "test_matrix: " << test_matrix[i][j] << endl; 
      }
      input_mem_req[0].Push(input_reg);
      wait(10);
    }

    trigger_wr.Push(1);    
    
    wait(10);

    trigger_wr2.Push(1); 

    // read mask request
    wait(10);
    cout  << sc_time_stamp() << " -----------Load/Read request to Mask Mem------------" << endl;
    for (int i = 0; i < TestNumTiles; i++) {
      spec::mask_req_t mask_reg;
      mask_reg.type.val = CLITYPE_T::LOAD;

      mask_reg.addr[0] = i + base;
      mask_reg.valids[0] = 1;
      cout << sc_time_stamp() << " Testbench Reading mask_reg at address: " << mask_reg.addr[0] << endl;
      mask_mem_req[3].Push(mask_reg);
      wait(10);
    }
    wait();
  } 

  void OutRun() {

    axi_mask_rsp_out.Reset();
    axi_input_rsp_out.Reset();

    vec_out.Reset();
    vec_out_to_gb.Reset();

    flip_mem_input.Reset();
    flip_mem_mask.Reset();
    flip_dec_out.Reset();

    wait();

    while(1) {
      spec::VectorType vec_out_reg;

      flip_mem_mask.Push(0);
      flip_mem_input.Push(0);
      flip_dec_out.Push(0);     
 
      if (vec_out.PopNB(vec_out_reg)) {
         for (int j = 0; j < spec::kVectorSize; j++) {
             cout << sc_time_stamp() << " Testbench - vec_out data = " << vec_out_reg.data[j] << endl;
         }
      }
      else if (vec_out_to_gb.PopNB(vec_out_reg)) {
         for (int j = 0; j < spec::kVectorSize; j++) {
             cout << sc_time_stamp() << " Testbench - vec_out_to_gb data = " << vec_out_reg.data[j] << endl;
         }
      }

      wait();
    }
  }
};

SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);
	sc_clock clk;
  sc_signal<bool> rst;

  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;

  Connections::Combinational<spec::InputBufferConfig> input_buffer_config;
  Connections::Combinational<spec::GBControlConfig> gbcontrol_config;
  Connections::Combinational<spec::IndexType>   base_offset;
  Connections::Combinational<spec::IndexType>   base_input;
  Connections::Combinational<bool> flip_mem_input;
  Connections::Combinational<bool> flip_mem_mask;
  Connections::Combinational<bool> flip_dec_out;
  Connections::Combinational<mask_req_t>    mask_mem_req[5];
  Connections::Combinational<input_req_t>   input_mem_req[3];
  Connections::Combinational<mask_rsp_t>   axi_mask_rsp_out;
  Connections::Combinational<input_rsp_t>  axi_input_rsp_out;
  Connections::Combinational<spec::VectorType>  vec_out; 
  Connections::Combinational<spec::VectorType>  vec_out_to_gb; 
  Connections::Combinational<NVUINT6> reset_mode;
  Connections::Combinational<spec::MatrixConfig> mat_config;

  Source src;
  NVHLS_DESIGN(DecodeTop) dut;

  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 4.0, SC_NS, 2.0, 0, SC_NS, true),
    rst("rst"),
    src("src"),
    dut("dut")
  {
    dut.clk(clk);
    dut.rst(rst);
    dut.base_input(base_input);
    dut.flip_mem_input(flip_mem_input);
    dut.flip_mem_mask(flip_mem_mask);
    dut.flip_dec_out(flip_dec_out);
    for (int i = 0; i < 5; i++) dut.mask_mem_req[i](mask_mem_req[i]);
    for (int i = 0; i < 3; i++) dut.input_mem_req[i](input_mem_req[i]);
    dut.axi_mask_rsp_out(axi_mask_rsp_out);
    dut.axi_input_rsp_out(axi_input_rsp_out);
    dut.vec_out(vec_out);
    dut.vec_out_to_gb(vec_out_to_gb);
    dut.input_buffer_config(input_buffer_config);
    dut.gbcontrol_config(gbcontrol_config);
    dut.base_offset(base_offset);
    dut.reset_mode(reset_mode);
    dut.mat_config(mat_config);

    src.clk(clk);
    src.rst(rst);
    src.base_input(base_input);
    src.flip_mem_input(flip_mem_input);
    src.flip_mem_mask(flip_mem_mask);
    src.flip_dec_out(flip_dec_out);
    for (int i = 0; i < 5; i++) src.mask_mem_req[i](mask_mem_req[i]);
    for (int i = 0; i < 3; i++) src.input_mem_req[i](input_mem_req[i]);
    src.axi_mask_rsp_out(axi_mask_rsp_out);
    src.axi_input_rsp_out(axi_input_rsp_out);
    src.vec_out(vec_out);
    src.vec_out_to_gb(vec_out_to_gb);
    src.input_buffer_config(input_buffer_config);
    src.gbcontrol_config(gbcontrol_config);
    src.base_offset(base_offset);
    src.reset_mode(reset_mode);
    src.mat_config(mat_config);

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
    wait(30000, SC_NS);
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


#include "Decode.h"
#include "../include/Spec.h"

#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>

#include <vector>

#define NVHLS_VERIFY_BLOCKS (Decode)
#include <nvhls_verify.h>
using namespace::std;

#include <testbench/nvhls_rand.h>

const int TestMatrixSize = 128;
const int TestNumTiles = TestMatrixSize*TestMatrixSize/spec::kVectorSize;

// can change from 0-100
const int sparsity = 0;


//#include <../include/SysSpec.h>

// Note that we are decoding a matrix in this test, 
//  but when using out-stationary flow
//  we need to decode a matrix "TestMatrixSize/TestNumTiles". 

SC_MODULE (Source) {
  sc_in <bool> clk;
  sc_in <bool> rst;
  typedef spec::IndexType IndexType;

  Connections::Out<IndexType> base_input; //
  Connections::Out<IndexType> base_offset; //
  Connections::Out<spec::input_rsp_t>  input_rsp; //
  Connections::Out<spec::mask_rsp_t>   mask_rsp; //
  Connections::In<spec::input_req_t> out_req; //
  Connections::In<spec::VectorType>  vec_out; //
  Connections::Out<NVUINT6> reset_mode; //
  Connections::Out<spec::MatrixConfig> mat_config;
  Connections::Out<spec::GBControlConfig> gbcontrol_config; //
  Connections::Out<spec::InputBufferConfig> input_buffer_config;

  // matrix is tiled as follows
  // Outer: left->right and top->down
  // Inner: top->down during tiling, left->right for data access
  // ----   ----
  // ----   ----
  // ----   ----
  // ----   ---- 
  //
  // ----   ----  
  // ----   ----
  // ----   ----  
  // ----   ----

  NVUINT8 test_matrix[TestNumTiles][spec::kVectorSize];
  NVUINT1    test_mask[TestNumTiles][spec::kVectorSize];
  NVUINTW(spec::kVectorSize) maskdata;

  SC_CTOR(Source) {
    SC_THREAD(MaskRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(MemRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(OutRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(BaseRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

    SC_THREAD(ResetModeRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

  }

  void ResetModeRun() {
    reset_mode.Reset();
    while(1) {
       reset_mode.Push(16); //run for enpy
    }
  }

  void BaseRun() {
    base_input.Reset();
    base_offset.Reset();
    gbcontrol_config.Reset();
    input_buffer_config.Reset();
    mat_config.Reset();

    spec::InputBufferConfig input_buffer_config_reg;
    input_buffer_config_reg.base_input[0] = 0;
    input_buffer_config_reg.base_input[1] = 32;
 
    spec::IndexType base_addr = 0;
    spec::IndexType base_offset_reg = 0;

    spec::GBControlConfig gbcontrol_config_reg;
    gbcontrol_config_reg.num_vector = 128;
    gbcontrol_config_reg.num_timestep = 2;
    gbcontrol_config_reg.adpbias_act1 = 2;
    gbcontrol_config_reg.adpbias_act2 = 2;
    gbcontrol_config_reg.adpbias_act3 = 2;

    spec::MatrixConfig mat_config_reg;
    mat_config_reg.N0 = 32;
    mat_config_reg.N1 = 32;
    mat_config_reg.M = 32;

    wait(10);
    while(1) {
      base_input.Push(base_addr); 
      base_offset.Push(base_offset_reg); 
      gbcontrol_config.Push(gbcontrol_config_reg);
      input_buffer_config.Push(input_buffer_config_reg);
      mat_config.Push(mat_config_reg);
      wait();
    } 
  }

  void MaskRun() {
    mask_rsp.Reset();

    wait(20);
    for (int i = 0; i < TestNumTiles; i++) {
      spec::mask_rsp_t mask_reg;
      for (int j = 0; j < spec::kVectorSize; j++) {
        maskdata.set_slc(j, test_mask[i][j]);
        //cout << "tb test_mask: " << test_mask[i][j] << endl; 
      }
      mask_reg.data[0] = maskdata;
      mask_reg.valids[0] = maskdata;
      mask_rsp.Push(mask_reg);
      //cout << "tb pushed mask: " << endl;
      wait();
    }
  }

  void MemRun() {
    out_req.Reset();
    input_rsp.Reset();

    wait(10);

    int req_count = 0; 
    while(1) {
      spec::input_req_t req_reg;
      spec::input_rsp_t rsp_reg;

      req_reg = out_req.Pop();
      assert (req_reg.type.val == CLITYPE_T::LOAD);

      for (int j = 0; j < spec::kVectorSize; j++) {
        // check mask value
        assert (req_reg.valids[j] == test_mask[req_count][j]);
        int bank_addr = req_reg.addr[j] / spec::kVectorSize;
        
        if (req_reg.valids[j] == 1) {
          rsp_reg.data[j] = test_matrix[bank_addr][j];
          rsp_reg.valids[j] = 1;
        }
        else {
          rsp_reg.data[j] = 0;
          rsp_reg.valids[j] = 0;
        }
      }
      input_rsp.Push(rsp_reg);

      req_count += 1;
      wait();
    }
  }

  void OutRun() {
    vec_out.Reset();
    spec::MatrixType mat_out_reg;
    wait(10);
    
    for (int t = 0; t < (TestNumTiles/spec::kVectorSize); t++) {
      for (int i = 0; i < spec::kVectorSize; i++) {
        mat_out_reg[i] = vec_out.Pop();
      }

      std::cout << "@" << sc_time_stamp() << " Pop Output Matrix\t" << t << std::endl;
      for (int i = 0; i < spec::kVectorSize; i++) {
        for (int j = 0; j < spec::kVectorSize; j++) {
          cout << mat_out_reg[i][j] << "\t";
        }
        cout << endl;
      }
      wait();
    }
    std::cout << "@" << sc_time_stamp() << " All output recieved" << std::endl;
    sc_stop();
  }
};



SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);
	sc_clock clk;
  sc_signal<bool> rst;

  Connections::Combinational<spec::input_rsp_t>       input_rsp;
  Connections::Combinational<spec::mask_rsp_t>      mask_rsp;
  Connections::Combinational<spec::input_req_t>        out_req;
  Connections::Combinational<spec::VectorType>   vec_out;
  Connections::Combinational<spec::IndexType>   base_input;
  Connections::Combinational<spec::IndexType> base_offset; //
  Connections::Combinational<NVUINT6> reset_mode; //
  Connections::Combinational<spec::MatrixConfig> mat_config;
  Connections::Combinational<spec::GBControlConfig> gbcontrol_config; //
  Connections::Combinational<spec::InputBufferConfig> input_buffer_config;


  Source src;
  NVHLS_DESIGN(Decode) dut;

  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
    rst("rst"),
    src("src"),
    dut("dut")
  {
    dut.clk(clk);
    dut.rst(rst);
    dut.input_rsp(input_rsp);
    dut.mask_rsp(mask_rsp);
    dut.base_input(base_input);
    dut.out_req(out_req);
    dut.vec_out(vec_out);
    dut.base_offset(base_offset);
    dut.reset_mode(reset_mode);
    dut.mat_config(mat_config);
    dut.gbcontrol_config(gbcontrol_config);
    dut.input_buffer_config(input_buffer_config);

    src.clk(clk);
    src.rst(rst);
    src.input_rsp(input_rsp);
    src.mask_rsp(mask_rsp);
    src.base_input(base_input);
    src.out_req(out_req);
    src.vec_out(vec_out);
    src.base_offset(base_offset);
    src.reset_mode(reset_mode);
    src.mat_config(mat_config);
    src.gbcontrol_config(gbcontrol_config);
    src.input_buffer_config(input_buffer_config);

    SC_THREAD(run); 
  }
  
  void run(){
    // randomly set matrix and mask
    for (int i = 0; i < TestNumTiles; i++) {
      for (int j = 0; j < spec::kVectorSize; j++) {
        src.test_matrix[i][j] = (rand() % 255) + 1;
        int _tmp = (rand() % 100);
        if (_tmp < sparsity) {
          src.test_mask[i][j] = 0;
        }
        else {
          src.test_mask[i][j] = 1;
        }
      }
    }
    
	  wait(2, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" Asserting reset" << std::endl;
    rst.write(false);
    wait(2, SC_NS );
    rst.write(true);
    std::cout << "@" << sc_time_stamp() <<" De-Asserting reset" << std::endl;
    wait(100000, SC_NS);
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



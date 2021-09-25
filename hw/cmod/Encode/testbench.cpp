#include "Encode.h"

#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>

#include <vector>

#define NVHLS_VERIFY_BLOCKS (Encode)
#include <nvhls_verify.h>
using namespace::std;

#include <testbench/nvhls_rand.h>

//const int TestMatrixSize = 4;
const int TestMatrixSize = 128;
const int TestNumTiles = TestMatrixSize*TestMatrixSize/spec::kVectorSize;

// can change from 0-100
const int sparsity = 50;

SC_MODULE (Source) {
  sc_in <bool> clk;
  sc_in <bool> rst;

  Connections::Out<spec::IndexType> base_output;
  Connections::Out<spec::VectorType>  vec_in;  
  Connections::In<spec::input_req_t>        out_req;
  Connections::In<spec::mask_req_t>       out_mask_req;

  NVUINT8 test_matrix[TestNumTiles][spec::kVectorSize];
  NVUINT1 test_mask[TestNumTiles][spec::kVectorSize];
  NVUINTW(spec::kVectorSize) maskdata;

  SC_CTOR(Source) {
    SC_THREAD(InRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(OutRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(BaseRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void BaseRun() {
    base_output.Reset();
    spec::IndexType base_addr = 48;
    
    //wait();
    while(1) {
      base_output.Push(base_addr); 
      wait();
    }
  }

  void InRun() {
    vec_in.Reset();
    spec::VectorType vec_in_reg;

    wait(20);
    for (int i = 0; i < TestNumTiles; i++) {
      for (int j = 0; j < spec::kVectorSize; j++) {
        vec_in_reg[j] = test_matrix[i][j]*test_mask[i][j];
        cout << "test_mask issued: " << test_mask[i][j] << endl; 
      }
      vec_in.Push(vec_in_reg);
      wait();
    }
  }

  void OutRun() {
    out_req.Reset();
    out_mask_req.Reset();

    spec::input_req_t req_reg;
    spec::mask_req_t mask_reg;

    wait(20);
    for (int i = 0; i < TestNumTiles; i++) {
      req_reg = out_req.Pop();
      mask_reg = out_mask_req.Pop();

      if ((i+1) % spec::kVectorSize == 0)
        std::cout << "@" << sc_time_stamp() << " Pop Output Matrix\t" << i << std::endl;

      for (int j = 0; j < spec::kVectorSize; j++) {
        maskdata.set_slc(j, test_mask[i][j]);
        cout << "test_mask received: " << nvhls::get_slc<1>(mask_reg.data[0], j) << endl; 
      }
      
      assert(mask_reg.data[0] == maskdata); 
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

  Connections::Combinational<spec::IndexType> base_output;
  Connections::Combinational<spec::VectorType>  vec_in;  
  Connections::Combinational<spec::input_req_t>        out_req;
  Connections::Combinational<spec::mask_req_t>       out_mask_req;

  Source src;
  NVHLS_DESIGN(Encode) dut;

  testbench(sc_module_name name)
  : sc_module(name),
    //clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
    clk("clk", 2.0, SC_NS, 1.0, 0, SC_NS, true),
    rst("rst"),
    src("src"),
    dut("dut")
  {
    dut.clk(clk);
    dut.rst(rst);
    dut.base_output(base_output);
    dut.vec_in(vec_in);
    dut.out_req(out_req);
    dut.out_mask_req(out_mask_req);

    src.clk(clk);
    src.rst(rst);
    src.base_output(base_output);
    src.vec_in(vec_in);
    src.out_req(out_req);
    src.out_mask_req(out_mask_req);


    SC_THREAD(run); 
  }
  
  void run(){
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
    wait(10000, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" sc_stop" << std::endl;
    sc_stop();
  }
};

int sc_main(int argc, char *argv[]) {
  //sc_trace_file* trace_file_ptr = sc_create_vcd_trace_file("trace");
  //Encode top("top");
  //trace_hierarchy(&top, trace_file_ptr);
  
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



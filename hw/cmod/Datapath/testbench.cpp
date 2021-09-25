
#include "Datapath.h"

#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>

#include <vector>

#define NVHLS_VERIFY_BLOCKS (Datapath)
#include <nvhls_verify.h>
using namespace::std;

#include <testbench/nvhls_rand.h>

#include "../include/Spec.h"
#include "../include/AdpfloatSpec.h"
#include "../include/AdpfloatUtils.h"

const int TestMatrixSize = 128;
const int sparsity = 90;

// Note that we need to modify total number of cycles by "TestMatrixSize/kVectorSize" for 
// overall matrix matrix mul

SC_MODULE (Source) {
  sc_in <bool> clk;
  sc_in <bool> rst;

  Connections::Out<spec::VectorType>   vec_in0; // N
  Connections::Out<spec::VectorType>   vec_in1; // M
  Connections::Out<spec::MatrixConfig>  mat_config; 
  Connections::In<spec::AccumVectorType>  vec_out; // P
  Connections::In<bool>  send_out;

  NVUINT8 test_matrix0[TestMatrixSize][TestMatrixSize];
  bool    test_mask0[TestMatrixSize][TestMatrixSize];

  NVUINT8 test_matrix1[TestMatrixSize][TestMatrixSize];
  bool    test_mask1[TestMatrixSize][TestMatrixSize];

  SC_CTOR(Source) {
    SC_THREAD(InRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

    SC_THREAD(OutRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

    SC_THREAD(SendOut);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void InRun() {
    mat_config.Reset();
    vec_in0.Reset();
    vec_in1.Reset();

    wait(20);
    spec::MatrixConfig mat_config_reg; 
    mat_config_reg.N0 = 128;
    mat_config_reg.N1 = 128;
    mat_config_reg.M = 128;
    mat_config.Push(mat_config_reg);

    //wait(20);
    for (int i = 0; i < TestMatrixSize; i += spec::kVectorSize) {
      for (int j = 0; j < TestMatrixSize; j += spec::kVectorSize) {

        spec::MatrixType mat_reg0;
        spec::MatrixType mat_reg1;
        for (int ii = 0; ii < spec::kVectorSize; ii++) {
          for (int jj = 0; jj < spec::kVectorSize; jj++) {
            mat_reg0[ii][jj] = test_matrix0[i+ii][j+jj]*test_mask0[i+ii][j+jj];
            mat_reg1[ii][jj] = test_matrix1[i+ii][j+jj]*test_mask1[i+ii][j+jj];
          }
        }

        for (int k = 0; k < spec::kVectorSize; k++) {
          vec_in0.Push(mat_reg0[k]);
          vec_in1.Push(mat_reg1[k]);
        }

        wait();
      }
    }
  }

  void SendOut() {
    send_out.Reset();
    wait();
    while (1) {
      bool send_out_reg;
      if (send_out.PopNB(send_out_reg)) {
         cout << "TB -- send_out received" << endl;
      }  
      wait();
    }
  }
  void OutRun() {
    vec_out.Reset();
    wait(20);
    for (int i = 0; i < TestMatrixSize; i += spec::kVectorSize) {
      for (int j = 0; j < TestMatrixSize; j += spec::kVectorSize) {
        spec::AccumMatrixType mat_out_reg;
        for (int k = 0; k < spec::kVectorSize; k++) {
          mat_out_reg[k] = vec_out.Pop();
        }

        std::cout << "@" << sc_time_stamp() << " Pop Output Matrix\t" << i << " " << j << std::endl;
        for (int ii = 0; ii < spec::kVectorSize; ii++) {
          for (int jj = 0; jj < spec::kVectorSize; jj++) {
            cout << mat_out_reg[ii][jj] << "\t";
          }
          cout << endl;
        }        
        wait();
      }
    }
    std::cout << "@" << sc_time_stamp() << " All output recieved" << std::endl;
    sc_stop();
  }
};

SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);
	sc_clock clk;
  sc_signal<bool> rst;

  Connections::Combinational<spec::VectorType>   vec_in0; // N
  Connections::Combinational<spec::VectorType>   vec_in1; // M
  Connections::Combinational<spec::AccumVectorType>  vec_out; // P
  Connections::Combinational<spec::MatrixConfig>  mat_config; 
  Connections::Combinational<bool>  send_out;

  
  Source src;
  NVHLS_DESIGN(Datapath) dut;

  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
    rst("rst"),
    src("src"),
    dut("dut")
  {
    dut.clk(clk);
    dut.rst(rst);
    dut.mat_config(mat_config);
    dut.vec_in0(vec_in0);
    dut.vec_in1(vec_in1);
    dut.vec_out(vec_out);
    dut.send_out(send_out);

    src.clk(clk);
    src.rst(rst);
    src.mat_config(mat_config);
    src.vec_in0(vec_in0);
    src.vec_in1(vec_in1);
    src.vec_out(vec_out);
    src.send_out(send_out);

    SC_THREAD(run); 
  }
  
  void run(){
    // randomly set matrix and mask
    for (int i = 0; i < TestMatrixSize; i++) {
      for (int j = 0; j < TestMatrixSize; j++) {
        src.test_matrix0[i][j] = (rand() % 255) + 1;
        int _tmp0 = (rand() % 100);
        if (_tmp0 < sparsity) {
          src.test_mask0[i][j] = 0;
        }
        else {
          src.test_mask0[i][j] = 1;
        }

        src.test_matrix1[i][j] = (rand() % 255) + 1;
        int _tmp1 = (rand() % 100);
        if (_tmp1 < sparsity) {
          src.test_mask1[i][j] = 0;
        }
        else {
          src.test_mask1[i][j] = 1;
        }
      }
    }
    
	  wait(2, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" Asserting reset" << std::endl;
    rst.write(false);
    wait(2, SC_NS );
    rst.write(true);
    std::cout << "@" << sc_time_stamp() <<" De-Asserting reset" << std::endl;
    wait(1000000, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" sc_stop" << std::endl;
    sc_stop();
  }
};

int sc_main(int argc, char *argv[]) {
  
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



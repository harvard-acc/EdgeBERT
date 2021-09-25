//#include "SysPE.h"
#include "Accum.h"

#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>

#include <vector>

#define NVHLS_VERIFY_BLOCKS (Accum)
#include <nvhls_verify.h>
using namespace::std;

#include <testbench/nvhls_rand.h>

const int TestMatrixSize = 128;
const int sparsity = 50;

SC_MODULE (Source) {
  sc_in <bool> clk;
  sc_in <bool> rst;

  Connections::Out<spec::AccelConfig>   accel_config;
  Connections::Out<spec::AccumVectorType>   vec_in;
  Connections::Out<bool>                    send_out;
  Connections::In<spec::VectorType>         vec_out;

  NVUINT8 test_matrix0[TestMatrixSize][TestMatrixSize];
  bool    test_mask0[TestMatrixSize][TestMatrixSize];
  spec::AccelConfig config_reg;

  SC_CTOR(Source) {
    SC_THREAD(InRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

    SC_THREAD(OutRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void InRun() {
    vec_in.Reset();
    send_out.Reset();
    accel_config.Reset();

    wait();
    config_reg.is_relu = 1; 
    config_reg.is_bias = 1; 
    config_reg.weight_bias = -1343; 
    config_reg.adf_accum_bias = 2; 
    config_reg.accum_right_shift = 2; 
    accel_config.Push(config_reg);   

    wait(20);
    for (int i = 0; i < TestMatrixSize; i += spec::kVectorSize) {
      for (int j = 0; j < TestMatrixSize; j += spec::kVectorSize) {
        spec::AccumMatrixType mat_reg0;
        for (int ii = 0; ii < spec::kVectorSize; ii++) {
          for (int jj = 0; jj < spec::kVectorSize; jj++) {
            mat_reg0[ii][jj] = test_matrix0[i+ii][j+jj]*test_mask0[i+ii][j+jj];
          }
        }

        for (int k = 0; k < spec::kVectorSize; k++) {
          vec_in.Push(mat_reg0[k]);
        }
        wait();
      }
      wait();
      wait();
      send_out.Push(1);
      wait();
    }
  }

  void OutRun() {
    vec_out.Reset();

    wait(20);
    for (int i = 0; i < TestMatrixSize; i += spec::kVectorSize) {
      spec::MatrixType mat_out_reg;
      for (int k = 0; k < spec::kVectorSize; k++) {
        mat_out_reg[k] = vec_out.Pop();
      }

      std::cout << "@" << sc_time_stamp() << " Pop Output Matrix\t" << i << std::endl;
      for (int ii = 0; ii < spec::kVectorSize; ii++) {
        for (int jj = 0; jj < spec::kVectorSize; jj++) {
          cout << mat_out_reg[ii][jj] << "\t";
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
  Connections::Combinational<spec::AccelConfig>   accel_config;
  Connections::Combinational<spec::AccumVectorType>   vec_in;
  Connections::Combinational<bool>                    send_out;
  Connections::Combinational<spec::VectorType>        vec_out;

  Source src;
  NVHLS_DESIGN(Accum) dut;

  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
    rst("rst"),
    src("src"),
    dut("dut")
  {
    dut.clk(clk);
    dut.rst(rst);
    dut.accel_config(accel_config);
    dut.vec_in(vec_in);
    dut.send_out(send_out);
    dut.vec_out(vec_out);

    src.clk(clk);
    src.rst(rst);
    src.accel_config(accel_config);
    src.vec_in(vec_in);
    src.send_out(send_out);
    src.vec_out(vec_out);

    SC_THREAD(run); 
  }
  
  void run(){
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
  cout << "Vector Size = " << spec::kVectorSize << endl;
  cout << "Sparsity = " << sparsity << endl;

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



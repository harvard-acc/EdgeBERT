/*
 * DecMem testbench
 */

#include "DecMem.h"
#include "../DecMemCore/DecMemCore.h"
#include "../include/Spec.h"
#include "../include/utils.h"

#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>
#include <vector>

#define NVHLS_VERIFY_BLOCKS (DecMem)
#include <nvhls_verify.h>
using namespace::std;
const int TestNumTiles = 8;

SC_MODULE (Source) {
  sc_in<bool> clk;
  sc_in<bool> rst;

  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;

  Connections::Out<bool> flip_mem_input;
  Connections::Out<bool> flip_mem_mask;
  Connections::Out<mask_req_t>    mask_mem_req[5];
  Connections::Out<input_req_t>   input_mem_req[4];
  Connections::In<mask_rsp_t>   mask_mem_rsp[2];
  Connections::In<input_rsp_t>  input_mem_rsp[2];  

  NVUINT8 test_matrix[TestNumTiles][spec::kVectorSize];
  NVUINT8 test_matrix_1[TestNumTiles][spec::kVectorSize];
  NVUINT1 test_mask[TestNumTiles][spec::kVectorSize];
  NVUINTW(spec::kVectorSize) maskdata;

  SC_CTOR(Source) {
    SC_THREAD(MaskRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(InputRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(OutRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void MaskRun() {
    for (int i=0; i<5; i++) mask_mem_req[i].Reset();
    // store
    wait(10);
    cout  << sc_time_stamp() << "--------Storing into Mask Mem--------" << endl;
    for (int i = 0; i < TestNumTiles; i++) {
      spec::mask_req_t mask_reg;
      mask_reg.type.val = CLITYPE_T::STORE;
      for (int j = 0; j < spec::kVectorSize; j++) {
        maskdata.set_slc(j, test_mask[i][j]);
        //cout << "test_mask: " << test_mask[i][j] << endl; 
      }
      mask_reg.data[0] = maskdata;
      mask_reg.addr[0] = i;
      mask_reg.valids[0] = 1;
      cout << sc_time_stamp() << " Testbench Storing mask_reg = " << mask_reg.data[0] << " at address: " << mask_reg.addr[0] << endl;
      mask_mem_req[i%5].Push(mask_reg);
      wait(10);
    }

    // load
    wait(10);
    cout  << sc_time_stamp() << " -----------Loading from Mask Mem------------" << endl;
    for (int i = 0; i < TestNumTiles; i++) {
      spec::mask_req_t mask_reg;
      mask_reg.type.val = CLITYPE_T::LOAD;

      mask_reg.addr[0] = i;
      mask_reg.valids[0] = 1;
      cout << sc_time_stamp() << " Testbench Reading mask_reg at address: " << mask_reg.addr[0] << endl;
      mask_mem_req[i%5].Push(mask_reg);
      wait(10);
    }
    wait();
  } 

  void InputRun() {
    for (int i=0; i<4; i++) input_mem_req[i].Reset();

    wait(10);
    cout  << sc_time_stamp() << "--------Storing via input_mem_req[0]--------" << endl;
    // store request via input_mem_req[0]
    for (int i = 0; i < TestNumTiles; i++) {
      spec::input_req_t input_reg;
      input_reg.type.val = CLITYPE_T::STORE;
      for (int j = 0; j < spec::kVectorSize; j++) {
        input_reg.data[j] = test_matrix[i][j];
        input_reg.addr[j] = j + i*(spec::kVectorSize);
        input_reg.valids[j] = 1;
        cout << sc_time_stamp() << " Testbench Storing input_reg.data[i] = " << input_reg.data[j] << " at address: " << input_reg.addr[j] << endl;
        //cout << "test_matrix: " << test_matrix[i][j] << endl; 
      }
      input_mem_req[0].Push(input_reg);
      wait(10);
    }
      
    wait(10);
    cout  << sc_time_stamp() << "--------Reading via input_mem_req[1]--------" << endl;
    // read request via input_mem_req[1]
    for (int i = 0; i < TestNumTiles; i++) {
      spec::input_req_t input_reg;
      input_reg.type.val = CLITYPE_T::LOAD;
      for (int j = 0; j < spec::kVectorSize; j++) {
        input_reg.addr[j] = j + i*(spec::kVectorSize);
        input_reg.valids[j] = 1;
        cout << sc_time_stamp() << " Testbench Reading input_reg at address: " << input_reg.addr[j] << endl;
      }
      input_mem_req[1].Push(input_reg);
      wait();
    }

    wait(10);
    cout  << sc_time_stamp() << "--------Storing via input_mem_req[2]--------" << endl;
    // store request via input_mem_req[2]
    for (int i = 0; i < TestNumTiles; i++) {
      spec::input_req_t input_reg;
      input_reg.type.val = CLITYPE_T::STORE;
      for (int j = 0; j < spec::kVectorSize; j++) {
        input_reg.data[j] = test_matrix_1[i][j];
        input_reg.addr[j] = j + i*(spec::kVectorSize);
        input_reg.valids[j] = 1;
        cout << sc_time_stamp() << " Testbench Storing input_reg.data[i] = " << input_reg.data[j] << " at address: " << input_reg.addr[j] << endl;
        //cout << "test_matrix_1: " << test_matrix_1[i][j] << endl; 
      }
      input_mem_req[2].Push(input_reg);
      wait();
    } 

    wait(10);
    cout  << sc_time_stamp() << "--------Reading via input_mem_req[3]--------" << endl;
    // read request via input_mem_req[3]
    for (int i = 0; i < TestNumTiles; i++) {
      spec::input_req_t input_reg;
      input_reg.type.val = CLITYPE_T::LOAD;
      for (int j = 0; j < spec::kVectorSize; j++) {
        input_reg.addr[j] = j + i*(spec::kVectorSize);
        input_reg.valids[j] = 1;
        cout << sc_time_stamp() << " Testbench Reading input_reg at address: " << input_reg.addr[j] << endl;
      }
      input_mem_req[3].Push(input_reg);
      wait();
    }
    wait();
  } 

  void OutRun() {
    flip_mem_input.Reset();
    flip_mem_mask.Reset();
    for (int i=0; i<2; i++) input_mem_rsp[i].Reset();
    for (int i=0; i<2; i++) mask_mem_rsp[i].Reset();

    wait();

    while(1) {
      input_rsp_t input_rsp_reg[2];
      mask_rsp_t  mask_rsp_reg[2];

      flip_mem_mask.Push(1);
      flip_mem_input.Push(0);
      
      if (input_mem_rsp[0].PopNB(input_rsp_reg[0])) {
         for (int j = 0; j < spec::kVectorSize; j++) {
             cout << sc_time_stamp() << " Testbench - input_mem_rsp[0] data = " << input_rsp_reg[0].data[j] << endl;
         }
      }

      if (input_mem_rsp[1].PopNB(input_rsp_reg[1])) {
         for (int j = 0; j < spec::kVectorSize; j++) {
             cout << sc_time_stamp() << " Testbench - input_mem_rsp[1] data = " << input_rsp_reg[1].data[j] << endl;
         }
      }

      if (mask_mem_rsp[0].PopNB(mask_rsp_reg[0])) {
          cout << sc_time_stamp() << " Testbench - mask_mem_rsp[0] data = " << mask_rsp_reg[0].data[0] << endl;
      }

      if (mask_mem_rsp[1].PopNB(mask_rsp_reg[1])) {
          cout << sc_time_stamp() << " Testbench - mask_mem_rsp[1] data = " << mask_rsp_reg[1].data[0] << endl;
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

  Connections::Combinational<bool> flip_mem_input;
  Connections::Combinational<bool> flip_mem_mask;
  Connections::Combinational<mask_req_t>    mask_mem_req[5];
  Connections::Combinational<input_req_t>   input_mem_req[4];
  Connections::Combinational<mask_rsp_t>   mask_mem_rsp[2];
  Connections::Combinational<input_rsp_t>  input_mem_rsp[2]; 

  Source src;
  NVHLS_DESIGN(DecMem) dut;

  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
    rst("rst"),
    src("src"),
    dut("dut")
  {
    dut.clk(clk);
    dut.rst(rst);
    dut.flip_mem_input(flip_mem_input);
    dut.flip_mem_mask(flip_mem_mask);
    for (int i = 0; i < 5; i++) dut.mask_mem_req[i](mask_mem_req[i]);
    for (int i = 0; i < 4; i++) dut.input_mem_req[i](input_mem_req[i]);
    for (int i = 0; i < 2; i++) dut.mask_mem_rsp[i](mask_mem_rsp[i]);
    for (int i = 0; i < 2; i++) dut.input_mem_rsp[i](input_mem_rsp[i]);

    src.clk(clk);
    src.rst(rst);
    src.flip_mem_input(flip_mem_input);
    src.flip_mem_mask(flip_mem_mask);
    for (int i = 0; i < 5; i++) src.mask_mem_req[i](mask_mem_req[i]);
    for (int i = 0; i < 4; i++) src.input_mem_req[i](input_mem_req[i]);
    for (int i = 0; i < 2; i++) src.mask_mem_rsp[i](mask_mem_rsp[i]);
    for (int i = 0; i < 2; i++) src.input_mem_rsp[i](input_mem_rsp[i]);

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
    src.test_mask[0][0] = 1;//15
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
    src.test_mask[7][0] = 0;//4 

    
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


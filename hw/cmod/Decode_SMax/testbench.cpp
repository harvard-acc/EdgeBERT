/*
 */

#include "Decode_SMax.h"
#include "../include/Spec.h"
#include "../include/utils.h"

#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>
#include <vector>

#define NVHLS_VERIFY_BLOCKS (Decode_SMax)
#include <nvhls_verify.h>
using namespace::std;

SC_MODULE (testbench) {
  sc_clock clk;
  sc_signal<bool> rst;

  SC_HAS_PROCESS(testbench);
  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 1, SC_NS, 0.5,0,SC_NS,true),
    rst("rst")
  {
    SC_THREAD(Run);
  }

  void Run() {
    rst = 1;
    wait(10.5, SC_NS);
    rst = 0;
    cout << "@" << sc_time_stamp() << " Asserting Reset " << endl ;
    wait(1, SC_NS);
    cout << "@" << sc_time_stamp() << " Deasserting Reset " << endl ;
    rst = 1;
    
    wait(1000, SC_NS);
    cout << "@" << sc_time_stamp() << " Stop " << endl ;
    sc_stop();
  }
};



int sc_main(int argc, char *argv[])
{
  nvhls::set_random_seed();
  testbench my_testbench("my_testbench");


  sc_start();
  cout << "CMODEL PASS" << endl;
  return 0;
};


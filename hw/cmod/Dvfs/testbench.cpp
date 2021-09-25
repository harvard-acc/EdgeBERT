
#include "Dvfs.h"
#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>

#include <vector>

//#include "../include/Spec.h"
//#include "../include/AxiSpec.h"
//#include "../include/AdpfloatSpec.h"
//#include "../include/AdpfloatUtils.h"

#include "../include/helper.h"

//#include <iostream>
//#include <sstream>
//#include <iomanip>


#define NVHLS_VERIFY_BLOCKS (Dvfs)
#include <nvhls_verify.h>
using namespace::std;

SC_MODULE(Source) {
  sc_in<bool> clk;
  sc_in<bool> rst; 

  Connections::Out<bool> start;
  Connections::Out<spec::DvfsConfig> dvfs_config;
  Connections::Out<spec::ActScalarType> enpy_val_in;

  Connections::Out<spec::DCOConfigA> dco_config_a;
  Connections::Out<spec::DCOConfigB> dco_config_b;
  Connections::Out<spec::DCOConfigC> dco_config_c;
  //
  Connections::Out<spec::LDOConfigA> ldo_config_a;
  Connections::Out<spec::LDOConfigB> ldo_config_b;
  Connections::Out<spec::LDOConfigC> ldo_config_c;
  Connections::Out<spec::LDOConfigD> ldo_config_d;

  Connections::In<NVUINT6> dco_sel_out;
  Connections::In<NVUINT8> ldo_sel_out;
  Connections::In<bool> done;



  bool start_src; 
  
    
  SC_CTOR(Source) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

    SC_THREAD(runOut);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }


  void runOut(){
    dco_sel_out.Reset();
    ldo_sel_out.Reset();
    done.Reset();

    wait();
 
    while (1) {
      NVUINT6 dco_sel_out_reg;
      NVUINT8 ldo_sel_out_reg;
      bool done_dest;

      if (dco_sel_out.PopNB(dco_sel_out_reg)) {
        cout << hex << sc_time_stamp() << " TB - Popped dco_sel_out_reg data" << dco_sel_out_reg << endl;
      }

      if (ldo_sel_out.PopNB(ldo_sel_out_reg)) {
        cout << hex << sc_time_stamp() << " TB - Popped ldo_sel_out_reg data" << ldo_sel_out_reg << endl;
      }

      if (done.PopNB(done_dest)) {
        cout << hex << sc_time_stamp() << " Done signal issued !!!!" << endl;
        sc_stop();
      }
      
      wait();    
    }
  }

  void run(){

    dvfs_config.Reset();
    enpy_val_in.Reset();
    start.Reset();
    dco_config_a.Reset();
    dco_config_b.Reset();
    dco_config_c.Reset();
    ldo_config_a.Reset();
    ldo_config_b.Reset();
    ldo_config_c.Reset();
    ldo_config_d.Reset();

    wait();
    enpy_val_in.Push(10240);
    cout << "TB - pushed enpy_val_in" << endl;

    wait();
    spec::DvfsConfig dvfs_config_reg;
    dvfs_config_reg.enpy_scale = 1;
    dvfs_config.Push(dvfs_config_reg);
    cout << "TB - pushed dvfs_config" << endl;

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
    start_src = 1;
    start.Push(start_src);
    cout << "TB - pushed start signal!!!" << endl;
    //wait(4);

  }
  
};

SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);
	sc_clock clk;
  sc_signal<bool> rst;

  Connections::Combinational<bool> start;
  Connections::Combinational<spec::DvfsConfig> dvfs_config;
  Connections::Combinational<spec::ActScalarType> enpy_val_in;

  Connections::Combinational<spec::DCOConfigA> dco_config_a;
  Connections::Combinational<spec::DCOConfigB> dco_config_b;
  Connections::Combinational<spec::DCOConfigC> dco_config_c;
  //
  Connections::Combinational<spec::LDOConfigA> ldo_config_a;
  Connections::Combinational<spec::LDOConfigB> ldo_config_b;
  Connections::Combinational<spec::LDOConfigC> ldo_config_c;
  Connections::Combinational<spec::LDOConfigD> ldo_config_d;

  Connections::Combinational<NVUINT6> dco_sel_out;
  Connections::Combinational<NVUINT8> ldo_sel_out;
  Connections::Combinational<bool> done;
 
  NVHLS_DESIGN(Dvfs) dut;
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
    dut.dvfs_config(dvfs_config);
    dut.enpy_val_in(enpy_val_in);
    dut.dco_config_a(dco_config_a);
    dut.dco_config_b(dco_config_b);
    dut.dco_config_c(dco_config_c);
    dut.ldo_config_a(ldo_config_a);
    dut.ldo_config_b(ldo_config_b);
    dut.ldo_config_c(ldo_config_c);
    dut.ldo_config_d(ldo_config_d);
    dut.dco_sel_out(dco_sel_out);
    dut.ldo_sel_out(ldo_sel_out);
    dut.done(done);
    
    source.clk(clk);
    source.rst(rst);
    source.start(start);
    source.dvfs_config(dvfs_config);
    source.enpy_val_in(enpy_val_in);
    source.dco_config_a(dco_config_a);
    source.dco_config_b(dco_config_b);
    source.dco_config_c(dco_config_c);
    source.ldo_config_a(ldo_config_a);
    source.ldo_config_b(ldo_config_b);
    source.ldo_config_c(ldo_config_c);
    source.ldo_config_d(ldo_config_d);
    source.dco_sel_out(dco_sel_out);
    source.ldo_sel_out(ldo_sel_out);
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



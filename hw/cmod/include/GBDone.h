#ifndef __GBDONE_H__
#define __GBDONE_H__
 
#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>

class GBDone : public match::Module { 
  static const int kDebugLevel = 3;
  SC_HAS_PROCESS(GBDone);
 public: 
  Connections::Out<bool> done; 
  Connections::In<bool> dvfs_done;
  Connections::In<bool> enpy_done;
  Connections::In<bool> smax_done;
  Connections::In<bool> layernorm_done;
  Connections::In<bool> elemadd_done;   
  
   // Constructor
  GBDone (sc_module_name nm)
      : match::Module(nm),
        dvfs_done("dvfs_done"),
        enpy_done("enpy_done"),
        smax_done("smax_done"), 
        layernorm_done("layernorm_done"),
        elemadd_done("elemadd_done")
  {
    SC_THREAD(GBDoneRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }    
  
  void GBDoneRun() {
    done.Reset();
    enpy_done.Reset();
    dvfs_done.Reset();
    smax_done.Reset();
    layernorm_done.Reset();
    elemadd_done.Reset();

    #pragma hls_pipeline_init_interval 1
    while(1) {
      bool is_done = 0, done_reg = 0;
      if (enpy_done.PopNB(done_reg)) {
        is_done = 1;
      }
      else if (dvfs_done.PopNB(done_reg)) {
        is_done = 1;
      }
      else if (smax_done.PopNB(done_reg)) {
        is_done = 1;
      }
      else if (layernorm_done.PopNB(done_reg)) {
        is_done = 1;
      }
      else if (elemadd_done.PopNB(done_reg)) {
        is_done = 1;
      }
      if (is_done == 1){
        done.Push(1);       
      }
      wait();
    }
  }
};

#endif

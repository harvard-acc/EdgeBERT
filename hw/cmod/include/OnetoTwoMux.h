
#ifndef __ONETOTWOMUX_H__
#define __ONETOTWOMUX_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_connections.h>
#include <nvhls_vector.h>

#include "../include/Spec.h"

template <typename DataType>
class OnetoTwoMux : public match::Module {
 public: 
  //sc_in_clk     clk;
  //sc_in<bool>   rst;

  Connections::In<bool> flip;
  Connections::In<DataType>   in;
  Connections::Out<DataType>  out[2];

  SC_HAS_PROCESS(OnetoTwoMux);
  OnetoTwoMux(sc_module_name name_) 
      : match::Module(name_) { 

    SC_THREAD (MuxRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);
  }

  void MuxRun() {
    flip.Reset();
    in.Reset();
    #pragma hls_unroll yes
    for (int i = 0; i < 2; i++) out[i].Reset();
    bool mem_id = 0;

    #pragma hls_pipeline_init_interval 1
    while(1) {
      DataType reg;
      
      if (in.PopNB(reg)) {
        if (mem_id == 0)  out[0].Push(reg);
        else              out[1].Push(reg);
      }

      bool tmp;
      if (flip.PopNB(tmp)) {
        mem_id = tmp;
      }       
      wait();
    }
  }

};

#endif

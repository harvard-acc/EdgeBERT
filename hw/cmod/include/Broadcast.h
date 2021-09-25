/*
 * Broadcast.h
 */

#ifndef __BROADCAST_H__
#define __BROADCAST_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_module.h>
#include <nvhls_connections.h>
//#include "../include/Spec.h"

template <typename DataType, unsigned int NumOutputs> 
class Broadcast : public match::Module {
 public:
  //static const int kDebugLevel = 0;
  //sc_in_clk    clk;
  //sc_in<bool>  rst; 
  Connections::In<NVUINT4> mode_config;
  Connections::In<DataType>     data_in;
  Connections::Out<DataType>    data_out[NumOutputs];

  SC_HAS_PROCESS(Broadcast);
  Broadcast(sc_module_name name_)
      : match::Module(name_) {
    
    SC_THREAD(Run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    
    // Disable Trace
    this->SetTraceLevel(0);
  }
  
  void Run() { 
    data_in.Reset();
    mode_config.Reset();
    #pragma hls_unroll yes    
    for (unsigned int i = 0; i < NumOutputs; i++) {
      data_out[i].Reset();
    }

    NVUINT4 valids = 0;
    bool mode_config_ready = 0;

    #pragma hls_pipeline_init_interval 1
    while(1) {
      NVUINT4 mode_reg;

      if (mode_config.PopNB(mode_reg)) {
        #pragma hls_unroll yes    
        for (unsigned int i = 0; i < NumOutputs; i++) {
          valids[i] = mode_reg[i];
        }
        //CDCOUT(sc_time_stamp()  << name() << " DUT - Broadcast mode_config: " << mode_reg << " and valids: " << valids << endl, 0);
        mode_config_ready = 1;
      } 

      DataType data_in_reg;
      if (data_in.PopNB(data_in_reg) && mode_config_ready) {
        //CDCOUT(sc_time_stamp()  << name() << " DUT - Broadcast check1 -" << endl, kDebugLevel);
        #pragma hls_unroll yes    
        for (unsigned int i = 0; i < NumOutputs; i++) {
          //data_out[i].Push(data_in_reg);
          if (valids[i] != 0) data_out[i].Push(data_in_reg);
        }
        //CDCOUT(sc_time_stamp()  << name() << " DUT - Broadcast valids: " << valids << endl, 0);
      } //if
      wait();
    } //while
  }  
};

#endif

/*
 * Xbar.h
 */

#ifndef __XBAR_H__
#define __XBAR_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>
#include <arbitrated_crossbar.h>

#include "../include/Spec.h"

template <typename DataType, unsigned int NumInputs, unsigned int NumOutputs,
          unsigned int LenInputBuffer, unsigned int LenOutputBuffer>
class Xbar : public match::Module {
  //static const int NumInputs       = 4;
  //static const int NumOutputs      = 1;
  //static const int LenInputBuffer  = 4; 
  //static const int LenOutputBuffer = 1;
  //typedef spec::input_req_t DataType;
  typedef NVUINTW(nvhls::index_width<NumOutputs>::val) OutIdxType;

  typedef DataType     DataInArray[NumInputs];
  typedef OutIdxType   OutIdxArray[NumInputs];
  typedef bool         ValidInArray[NumInputs];
  typedef DataType     DataOutArray[NumOutputs];
  typedef bool         ValidOutArray[NumOutputs];
  typedef bool         ReadyArray[NumInputs];

 public:
  Connections::In<DataType>     data_in[NumInputs];
  Connections::Out<DataType>    data_out[NumOutputs];

  ArbitratedCrossbar<DataType, NumInputs, NumOutputs, LenInputBuffer, LenOutputBuffer> arbxbar;

  SC_HAS_PROCESS(Xbar);
  Xbar(sc_module_name name_)
      : match::Module(name_) {
    
    SC_THREAD(Run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    
    // Disable Trace
    this->SetTraceLevel(0);
  }
  
  void Run() { 
    #pragma hls_unroll yes
    for(unsigned int inp_lane=0; inp_lane<NumInputs; inp_lane++) {
      data_in[inp_lane].Reset();
    }

    #pragma hls_unroll yes
    for(unsigned int out_lane=0; out_lane<NumOutputs; out_lane++) {
      data_out[out_lane].Reset();
    }

    #pragma hls_pipeline_init_interval 1
    while(1) {
      wait();
      T(1) << "##### Entered DUT #####" << EndT;

      DataInArray     data_in_reg;
      OutIdxArray     dest_in_reg;
      ValidInArray    valid_in_reg;
      #pragma hls_unroll yes
      for(unsigned int inp_lane=0; inp_lane<NumInputs; inp_lane++) {
	      dest_in_reg[inp_lane]  = 0;
        if(!arbxbar.isInputFull(inp_lane) && LenInputBuffer > 0) {
	        valid_in_reg[inp_lane] = data_in[inp_lane].PopNB(data_in_reg[inp_lane]);
        } else {
          valid_in_reg[inp_lane] = false;
        }
          T(2) << "data_in["   << inp_lane << "] = " << data_in_reg[inp_lane]
               << " dest_in["  << inp_lane << "] = " << dest_in_reg[inp_lane]
               << " valid_in[" << inp_lane << "] = " << valid_in_reg[inp_lane] << EndT;
      }

      DataOutArray  data_out_reg;
      ValidOutArray valid_out_reg;
      ReadyArray    ready_reg;
      arbxbar.run(data_in_reg, dest_in_reg, valid_in_reg, data_out_reg, valid_out_reg, ready_reg);

      if(LenOutputBuffer > 0) {
        arbxbar.pop_all_lanes(valid_out_reg);
      }

      // Push only the valid outputs.
      #pragma hls_unroll yes
      for(unsigned int out_lane=0; out_lane<NumOutputs; out_lane++) {
        if(valid_out_reg[out_lane]) {
          data_out[out_lane].Push(data_out_reg[out_lane]);
          T(2) << "data_out[" << out_lane << "] = " << data_out_reg[out_lane] << EndT;
        }
      }
    }
  }  
};

#endif

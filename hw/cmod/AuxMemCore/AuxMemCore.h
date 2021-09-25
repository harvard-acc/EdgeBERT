/*
 * AuxMemCore.h
 */

#ifndef __AUXMEMCORE_H__
#define __AUXMEMCORE_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>

#include <ArbitratedScratchpad.h>
#include <ArbitratedScratchpad/ArbitratedScratchpadTypes.h>

#include "../include/Spec.h"

SC_MODULE(AuxMemCore)
{
 public: 
  sc_in_clk     clk;
  sc_in<bool>   rst;

  // Memory Type Definitions
  typedef spec::AuxMemType AuxMemType;
  typedef spec::aux_req_t aux_req_t;
  typedef spec::aux_rsp_t aux_rsp_t;
  static const int N = spec::N;

  // Memory instantiations
  AuxMemType aux_mem_inst;
  
  Connections::In<aux_req_t> aux_req;
  Connections::Out<aux_rsp_t> aux_rsp;

  SC_HAS_PROCESS(AuxMemCore);
  AuxMemCore(sc_module_name name_) : sc_module(name_) {
    SC_THREAD (AuxMemRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 
  }

  void AuxMemRun() {
    aux_req.Reset();
    aux_rsp.Reset();
 
    #pragma hls_pipeline_init_interval 1
    while(1) {
      aux_req_t req_reg;
      aux_rsp_t rsp_reg;
      bool aux_ready[N];
      bool is_read = 0;    

      if (aux_req.PopNB(req_reg)) {
        is_read = (req_reg.type.val == CLITYPE_T::LOAD);
        aux_mem_inst.load_store(req_reg, rsp_reg, aux_ready);
        
        if (is_read) {
          aux_rsp.Push(rsp_reg);
        }
      }

      wait();
    }
  } 

};

#endif


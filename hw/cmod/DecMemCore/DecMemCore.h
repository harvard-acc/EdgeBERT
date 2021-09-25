/*
 * DecMemCore.h
 */

#ifndef __DECMEMCORE_H__
#define __DECMEMCORE_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>

#include <ArbitratedScratchpad.h>
#include <ArbitratedScratchpad/ArbitratedScratchpadTypes.h>

#include <string>
#include <one_hot_to_bin.h>

#include "../include/Spec.h"

SC_MODULE(DecMemCore)
{
 public: 
  sc_in_clk     clk;
  sc_in<bool>   rst;

  // Memory Type Definitions
  typedef spec::InputMemType InputMemType;
  typedef spec::MaskMemType MaskMemType;
  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;
  static const int N = spec::N;
  static const int kDebugLevel = 0;

  // Memory instantiations
  InputMemType input_mem_inst;
  MaskMemType mask_mem_inst;
  
  Connections::In<input_req_t> input_req_inter;
  Connections::Out<input_rsp_t> input_rsp_inter;
  Connections::In<mask_req_t> mask_req_inter;
  Connections::Out<mask_rsp_t> mask_rsp_inter;

  SC_HAS_PROCESS(DecMemCore);
  DecMemCore(sc_module_name name_) : sc_module(name_) {

    SC_THREAD (DecInputMemRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 

    SC_THREAD (DecMaskMemRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 

  }

  void DecInputMemRun() {
    input_req_inter.Reset();
    input_rsp_inter.Reset();
 
    #pragma hls_pipeline_init_interval 1
    while(1) {
      input_req_t req_reg;
      input_rsp_t rsp_reg;
      bool input_ready[N];
      bool is_read = 0;    

      if (input_req_inter.PopNB(req_reg)) {
        is_read = (req_reg.type.val == CLITYPE_T::LOAD);
        /*if (is_read) {
           for (int i = 0; i<spec::kVectorSize; i++) { 
               CDCOUT(sc_time_stamp() << name() << " - DUT: DecMemCore - InputMem Request at address: " << req_reg.addr[i] << " with data: " << req_reg.data[i] << " is_read: " << is_read << endl, 0);
           }
        } */
        input_mem_inst.load_store(req_reg, rsp_reg, input_ready);
        if (is_read) {
          input_rsp_inter.Push(rsp_reg);
          //CDCOUT(sc_time_stamp()  << name() << " - DUT: DecMemCore - InputMem Request" << endl, kDebugLevel); 
        }
      }

      wait();
    }
  } 

  void DecMaskMemRun() {
    mask_req_inter.Reset();
    mask_rsp_inter.Reset();
    
    #pragma hls_pipeline_init_interval 1
    while(1) {
      
      mask_req_t mask_req_reg;
      mask_rsp_t mask_rsp_reg;
      bool mask_ready[1];
      bool is_mask_read = 0;      
      if (mask_req_inter.PopNB(mask_req_reg)) {
        is_mask_read = (mask_req_reg.type.val == CLITYPE_T::LOAD);
        /*if (is_mask_read) {
           CDCOUT(sc_time_stamp() << name() << " - DUT: DecMemCore - MaskMem Request at address: " << mask_req_reg.addr[0] << " with data: " << mask_req_reg.data[0] << " is_mask_read: " << is_mask_read << endl, 0); 
        } */
        mask_mem_inst.load_store(mask_req_reg, mask_rsp_reg, mask_ready);
        
        if (is_mask_read) {
          mask_rsp_inter.Push(mask_rsp_reg);
          //CDCOUT(sc_time_stamp()  << name() << " - DUT: DecMemCore - MaskMem Read Request" << endl, kDebugLevel); 
          //cout << sc_time_stamp() << " Pushed mask response data from Mask buffer: " << mask_rsp_reg.data[0] << endl;
        }
      }
      wait();
    }
  } 

};

#endif


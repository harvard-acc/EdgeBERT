/* 
 * this module fetches data from scratchpad and decode it with bit mask coding
 * we assume the data is already written in scratchpad. This module only simulates
 *
 * the memory fetching and decoding and push decoded matrix to output
 *  
 * For each kVectorSize*kVectorSize matrix, we decode it with a parallel of 
 * kVectorSize
 *  
 * For expriments, we hardcode matrix size to 128*128 (16*16 blocks)
 *  
 * FIXME: Currently, we use rfs for storing memory 
 */

#ifndef __ENCODE_H__
#define __ENCODE_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>
//#include <ac_std_float.h>
#include "../include/Spec.h"
//#include "../include/ArbitratedScratchpad.h"

// TODO may need to add input channels (fixed config pattern is OK)
SC_MODULE(Encode)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef spec::IndexType IndexType;
  IndexType base_addr;
  static const int kDebugLevel = 0;

  Connections::In<IndexType> base_output;
  Connections::In<spec::VectorType>  vec_in; 
  Connections::Out<spec::input_req_t> out_req;
  Connections::Out<spec::mask_req_t> out_mask_req;

  NVUINT16 bank_ctrs[spec::kVectorSize];
  NVUINTW(spec::kVectorSize) maskdata;
  NVUINT1 one;
  NVUINT1 zero;

  SC_HAS_PROCESS(Encode);
  Encode(sc_module_name name_) : sc_module(name_) {
    SC_THREAD (Run);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);
  }

  void Run() {
    base_output.Reset();
    vec_in.Reset();
    out_req.Reset();
    out_mask_req.Reset();
    
    #pragma hls_unroll yes    
    for (int i = 0; i < spec::kVectorSize; i++) {
      bank_ctrs[i] = 0;
    }
    NVUINT16 mask_ctrs = 0;
    zero = 0;
    one = 1;
    maskdata = 0;
    base_addr = 0;
    bool valid = 0;
    //wait();
    #pragma hls_pipeline_init_interval 1
    while(1) {
      IndexType base_output_reg;
      if (base_output.PopNB(base_output_reg)) {
         base_addr = base_output_reg;
         valid = 1;
      }
      if (valid == 1) {
        spec::VectorType vec_in_reg;
        spec::input_req_t req_reg;
        spec::mask_req_t mask_reg;
   
        vec_in_reg = vec_in.Pop();
        //CDCOUT(sc_time_stamp()  << name() << " - DUT: Encode - Received Input with base_addr: " << base_addr << endl, kDebugLevel);
        req_reg.type.val = CLITYPE_T::STORE;
        mask_reg.type.val = CLITYPE_T::STORE;

        #pragma hls_unroll yes
        for (int i = 0; i < spec::kVectorSize; i++) {
          //req_reg.addr[i] = base_output_reg + spec::kVectorSize*bank_ctrs[i] + i;
          req_reg.addr[i] = base_addr + spec::kVectorSize*bank_ctrs[i] + i;
          req_reg.data[i] = vec_in_reg[i];
          if (vec_in_reg[i] != 0) {
            req_reg.valids[i] = 1;
            maskdata.set_slc(i, nvhls::get_slc<1>(one, 0));
            bank_ctrs[i] += 1;
          } else {
            req_reg.valids[i] = 0;
            maskdata.set_slc(i, nvhls::get_slc<1>(zero, 0));
          }
          //CDCOUT(sc_time_stamp()  << name() << " - DUT: Encode - req_reg data addr: " << req_reg.addr[i] << " and req_reg data valid: " << req_reg.valids[i] << " with maskdata: " << maskdata << endl, 0);
        }
        mask_reg.data[0] = maskdata;
        //CDCOUT(sc_time_stamp()  << name() << " - DUT: Encode - maskctrs is: " << mask_ctrs << endl, kDebugLevel);
        //mask_reg.addr[0] = base_output_reg + mask_ctrs;
        mask_reg.addr[0] = base_addr + mask_ctrs;
        mask_reg.valids[0] = 1;
        out_mask_req.Push(mask_reg);
        out_req.Push(req_reg);

        mask_ctrs += 1;
      } // if valid == 1
      wait();
    } // while
  }
};


#endif

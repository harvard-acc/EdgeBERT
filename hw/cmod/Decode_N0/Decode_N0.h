
#ifndef __DECODE_N0_H__
#define __DECODE_N0_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>
#include "../include/Spec.h"

SC_MODULE(Decode_N0)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  static const int N = spec::N;   
  static const int log2v = nvhls::log2_floor<N>::val; 
  
  typedef spec::IndexType IndexType;
  IndexType base_addr;
  IndexType offset;
  NVUINT10  N0;
  NVUINT10  N1;
  NVUINT12  M;
  NVUINT6  _N0;
  NVUINT6  _N1;
  NVUINT24 M_DecN0;
  NVUINT18 C_DecN0;

  NVUINT12 bank_ctrs[N];
  NVUINT12 record_ctrs[N];

  bool base_input_ready, base_offset_ready, basic_ready, mat_config_ready;

  static const int kDebugLevel = 0;

  Connections::In<IndexType> base_input;
  Connections::In<IndexType> base_offset;
  Connections::In<spec::mask_rsp_t>   mask_rsp;
  Connections::Out<spec::input_req_t> out_req;
  Connections::In<spec::MatrixConfig> mat_config;

  SC_HAS_PROCESS(Decode_N0);
  Decode_N0(sc_module_name name_) : sc_module(name_) {

    SC_THREAD (RunInputAddrGen);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

  }
  
  void ResetVars() {
    base_input_ready = 0;
    base_offset_ready = 0;
    mat_config_ready = 0;
    basic_ready = 0;
  }

  void ResetCtrs() {
    #pragma hls_unroll yes    
    for (int i = 0; i < N; i++) {
      bank_ctrs[i] = 0;
      record_ctrs[i] = 0;
    }   
  }

  void RunInputAddrGen() {
    mask_rsp.Reset();
    mat_config.Reset();
    base_input.Reset();
    base_offset.Reset();
    out_req.Reset();
   
    base_addr = 0;
    offset = 0;
    NVUINT12 count = 0;

    ResetCtrs();
    ResetVars();

    N0 = 1;
    N1 = 1;
    M = 1;
    _N0 = 1 ;
    _N1 = 1;

    wait();

    #pragma hls_pipeline_init_interval 2
    while (1) { 

      IndexType  base_input_reg;
      if (base_input.PopNB(base_input_reg)) {
          base_addr = base_input_reg;
          base_input_ready = 1;
      } 

      IndexType offset_reg;
      if (base_offset.PopNB(offset_reg)) {
         offset = offset_reg;
         base_offset_ready = 1;
      }

      spec::MatrixConfig matrix_config_tmp;
      if (mat_config.PopNB(matrix_config_tmp)) {
         N0 = matrix_config_tmp.N0;
         N1 = matrix_config_tmp.N1;
         M =  matrix_config_tmp.M;
         _N0 = N0 >> log2v;
         _N1 = N1 >> log2v; 
         M_DecN0 = _N0*_N1*M;
         C_DecN0 = _N0*M - 1; 
         mat_config_ready = 1;
      }

      if (base_input_ready && base_offset_ready && mat_config_ready) {
         basic_ready =1; 
      }

      if (basic_ready)  { 
        CDCOUT(sc_time_stamp()  << name() << " - DUT: Decode_N0: N0 mode" << endl, 0);

        ResetCtrs();

        count = 0;
        #pragma hls_pipeline_init_interval 1
        for (int i = 0; i < M_DecN0; i++) {
    
          spec::mask_rsp_t in_mask_reg = mask_rsp.Pop();

          spec::input_req_t req_reg;
          req_reg.type.val = CLITYPE_T::LOAD;

          #pragma hls_unroll yes    
          for (int i = 0; i < N; i++) {
            req_reg.addr[i] = N*bank_ctrs[i] + i + base_addr - offset;
            //cout << sc_time_stamp() << name() <<  " - DUT: Decode_N0/InputAddrGen0 - input req address from decoder: " << req_reg.addr[i] << endl;
            if (in_mask_reg.data[0][i] == 1) {
              bank_ctrs[i] += 1;
              req_reg.valids[i] = 1;
            } else {
              req_reg.valids[i] = 0;
            }
          }
          out_req.Push(req_reg);

          if (count == C_DecN0) {
             count = 0;
             #pragma hls_unroll yes 
             for (int i = 0; i < N; i++) {
                 bank_ctrs[i] = record_ctrs[i];
             }
          } 
          else count += 1;

          wait();
        } // for loop sequential    
        ResetVars();
      } // if basic_read
      wait(); 
    } // while(1)

  }

};


#endif

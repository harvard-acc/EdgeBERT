/*
 * AuxMem.h
 */

#ifndef __AUXMEM_H__
#define __AUXMEM_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>

#include "../include/Spec.h"
#include "../include/Xbar.h"
#include "../AuxMemCore/AuxMemCore.h"

SC_MODULE(AuxMem)
{
 public: 
  sc_in_clk     clk;
  sc_in<bool>   rst;

  static const int NumInputs       = 4;
  static const int NumOutputs      = 1;
  static const int LenInputBuffer  = 4; 
  static const int LenOutputBuffer = 1; 

  typedef spec::aux_req_t aux_req_t;
  typedef spec::aux_rsp_t aux_rsp_t;

  Connections::In<bool> use_axi;
 
  Connections::In<aux_req_t>   aux_mem_req[4];
  Connections::Out<aux_rsp_t>  aux_mem_rsp;
  Connections::Out<spec::VectorType> aux_vec_out;

  Connections::Combinational<aux_req_t> aux_req_inter;
  Connections::Combinational<aux_rsp_t> aux_rsp_inter;

  Xbar<aux_req_t, NumInputs, NumOutputs, LenInputBuffer, LenOutputBuffer> arbxbar_aux_inst;
  AuxMemCore mem_inst;

  SC_HAS_PROCESS(AuxMem);
  AuxMem(sc_module_name name_) : 
    arbxbar_aux_inst("arbxbar_aux_inst"),
    mem_inst("mem_inst")
  {

    arbxbar_aux_inst.clk(clk); 
    arbxbar_aux_inst.rst(rst);
    for (int i = 0; i < NumInputs; i++) {     
      arbxbar_aux_inst.data_in[i](aux_mem_req[i]);
    }   
    arbxbar_aux_inst.data_out[0](aux_req_inter);

    mem_inst.clk(clk);
    mem_inst.rst(rst);
    mem_inst.aux_req(aux_req_inter);
    mem_inst.aux_rsp(aux_rsp_inter); 

    SC_THREAD (MergeRspRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst); 

  }

  void MergeRspRun() {
    use_axi.Reset();
    aux_rsp_inter.ResetRead();
    aux_vec_out.Reset();
    aux_mem_rsp.Reset();
    
    bool mem_id = 0;
    #pragma hls_pipeline_init_interval 1
    while(1) {
      aux_rsp_t rsp_reg;
      spec::VectorType vec_out_reg;     
 
      if (aux_rsp_inter.PopNB(rsp_reg)) {
        if (mem_id == 0) {
           #pragma hls_unroll yes
           for (int j = 0; j < spec::kVectorSize; j++) {
             if (rsp_reg.valids[j] == 1) vec_out_reg[j] = rsp_reg.data[j];
             else vec_out_reg[j] = 0;
           }
           aux_vec_out.Push(vec_out_reg);
        }
        else aux_mem_rsp.Push(rsp_reg);
      }

      bool tmp;
      if (use_axi.PopNB(tmp)) {
        //cout << sc_time_stamp() << " use_axi from AuxMem: " << tmp << endl;
        mem_id = tmp;
      }       
      wait();
    }
  }

};

#endif


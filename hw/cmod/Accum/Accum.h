
#ifndef __ACCUM_H__
#define __ACCUM_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>
//#include <ac_std_float.h>
#include <ArbitratedScratchpad.h>
#include "../include/Spec.h"
#include "../include/AdpfloatSpec.h"
#include "../include/AdpfloatUtils.h"

// TODO may need to add input channels (fixed config pattern is OK)
SC_MODULE(Accum)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;
  static const int kDebugLevel = 0;

  bool    is_relu;
  bool    is_bias;
  NVINT8 weight_bias;
  spec::AdpfloatBiasType adf_accum_bias;
  NVUINT5 accum_right_shift;

  Connections::In<spec::AccelConfig>   accel_config;
  Connections::In<spec::AccumVectorType>  vec_in;  // accume data in adpfloat format
  Connections::In<bool> send_out;
  Connections::Out<spec::VectorType>  vec_out; // randomly set array size 

  SC_HAS_PROCESS(Accum);
  Accum(sc_module_name name_) : sc_module(name_) {
    SC_THREAD (Run);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);
  }
  
  spec::AccumMatrixType accum_mat;

  void Run() {
    accel_config.Reset();
    vec_in.Reset();
    send_out.Reset();
    vec_out.Reset();

    NVUINT5 in_ctr = 0;

    is_relu = 0;
    is_bias = 0;
    weight_bias = 0;
    adf_accum_bias = 0;
    accum_right_shift = 0;

    #pragma hls_pipeline_init_interval 1
    while(1) {
      spec::AccelConfig accel_config_tmp;
      spec::AccumVectorType vec_in_reg;
      spec::AccumVectorType accum_vector_out;
      bool send_out_reg;
      spec::VectorType vec_out_reg;

      if (accel_config.PopNB(accel_config_tmp)) {
        is_relu = accel_config_tmp.is_relu;
        is_bias = accel_config_tmp.is_bias;
        weight_bias = accel_config_tmp.weight_bias;
        adf_accum_bias = accel_config_tmp.adf_accum_bias;
        accum_right_shift = accel_config_tmp.accum_right_shift;
      }

      if (vec_in.PopNB(vec_in_reg)) {
        //CDCOUT(sc_time_stamp()  << name() << " - DUT: Accum - in_ctr is: " << in_ctr << endl, 0);
        CDCOUT(sc_time_stamp()  << name() << " - DUT: Accum - Received Datapath Output wth in_ctr: " << in_ctr << endl, 0);
        if (vec_in_reg.to_rawbits() != 0) {
          #pragma hls_unroll yes
          for (int i = 0; i < spec::kVectorSize; i++) {
            accum_mat[in_ctr][i] += vec_in_reg[i];
          }
        }
        if (in_ctr == spec::kVectorSize-1) {
          in_ctr = 0;
        } else {
          in_ctr += 1;
        }
      }
     
      if (send_out.PopNB(send_out_reg)) {
         if (send_out_reg != 0) {
             CDCOUT(sc_time_stamp()  << name() << " - DUT: Accum - Received send_out from Datapath with is_relu: " << is_relu << endl, 0);
             #pragma hls_pipeline_init_interval 1
             for (int i = 0; i < spec::kVectorSize; i++) {  
                 #pragma hls_unroll yes
                 for (int j = 0; j < spec::kVectorSize; j++) {   
                     // AdapativFloat right shift
                     accum_vector_out[j] = accum_mat[i][j] >> accum_right_shift;

                     //bias addition
                     if (is_bias==1) accum_vector_out[j] += weight_bias;

                     //Relu
                     if ((is_relu==1) && (accum_vector_out[j] < 0)) accum_vector_out[j] = 0;

                     //Truncation
                     if (accum_vector_out[j] > spec::kActWordMax)
                        accum_vector_out[j] = spec::kActWordMax;
                     else if (accum_vector_out[j] < spec::kActWordMin) 
                        accum_vector_out[j] = spec::kActWordMin;

                     // quantize back to AdaptivFloat (8 cycles)
                     AdpfloatType<8,3> _tmp;
                     _tmp.set_value_fixed<16, 12>(accum_vector_out[j], adf_accum_bias);
                     vec_out_reg[j] = _tmp.to_rawbits();
                     accum_mat[i][j] = 0;
                 }  
                 vec_out.Push(vec_out_reg);
                 CDCOUT(sc_time_stamp()  << name() << " - DUT: Accum - pushing vectors to Encoder: " << endl, 0);
                 wait();    
             }
         } // if send_out != 0
      } // if send_out
      wait();
    } // while
  }  // Run
};


#endif

/* 
 * An AdaptivFloat datapath design with kVectorSize*kVectorSize MACs to 
 * compute Matrix-Matrix Mul of size kVectorSize with kVectorSize 
 * Cycles
 *  
 * this module only computes Matrix-Matrix and will push result to an accumulator
 *  
 * The accumulator is missing
 */


#ifndef __DATAPATH_TOP_H__
#define __DATAPATH_TOP_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <nvhls_connections.h>
#include <ac_std_float.h>

#include "../include/Spec.h"
#include "../include/AdpfloatSpec.h"
#include "../include/AdpfloatUtils.h"


//const int kTileSizeI = 1;
//const int kTileSizeK = 1;
//const int kTileSizeJ = 1;


SC_MODULE(Datapath_Top)
{
  public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  static const int N = spec::N;   
  static const int log2v = nvhls::log2_floor<N>::val; 
  static const int kDebugLevel = 0;

  Connections::In<spec::MatrixConfig> mat_config;
  Connections::In<spec::VectorType>   vec_in0 ; // N
  Connections::In<spec::VectorType>   vec_in1; // M
  Connections::Out<spec::AccumVectorType>  vec_out; // P
  Connections::Out<bool> send_out;

  SC_HAS_PROCESS(Datapath_Top);
  Datapath_Top(sc_module_name name_) : sc_module(name_) {
    SC_THREAD (Run);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);
  }

  void Run() {
    mat_config.Reset();
    vec_in0.Reset();
    vec_in1.Reset();
    vec_out.Reset();
    send_out.Reset();
    NVUINT8 count = 0;
    NVUINT8   _M;
    //bool to_send = 0;
    bool valid = 0;
    wait(); 
    #pragma hls_pipeline_init_interval 1
    while(1) {
      
      spec::MatrixConfig matrix_config_tmp;
      if (mat_config.PopNB(matrix_config_tmp)) {
         NVUINT12  M =  matrix_config_tmp.M;
         _M = M >> log2v;
         valid = 1;
         //wait();
         CDCOUT(sc_time_stamp()  << name() << " - DUT: Datapath_Top - _M is: " << _M << endl, 0);
      } 
      while (valid) {
        // recieve two matrice
        spec::MatrixType mat_in0_reg;
        spec::MatrixType mat_in1_reg;
        #pragma hls_pipeline_init_interval 1
        for (int i = 0; i < spec::kVectorSize; i++) {
          mat_in0_reg[i] = vec_in0.Pop();
          mat_in1_reg[i] = vec_in1.Pop();
          //CDCOUT(sc_time_stamp()  << name() << " - DUT: Datapath_Top - Received Vector Input" << endl, kDebugLevel);
          wait();
        }

        CDCOUT(sc_time_stamp()  << name() << " DUT: Datapath_Top - count check is: " << count << " with _M: " << _M << endl, 0);

        spec::VectorType dp_in0[spec::kVectorSize];
        #pragma hls_unroll yes
        for (int i = 0; i < spec::kVectorSize; i++) {
          dp_in0[i] = mat_in0_reg[i];
        }

        spec::AccumVectorType dp_out;
        #pragma hls_pipeline_init_interval 1
        for (int i = 0; i < spec::kVectorSize; i++) {
            VecMul(dp_in0, mat_in1_reg[i], dp_out); 
            vec_out.Push(dp_out);
            wait();
        }
        
        if (count == _M - 1) {
           //CDCOUT(sc_time_stamp()  << name() << " DUT: Datapath_Top - count check is: " << count << " with _M: " << _M << endl, 0);
           cout << sc_time_stamp()  << name() << "count check2: " << count << " with _M: " << _M << endl;
           //CDCOUT(sc_time_stamp()  << name() << " - DUT: Datapath_Top - Issued Send_out to Accum" << endl, 0);
           count = 0;
           //to_send = 1;
           send_out.Push(1); 
        } else {
          count += 1;
          //to_send = 0;
          send_out.Push(0); 
        }
        //send_out.Push(to_send); 
        //CDCOUT(sc_time_stamp()  << name() << " - DUT: Datapath_Top - to_send: " << to_send << endl, 0);
        wait();
      }
      wait();
    } // while 
  }
  
  // HLS, kVectorSize 16, passed II = 1 
  // HLS, kVectorSize 32, failed II = 1 
  inline void ProductSum(const spec::VectorType in_1, const spec::VectorType in_2, spec::AccumScalarType& out) {
    spec::AccumScalarType out_tmp = 0; 
    
    #pragma hls_unroll yes
    #pragma cluster addtree 
    #pragma cluster_type both  
    for (int j = 0; j < spec::kVectorSize; j++) {
      AdpfloatType<8,3> in_1_adpfloat(in_1[j]);
      AdpfloatType<8,3> in_2_adpfloat(in_2[j]);
      spec::AccumScalarType acc_tmp;
      adpfloat_mul(in_1_adpfloat, in_2_adpfloat, acc_tmp);
      out_tmp += acc_tmp;
    }
    out = out_tmp;
  }


  inline void VecMul(spec::VectorType weight_in[spec::kVectorSize], spec::VectorType input_in, spec::AccumVectorType& accum_out)
  {
    spec::AccumVectorType accum_out_tmp;            
    #pragma hls_unroll yes 
    for (int i = 0; i < spec::kVectorSize; i++) {

      accum_out_tmp[i] = 0;
      NVUINTW(spec::kVectorSize) mask0;
      NVUINTW(spec::kVectorSize) mask1;
      #pragma hls_unroll yes    
      for (int j = 0; j < spec::kVectorSize; j++) { 
        mask0[j] = (input_in[j] != 0);
        mask1[j] = (weight_in[i][j] != 0);
      }
      
      if (mask1 && mask0) {
        ProductSum(weight_in[i], input_in, accum_out_tmp[i]);
      }
    }
    
    accum_out = accum_out_tmp;
  }
  
};

#endif

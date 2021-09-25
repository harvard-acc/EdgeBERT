/* 
 * An AdaptivFloat datapath design with kVectorSize*kVectorSize MACs to 
 * compute Matrix-Matrix Mul of size kVectorSize with kVectorSize 
 * Cycles
 *  
 * this module only computes Matrix-Matrix and will push result to an accumulator
 *  
 * The accumulator is missing
 */


#ifndef __DATAPATH_H__
#define __DATAPATH_H__

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


SC_MODULE(Datapath)
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

  SC_HAS_PROCESS(Datapath);
  Datapath(sc_module_name name_) : sc_module(name_) {
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
    #pragma hls_pipeline_init_interval 2
    while(1) {
      
      spec::MatrixConfig matrix_config_tmp;
      if (mat_config.PopNB(matrix_config_tmp)) {
         NVUINT12  M =  matrix_config_tmp.M;
         _M = M >> log2v;
         valid = 1;
         //wait();
         CDCOUT(sc_time_stamp()  << name() << " - DUT: Datapath - _M is: " << _M << endl, 0);
      } 
      while (valid) {
        // recieve two matrice
        spec::MatrixType mat_in0_reg;
        spec::MatrixType mat_in1_reg;
        #pragma hls_pipeline_init_interval 1
        for (int i = 0; i < spec::kVectorSize; i++) {
          mat_in0_reg[i] = vec_in0.Pop();
          mat_in1_reg[i] = vec_in1.Pop();
          //CDCOUT(sc_time_stamp()  << name() << " - DUT: Datapath - Received Vector Input" << endl, kDebugLevel);
          wait();
        }

        CDCOUT(sc_time_stamp()  << name() << " DUT: Datapath - count check is: " << count << " with _M: " << _M << endl, 0);

        spec::AccumMatrixType mat_out_reg = MatMul(mat_in0_reg, mat_in1_reg);

        #pragma hls_pipeline_init_interval 1
        for (int i = 0; i < spec::kVectorSize; i++) {
          vec_out.Push(mat_out_reg[i]);
          wait();      
        }
        if (count == _M - 1) {
           //CDCOUT(sc_time_stamp()  << name() << " DUT: Datapath - count check is: " << count << " with _M: " << _M << endl, 0);
           cout << sc_time_stamp()  << name() << "count check2: " << count << " with _M: " << _M << endl;
           //CDCOUT(sc_time_stamp()  << name() << " - DUT: Datapath - Issued Send_out to Accum" << endl, 0);
           count = 0;
           //to_send = 1;
           send_out.Push(1); 
        } else {
          count += 1;
          //to_send = 0;
          send_out.Push(0); 
        }
        //send_out.Push(to_send); 
        //CDCOUT(sc_time_stamp()  << name() << " - DUT: Datapath - to_send: " << to_send << endl, 0);
        wait();
      }
      wait();
    } // while 
  }
  
  // HLS, kVectorSize 16, passed II = 1 
  // HLS, kVectorSize 32, failed II = 1 
  spec::AccumMatrixType MatMul(const spec::MatrixType& _mat_in0, const spec::MatrixType& _mat_in1) {
    // one cycle per output matrix row (may need to test HLS in another file)
    spec::AccumMatrixType _mat_out;

    #pragma hls_pipeline_init_interval 2
    for (int i = 0; i < spec::kVectorSize; i++) { // Out inner row
      spec::VectorType _vec_in1;
      NVUINTW(spec::kVectorSize) mask1;
      #pragma hls_unroll yes    
      for (int j = 0; j < spec::kVectorSize; j++) { // copy a row from _mat_in1
        _vec_in1[j] = _mat_in1[i][j];
        mask1[j] = (_mat_in1[i][j] != 0);
      }

      #pragma hls_unroll yes    
      for (int j = 0; j < spec::kVectorSize; j++) { // out inner col  
        spec::VectorType _vec_in0;
        spec::AccumScalarType _out = 0;
        NVUINTW(spec::kVectorSize) mask0;
        #pragma hls_unroll yes    
        for (int k = 0; k < spec::kVectorSize; k++) { // copy a col from _mat_in0
          _vec_in0[k] = _mat_in0[k][j];
          mask0[j] = (_mat_in0[i][j] != 0);
        }
        // Clock gating
        if (mask1 && mask0) {
          ProductSum(_vec_in0, _vec_in1, _out);
        }
        // clock gate zero skipping
        
        _mat_out[i][j] = _out;
      
      }
      wait();
    }

    return _mat_out;
  }
  
  void ProductSum(const spec::VectorType in_0, const spec::VectorType in_1, spec::AccumScalarType& out) {
    spec::AccumScalarType  out_tmp = 0; 
    
    #pragma hls_unroll yes
    #pragma cluster addtree 
    #pragma cluster_type both  
    for (int j = 0; j < spec::kVectorSize; j++) {
      AdpfloatType<8,3> in_0_adpfloat(in_0[j]);
      AdpfloatType<8,3> in_1_adpfloat(in_1[j]);
      spec::AccumScalarType acc_tmp;
      adpfloat_mul(in_0_adpfloat, in_1_adpfloat, acc_tmp);
      out_tmp += acc_tmp;
    }
    out = out_tmp;
  }
};

#endif

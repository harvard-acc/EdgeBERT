
#ifndef __INPUTSETUP_H__
#define __INPUTSETUP_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_connections.h>
#include <nvhls_vector.h>

#include "../include/Spec.h"
#include "../DecodeTop/DecodeTop.h"

SC_MODULE(InputSetup)
{
 public: 
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef DecodeTop MemType; 
  static const int N = spec::N;     // # banks = N
  //static const int log2v = nvhls::nbits<N - 1>::val;
  static const int log2v = nvhls::log2_floor<N>::val; 
  static const int matid = 10 - log2v; // 6 if vector size is 16
  static const int kDebugLevel = 0;

  typedef spec::IndexType IndexType;
  IndexType base_input_reg[2];
  //bool valid[2]; 
  // I/O 
  Connections::Out<bool>  com_IRQ;       // computation IRQ
 
  Connections::In<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations
  
  Connections::In<spec::MatrixConfig> start;  // matrix configurations. Also used as start signal!!!
  Connections::Out<IndexType> base_input[2];  // base address of input reads

  Connections::In<spec::VectorType>  accum_out_vec;    // From Accum
  Connections::Out<spec::VectorType> act_in_vec[2];  // To Datapath

  // Memory channels
  Connections::In<spec::VectorType>  act_dec_rsp[2]; // From the two DecodeTop modules 
  Connections::Out<spec::VectorType> accum_wr_req;    // To DecodeTop to be stored via Encode
  Connections::Out<MemType::mask_req_t>   mask_rd_req[2]; // To DecodeTop for mask read requests

  //Connections::Out<bool>  reset_ctr[2];
  //Connections::Out<bool>  record_ctr[2];

  // interconnect trigger
  Connections::Combinational<spec::MatrixConfig> start_wr_trig[3];
  Connections::Combinational<IndexType> base_input_wr[2];  

  SC_HAS_PROCESS(InputSetup);
  InputSetup(sc_module_name name_) : sc_module(name_) {

    SC_THREAD (ReadReqRun0);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

    SC_THREAD (ReadReqRun1);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);  
    
    SC_THREAD (ReadRspRun0);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

    SC_THREAD (ReadRspRun1);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

    SC_THREAD (WriteReqRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

    SC_THREAD (StartRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);

    SC_THREAD (BaseRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);
  }

  void StartRun() {
    start.Reset();

    #pragma hls_unroll yes
    for (int i=0; i<3; i++) start_wr_trig[i].ResetWrite(); 

    #pragma hls_pipeline_init_interval 1
    while(1) {
      spec::MatrixConfig matrix_config_tmp;
      matrix_config_tmp = start.Pop(); // start computation

      #pragma hls_unroll yes
      for (int i=0; i<3; i++) start_wr_trig[i].Push(matrix_config_tmp);
      CDCOUT(sc_time_stamp()  << name() << " - DUT: InputSetup: Computation Start!!!" << endl, 0);
      wait();
    }
  }

  void BaseRun() {
    #pragma hls_unroll yes
    for (int i=0; i<2; i++) {
      base_input[i].Reset();
      base_input_wr[i].ResetWrite();
      base_input_reg[i] = 0;
    }
    input_buffer_config.Reset();

    //wait(); 
    #pragma hls_pipeline_init_interval 1
    while(1) {
      spec::InputBufferConfig input_buffer_config_tmp;
      if (input_buffer_config.PopNB(input_buffer_config_tmp)) {
          #pragma hls_unroll yes
          for (int i=0; i<2; i++) {
            base_input_reg[i] = input_buffer_config_tmp.base_input[i];
            base_input[i].Push(base_input_reg[i]);
            base_input_wr[i].Push(base_input_reg[i]);
          }
      }

      wait();
    }
  }

  // The main process that generate read req for DecodeTop0
  void ReadReqRun0() {
    //start.Reset();
    start_wr_trig[0].ResetRead();

    base_input_wr[0].ResetRead();
    mask_rd_req[0].Reset();

    //#pragma hls_pipeline_init_interval 1
    while(1) {
      IndexType base;
      if (base_input_wr[0].PopNB(base)) {
        //IndexType base = base_input_wr[0].Pop();

        spec::MatrixConfig matrix_config_tmp = start_wr_trig[0].Pop();
        NVUINT10  N0 = matrix_config_tmp.N0;
        NVUINT10  N1 = matrix_config_tmp.N1;
        NVUINT12  M =  matrix_config_tmp.M;

        NVUINTW(matid) _N0 = N0 >> log2v;
        NVUINTW(matid) _N1 = N1 >> log2v;

        NVUINT24  M_Run0 = _N0*_N1*M;
        NVUINT18  C_Run0 = _N0*M;
        //CDCOUT(sc_time_stamp()  << name() << " - DUT: InputSetup-ReadReqRun0 - _N0: " << _N0 << " _N1: " << _N1 << " M: " << M << endl, 0);
        //cout << "N0 is: " << N0 << ", and N1 is: " << N1 << ", and M is: " << M << endl; 
   
        MemType::mask_req_t req_reg;
        req_reg.type.val = CLITYPE_T::LOAD;

        NVUINT12 count = 0;

        #pragma hls_pipeline_init_interval 1
        for (int i = 0; i < M_Run0; i++) {
          req_reg.addr[0] = base + count;    
          //CDCOUT(sc_time_stamp()  << name() << " - DUT: InputSetup-ReadReqRun0 - addr: " << req_reg.addr[0] << endl, 0); 
          req_reg.valids[0] = 1; 
          if (count == (C_Run0 - 1)) {
             //CDCOUT(sc_time_stamp()  << name() << " - DUT: InputSetup-ReadReqRun0 count reset with count:  " << count << " and mask address: " << req_reg.addr[0] << endl, 0);
             //reset_ctr[0].Push(1);
             count = 0;
          } else { 
            count += 1;
          }
          mask_rd_req[0].Push(req_reg);
          wait();
        } //for
      } // if
      wait();
    } // while(1)
  }

  // The main process that generate read req for DecodeTop1
  void ReadReqRun1() {
    //start.Reset();
    start_wr_trig[1].ResetRead();

    base_input_wr[1].ResetRead();
    mask_rd_req[1].Reset();

    //reset_ctr[1].Reset();
    //record_ctr[1].Reset();

    //#pragma hls_pipeline_init_interval 1
    while(1) {
      IndexType base;
      if (base_input_wr[1].PopNB(base)) {

        spec::MatrixConfig matrix_config_tmp = start_wr_trig[1].Pop();
        NVUINT10  N0 = matrix_config_tmp.N0;
        NVUINT10  N1 = matrix_config_tmp.N1;
        NVUINT12  M =  matrix_config_tmp.M;
        NVUINT12  M_1 = M-1;

        NVUINTW(matid) _N0 = N0 >> log2v;
        NVUINTW(matid) _N1 = N1 >> log2v;
        NVUINT24 M_Run1 = _N0*_N1*M;
        //CDCOUT(sc_time_stamp()  << name() << " - DUT: InputSetup-ReadReqRun1 - _N0: " << _N0 << " _N1: " << _N1 << " M: " << M << endl, 0);

   
        MemType::mask_req_t req_reg;
        req_reg.type.val = CLITYPE_T::LOAD;

        NVUINT12 count = 0;
        //NVUINT12 k = 0; 
        NVUINTW(matid) k = 0; 
        NVUINT12 p = 0; 
        #pragma hls_pipeline_init_interval 1
        for (int i = 0; i < M_Run1; i++) {
          req_reg.addr[0] = base + count + M*p;    
          //CDCOUT(sc_time_stamp()  << name() << " - DUT: InputSetup-ReadReqRun1 - addr: " << req_reg.addr[0] << " and k is: " << k << " and p is: " << p << endl, 0); 
          req_reg.valids[0] = 1; 
          if (count == M_1) {
             //reset_ctr[1].Push(1);
             count = 0;
             k += 1;
             if (k == _N0) {
                //record_ctr[1].Push(1);
                p += 1;
                k = 0;
             }
          }
          else count += 1;

          mask_rd_req[1].Push(req_reg);
          wait();
        } // for
      } // if
      wait();
    } // while(1)
  }


  void ReadRspRun0() {
    act_dec_rsp[0].Reset();
    act_in_vec[0].Reset();
    
    #pragma hls_pipeline_init_interval 1
    while(1) {
      spec::VectorType rsp_reg;
      if (act_dec_rsp[0].PopNB(rsp_reg)) {
         //valid[i] = 1;
         //if (valid[i] == true) act_in_vec[i].Push(rsp_reg[i]);
         act_in_vec[0].Push(rsp_reg);
         //CDCOUT(sc_time_stamp()  << name() << " - DUT: InputSetup/ReadRspRun0 - Pushed Decoded Vectors to Datapath" << endl, kDebugLevel);
        }   
      wait();
    }
  }

  void ReadRspRun1() {
    act_dec_rsp[1].Reset();
    act_in_vec[1].Reset();
    
    #pragma hls_pipeline_init_interval 1
    while(1) {
      spec::VectorType rsp_reg;
      if (act_dec_rsp[1].PopNB(rsp_reg)) {
         //valid[i] = 1;
         //if (valid[i] == true) act_in_vec[i].Push(rsp_reg[i]);
         act_in_vec[1].Push(rsp_reg);
         //CDCOUT(sc_time_stamp()  << name() << " - DUT: InputSetup/ReadRspRun1 - Pushed Decoded Vectors to Datapath" << endl, kDebugLevel);
        }   
      wait();
    }
  }


  void WriteReqRun() {
    accum_out_vec.Reset();
    accum_wr_req.Reset();

    start_wr_trig[2].ResetRead();
    com_IRQ.Reset();

    while(1) {
      spec::MatrixConfig matrix_config_tmp = start_wr_trig[2].Pop();
      NVUINT10  N0 = matrix_config_tmp.N0;
      NVUINT10  N1 = matrix_config_tmp.N1;
      //NVUINT12  M =  matrix_config_tmp.M;

      NVUINTW(matid) _N0 = N0 >> log2v;
      //NVUINTW(matid) _N1 = N1 >> log2v;

      // FIXME: can use lower bit width for counters
      NVUINT16 ctr = 0;
      NVUINT16 C_WrReq = _N0*N1;

      bool is_done = 0; 
      #pragma hls_pipeline_init_interval 1
      while (!is_done) {
        NVUINT1 valid = 0;
        spec::VectorType accum_out_vec_reg;
        
        valid = accum_out_vec.PopNB(accum_out_vec_reg);

        if (valid) {
           ctr += 1;
        }
        
        if (valid != 0) {
          accum_wr_req.Push(accum_out_vec_reg);
          //CDCOUT(sc_time_stamp()  << name() << " - DUT: InputSetup/WriteReqRun - pushed accum_out_vec " << " with ctr: " << ctr << endl, 0);
        }
        //cout << "ctr: " << ctr << endl;

        if (ctr == C_WrReq) {
          is_done = 1;
        }
        wait();
      }
      //cout << "compute IRQ" << endl;

      com_IRQ.Push(1);
      CDCOUT(sc_time_stamp()  << name() << " Pushed com_IRQ Interrupt " << endl, 0);
      wait();
    }
  }
};

#endif


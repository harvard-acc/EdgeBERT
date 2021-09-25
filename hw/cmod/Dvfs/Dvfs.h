#ifndef __DVFS_H__
#define __DVFS_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_connections.h>
#include <nvhls_vector.h>
#include <nvhls_module.h>
#include "../include/Spec.h"
//#include "../DecodeTop/DecodeTop.h"
//#include "../include/PPU.h"
//#include "../AuxMem/AuxMem.h"

SC_MODULE(Dvfs)
{
 public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  //typedef DecodeTop MaskMemType; 
  typedef spec::IndexType IndexType;
  //IndexType base_input_reg; 
  //typedef AuxMem AuxMemType;  
  static const int N = spec::N; 
  //static const int log2v = nvhls::log2_floor<N>::val;
  //NVUINTW(log2v) 
  static const int kDebugLevel = 0;

  NVUINT8   enpy_scale;

  NVUINT6   dco_val_config_a[5];
  NVUINT6   dco_val_config_b[5];
  NVUINT6   dco_val_config_c[5];

  NVUINT8   ldo_val_config_a[4];
  NVUINT8   ldo_val_config_b[4];
  NVUINT8   ldo_val_config_c[4];
  NVUINT8   ldo_val_config_d[4];

  spec::ActScalarType enpy_val_reg, enpy_val_scaled;
  //spec::VddConfig vdd_config_tmp;

  //NVUINT8 freq_opt_tmp;
  //NVUINT16 ncycles_tmp;
  //NVUINT8 vdd_opt_tmp;
  //NVUINT8 freq_scaled;
  //spec::ActScalarType sum_exp, sum_xexp, maximum_value, enpy_result, enpy_threshold;

  Connections::In<bool> start;
  Connections::In<spec::DvfsConfig> dvfs_config;
  //Connections::In<spec::VddConfig> vdd_config;
  //Connections::In<spec::VectorType>  dvfs_param_rsp;  // To AuxMem for read response
  Connections::In<spec::ActScalarType> enpy_val_in;

  //Connections::Out<AuxMemType::aux_req_t>  dvfs_param_req;  // To AuxMem for read request
  Connections::Out<NVUINT6> dco_sel_out;
  Connections::Out<NVUINT8> ldo_sel_out;

  Connections::In<spec::DCOConfigA> dco_config_a;
  Connections::In<spec::DCOConfigB> dco_config_b;
  Connections::In<spec::DCOConfigC> dco_config_c;
  //
  Connections::In<spec::LDOConfigA> ldo_config_a;
  Connections::In<spec::LDOConfigB> ldo_config_b;
  Connections::In<spec::LDOConfigC> ldo_config_c;
  Connections::In<spec::LDOConfigD> ldo_config_d;

  Connections::Out<bool> done;

  // Constructor
  SC_HAS_PROCESS(Dvfs);
  Dvfs(sc_module_name name_) : sc_module(name_) {

    SC_THREAD(DvfsRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);  
  }

  bool is_start, is_start_config, is_start_dco, is_start_ldo, enpy_val_ok, dvfs_config_ok;
  bool dco_config_a_ok, dco_config_b_ok, dco_config_c_ok;
  bool ldo_config_a_ok, ldo_config_b_ok, ldo_config_c_ok, ldo_config_d_ok;

  enum FSM {
    IDLE, SCALE_ENPY, PUSH_VF, SCALE_FREQ, VDD_REC, NEXT
  };
  FSM state;


  void Reset() {
    state = IDLE;
    is_start = 0;
    is_start_config = 0;
    is_start_dco = 0;
    is_start_ldo = 0;
    dvfs_config_ok = 0;
    enpy_val_ok = 0;
    dco_config_a_ok=0;
    dco_config_b_ok=0;
    dco_config_c_ok=0;
    ldo_config_a_ok=0;
    ldo_config_b_ok=0;
    ldo_config_c_ok=0;
    ldo_config_d_ok=0;
    ResetPorts();
    ResetDvfs();
  }
  
  void ResetPorts() { 
    start.Reset();
    done.Reset();
    enpy_val_in.Reset();
    dco_sel_out.Reset();
    ldo_sel_out.Reset();
    dvfs_config.Reset();
    dco_config_a.Reset();
    dco_config_b.Reset();
    dco_config_c.Reset();
    ldo_config_a.Reset();
    ldo_config_b.Reset();
    ldo_config_c.Reset();
    ldo_config_d.Reset();
  }
 
  void ResetDvfs() {
    //freq_opt_tmp = 0;
    //ncycles_tmp = 0;
    //vdd_opt_tmp = 0;
  } 
 
  void CheckStart() {
    if (dvfs_config_ok && enpy_val_ok) {
      is_start_config = 1;
    }
    if (dco_config_a_ok && dco_config_b_ok && dco_config_c_ok) {
      is_start_dco = 1;
    }
    if (ldo_config_a_ok && ldo_config_b_ok && ldo_config_c_ok && ldo_config_d_ok) {
      is_start_ldo = 1;
    }
    bool start_reg;
    if (start.PopNB(start_reg) && is_start_config && is_start_dco && is_start_ldo) {
      is_start = 1;
      CDCOUT(sc_time_stamp()  << name() << " Dvfs Start !!!" << endl, kDebugLevel);
    }
  }

  void RunFSM() {
    spec::ActScalarType enpy_val_tmp;
    if (enpy_val_in.PopNB(enpy_val_tmp)) {
      enpy_val_ok = 1;
      enpy_val_reg = enpy_val_tmp;
      cout << sc_time_stamp()  << name() << " Popped enpy_val: " << enpy_val_tmp << endl;
    }

    spec::DvfsConfig dvfs_config_tmp;
    if (dvfs_config.PopNB(dvfs_config_tmp)) {
      dvfs_config_ok = 1;
      enpy_scale = dvfs_config_tmp.enpy_scale;
      cout << sc_time_stamp() << "DVFS DUT - dvfs_config_ok " << endl;
    }

    spec::DCOConfigA dco_config_a_tmp;
    if (dco_config_a.PopNB(dco_config_a_tmp)) {
      dco_config_a_ok = 1;
      dco_val_config_a[0] = dco_config_a_tmp.dco_val0;
      dco_val_config_a[1] = dco_config_a_tmp.dco_val1;
      dco_val_config_a[2] = dco_config_a_tmp.dco_val2;
      dco_val_config_a[3] = dco_config_a_tmp.dco_val3;
      dco_val_config_a[4] = dco_config_a_tmp.dco_val4;
      cout << sc_time_stamp() << "DVFS DUT - dco_config_a_ok " << endl;
    }

    spec::DCOConfigB dco_config_b_tmp;
    if (dco_config_b.PopNB(dco_config_b_tmp)) {
      dco_config_b_ok = 1;
      dco_val_config_b[0] = dco_config_b_tmp.dco_val5;
      dco_val_config_b[1] = dco_config_b_tmp.dco_val6;
      dco_val_config_b[2] = dco_config_b_tmp.dco_val7;
      dco_val_config_b[3] = dco_config_b_tmp.dco_val8;
      dco_val_config_b[4] = dco_config_b_tmp.dco_val9;
      cout <<  sc_time_stamp() << "DVFS DUT - dco_config_b_ok " << endl;
    }

    spec::DCOConfigC dco_config_c_tmp;
    if (dco_config_c.PopNB(dco_config_c_tmp)) {
      dco_config_c_ok = 1;
      dco_val_config_c[0] = dco_config_c_tmp.dco_val10;
      dco_val_config_c[1] = dco_config_c_tmp.dco_val11;
      dco_val_config_c[2] = dco_config_c_tmp.dco_val12;
      dco_val_config_c[3] = dco_config_c_tmp.dco_val13;
      dco_val_config_c[4] = dco_config_c_tmp.dco_val14;
      cout <<  sc_time_stamp() << "DVFS DUT - dco_config_c_ok " << endl;
    }

    spec::LDOConfigA ldo_config_a_tmp;
    if (ldo_config_a.PopNB(ldo_config_a_tmp)) {
      ldo_config_a_ok = 1;
      ldo_val_config_a[0] = ldo_config_a_tmp.ldo_val0;
      ldo_val_config_a[1] = ldo_config_a_tmp.ldo_val1;
      ldo_val_config_a[2] = ldo_config_a_tmp.ldo_val2;
      ldo_val_config_a[3] = ldo_config_a_tmp.ldo_val3;
      cout << sc_time_stamp() << "DVFS DUT - ldo_config_a_ok " << endl;
    }

    spec::LDOConfigB ldo_config_b_tmp;
    if (ldo_config_b.PopNB(ldo_config_b_tmp)) {
      ldo_config_b_ok = 1;
      ldo_val_config_b[0] = ldo_config_b_tmp.ldo_val4;
      ldo_val_config_b[1] = ldo_config_b_tmp.ldo_val5;
      ldo_val_config_b[2] = ldo_config_b_tmp.ldo_val6;
      ldo_val_config_b[3] = ldo_config_b_tmp.ldo_val7;
      cout <<  sc_time_stamp() << "DVFS DUT - ldo_config_b_ok " << endl;
    }

    spec::LDOConfigC ldo_config_c_tmp;
    if (ldo_config_c.PopNB(ldo_config_c_tmp)) {
      ldo_config_c_ok = 1;
      ldo_val_config_c[0] = ldo_config_c_tmp.ldo_val8;
      ldo_val_config_c[1] = ldo_config_c_tmp.ldo_val9;
      ldo_val_config_c[2] = ldo_config_c_tmp.ldo_val10;
      ldo_val_config_c[3] = ldo_config_c_tmp.ldo_val11;
      cout <<  sc_time_stamp() << "DVFS DUT - ldo_config_c_ok " << endl;
    }

    spec::LDOConfigD ldo_config_d_tmp;
    if (ldo_config_d.PopNB(ldo_config_d_tmp)) {
      ldo_config_d_ok = 1;
      ldo_val_config_d[0] = ldo_config_d_tmp.ldo_val12;
      ldo_val_config_d[1] = ldo_config_d_tmp.ldo_val13;
      ldo_val_config_d[2] = ldo_config_d_tmp.ldo_val14;
      ldo_val_config_d[3] = ldo_config_d_tmp.ldo_val15;
      cout <<  sc_time_stamp() << "DVFS DUT - ldo_config_d_ok " << endl;
    }

    switch (state) {
      case IDLE: {
        ResetDvfs();
        break;
      }
      case SCALE_ENPY: {
        CDCOUT(sc_time_stamp()  << name() << " case SCALE_ENPY" << endl, kDebugLevel);
        enpy_val_scaled = enpy_val_reg / enpy_scale;
        
        CDCOUT(sc_time_stamp()  << name() << " - enpy_val_reg: " << enpy_val_reg << ", enpy_scale: " << enpy_scale << " ,enpy_val_scaled: " << enpy_val_scaled << endl, kDebugLevel);

        break;
      }
      case PUSH_VF: {
        CDCOUT(sc_time_stamp()  << name() << " case PUSH_VF" << endl, kDebugLevel);

        if (enpy_val_scaled < 256) {
           cout << "LUT result: 0" << endl;
           dco_sel_out.Push(dco_val_config_a[0]);
           ldo_sel_out.Push(ldo_val_config_a[0]);

        } else if ((enpy_val_scaled >= 256) && (enpy_val_scaled < 512)) {
           cout << "LUT result: 1" << endl;
           dco_sel_out.Push(dco_val_config_a[1]);
           ldo_sel_out.Push(ldo_val_config_a[1]);

        } else if ((enpy_val_scaled >= 512) && (enpy_val_scaled < 1024)) {
           cout << "LUT result: 2" << endl;
           dco_sel_out.Push(dco_val_config_a[2]);
           ldo_sel_out.Push(ldo_val_config_a[2]);


        } else if ((enpy_val_scaled >= 1024) && (enpy_val_scaled < 2048)) {
           cout << "LUT result: 3" << endl;
           dco_sel_out.Push(dco_val_config_a[3]);
           ldo_sel_out.Push(ldo_val_config_a[3]);
         
        } else if ((enpy_val_scaled >= 2048) && (enpy_val_scaled < 3072)) {
           cout << "LUT result: 4" << endl;
           dco_sel_out.Push(dco_val_config_a[4]);
           ldo_sel_out.Push(ldo_val_config_b[0]);


        } else if ((enpy_val_scaled >= 3072) && (enpy_val_scaled < 4096)) {
           cout << "LUT result: 5" << endl;
           dco_sel_out.Push(dco_val_config_b[0]);
           ldo_sel_out.Push(ldo_val_config_b[1]);


        } else if ((enpy_val_scaled >= 4096) && (enpy_val_scaled < 5120)) {
           cout << "LUT result: 6" << endl;
           dco_sel_out.Push(dco_val_config_b[1]);
           ldo_sel_out.Push(ldo_val_config_b[2]);


        } else if ((enpy_val_scaled >= 5120) && (enpy_val_scaled < 6144)) {
           cout << "LUT result: 7" << endl;
           dco_sel_out.Push(dco_val_config_b[2]);
           ldo_sel_out.Push(ldo_val_config_b[3]);


        } else if ((enpy_val_scaled >= 6144) && (enpy_val_scaled < 8192)) {
           cout << "LUT result: 8" << endl;
           dco_sel_out.Push(dco_val_config_b[3]);
           ldo_sel_out.Push(ldo_val_config_c[0]);


        } else if ((enpy_val_scaled >= 8192) && (enpy_val_scaled < 10240)) {
           cout << "LUT result: 9" << endl;
           dco_sel_out.Push(dco_val_config_b[4]);
           ldo_sel_out.Push(ldo_val_config_c[1]);


        } else if ((enpy_val_scaled >= 10240) && (enpy_val_scaled < 12288)) {
           cout << "LUT result: A" << endl;
           dco_sel_out.Push(dco_val_config_c[0]);
           ldo_sel_out.Push(ldo_val_config_c[2]);


        } else if ((enpy_val_scaled >= 12288) && (enpy_val_scaled < 16384)) {
           cout << "LUT result: B" << endl;
           dco_sel_out.Push(dco_val_config_c[1]);
           ldo_sel_out.Push(ldo_val_config_c[3]);


        } else if ((enpy_val_scaled >= 16384) && (enpy_val_scaled < 20480)) {
           cout << "LUT result: C" << endl;
           dco_sel_out.Push(dco_val_config_c[2]);
           ldo_sel_out.Push(ldo_val_config_d[0]);


        } else if ((enpy_val_scaled >= 20480) && (enpy_val_scaled < 24576)) {
           cout << "LUT result: D" << endl;
           dco_sel_out.Push(dco_val_config_c[3]);
           ldo_sel_out.Push(ldo_val_config_d[1]);


        } else if ((enpy_val_scaled >= 24576) && (enpy_val_scaled < 28672)) {
           cout << "LUT result: E" << endl;
           dco_sel_out.Push(dco_val_config_c[4]);
           ldo_sel_out.Push(ldo_val_config_d[2]);


        } else if (enpy_val_scaled >= 28672) {
           cout << "LUT result: F" << endl;
           dco_sel_out.Push(dco_val_config_c[4]);
           ldo_sel_out.Push(ldo_val_config_d[3]);
        }


        break;
      }

      case NEXT: {
        break;
      }

      default: {
        break;
      }
    }
  }
  
  void UpdateFSM() {
    FSM next_state;
    switch (state) {
      case IDLE: {
        if (is_start) {
          next_state = SCALE_ENPY;
        }
        else {
          next_state = IDLE;
        }
        break;
      }
      case SCALE_ENPY: {
        next_state = PUSH_VF;
        break;
      }
      case PUSH_VF: {
        next_state = NEXT;
        break;
      }
      case NEXT: {
        is_start = 0;
        next_state = IDLE;
        CDCOUT(sc_time_stamp()  <<  name() << " Dvfs Finish" << endl, kDebugLevel);
        done.Push(1);    
        break;
      }
      default: {
        next_state = IDLE;
        break;
      }
    }      
    state = next_state;
  }
  

  void DvfsRun() {
    Reset();

    #pragma hls_pipeline_init_interval 4 
    while(1) {
      RunFSM();
      if (is_start == 0) {
        CheckStart();
      }
      UpdateFSM();
      wait();
    }
  }
};
#endif


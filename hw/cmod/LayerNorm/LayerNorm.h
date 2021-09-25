
#ifndef __LAYERNORM_H__
#define __LAYERNORM_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_connections.h>
#include <nvhls_vector.h>
#include <nvhls_module.h>
#include "../include/Spec.h"
#include "../DecodeTop/DecodeTop.h"
#include "../AuxMem/AuxMem.h"
#include "../include/PPU.h"

SC_MODULE(LayerNorm)
{
 public:
  sc_in_clk     clk;
  sc_in<bool>   rst;

  typedef DecodeTop MaskMemType; 
  typedef AuxMem AuxMemType;
  typedef spec::IndexType IndexType;
  IndexType base_input_reg;   
  static const int N = spec::N; 
  static const int kDebugLevel = 0;

  spec::LayerNormSumType sum, sqsum;
  spec::ActScalarType mean, inv_std;
  spec::ActVectorType negmean_vector, inv_std_vector;
  spec::ActVectorType out_data; 

  Connections::In<bool> start;
  Connections::In<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations
  Connections::In<spec::PeriphConfig> layernorm_config;
  Connections::In<spec::GBControlConfig> gbcontrol_config;
  Connections::In<spec::VectorType>  act_rsp; //Activation response from DecodeTop
  Connections::In<spec::VectorType>  layernorm_param_rsp;  // To AuxMem for read response

  Connections::Out<AuxMemType::aux_req_t>  layernorm_param_req;  // To AuxMem for read request
  Connections::Out<spec::VectorType>  act_out_vec; // To be stored in DecodeTop
  Connections::Out<MaskMemType::mask_req_t>  mask_rd_req; // To DecodeTop for mask read requests
  Connections::Out<bool> done;

  // Constructor
  SC_HAS_PROCESS(LayerNorm);
  LayerNorm(sc_module_name name_) : sc_module(name_) {

    SC_THREAD(LayerNormRun);
    sensitive << clk.pos();
    NVHLS_NEG_RESET_SIGNAL_IS(rst);  
  }

  bool is_start, input_ok, ln_config_ok, gb_config_ok;

  enum FSM {
    IDLE, MEAN, MEAN2, VAR, NORM, NORM2, GAMMA, GAMMA2, BETA, BETA2, BETA3, NEXT
  };
  FSM state;

  spec::PeriphConfig layernorm_config_tmp;
  spec::GBControlConfig gbcontrol_config_tmp;
  spec::InputBufferConfig input_buffer_config_tmp;
  
  NVUINT8  base_gamma;
  NVUINT8  base_beta;
  spec::AdpfloatBiasType adpbias_gamma;
  spec::AdpfloatBiasType adpbias_beta;

  NVUINT8   num_vector;
  NVUINT8   num_timestep;
  spec::AdpfloatBiasType adpbias_act1;

  NVUINT8 vector_index;
  NVUINT8 timestep_index;

  void Reset() {
    state = IDLE;
    is_start = 0;
    input_ok = 0;
    ln_config_ok = 0;
    gb_config_ok = 0;
    base_input_reg = 0;
    base_gamma = 0;
    base_beta = 0;
    adpbias_gamma = 0;
    adpbias_beta = 0;
    num_vector      = 1;
    num_timestep    = 1;
    vector_index = 0;
    timestep_index = 0;
    adpbias_act1    = 0;
    ResetPorts();
    ResetMeanVar();
  }
  
  void ResetPorts() { 
    start.Reset();
    done.Reset();
    act_rsp.Reset();
    input_buffer_config.Reset();
    layernorm_param_rsp.Reset();
    layernorm_param_req.Reset();
    layernorm_config.Reset();
    gbcontrol_config.Reset();
    act_out_vec.Reset();
    mask_rd_req.Reset();
  }
  
  void ResetMeanVar() {
    sum = 0;
    sqsum = 0;
    mean = 0;
    inv_std = 0;
  }
 
  void ResetCounter() {
    vector_index      = 0;
    timestep_index    = 0;  
  }

  NVUINT8 GetVectorIndex() const {
    return vector_index;
  }
  NVUINT8 GetTimestepIndex() const {
    return timestep_index;
  }
  
  void UpdateVectorCounter(bool& is_end) {
    if (vector_index >= (num_vector - 1)) {
      is_end = 1;
      vector_index = 0;
    }
    else {
      vector_index += 1;
    }
  }
  
  void UpdateTimestepCounter(bool& is_end) {
    is_end = 0;
    if (timestep_index >= (num_timestep - 1)) {
      is_end = 1;
      timestep_index = 0;
    }
    else {
      timestep_index += 1;
    }
  } 
 
  void CheckStart() {
    bool start_reg;
    if (start.PopNB(start_reg) && gb_config_ok && ln_config_ok && input_ok) {
      is_start = 1;
      CDCOUT(sc_time_stamp()  << name() << " LayerNorm Start !!!" << endl, kDebugLevel);
    }
  }

  void RunFSM() {
    if (gbcontrol_config.PopNB(gbcontrol_config_tmp)) {
      num_vector = gbcontrol_config_tmp.num_vector;
      num_timestep = gbcontrol_config_tmp.num_timestep;
      adpbias_act1 = gbcontrol_config_tmp.adpbias_act1;
      CDCOUT(sc_time_stamp()  << name() << " DUT - num_vector is: " << num_vector << endl, kDebugLevel);
      CDCOUT(sc_time_stamp()  << name() << " DUT - num_timestep is: " << num_timestep << endl, kDebugLevel);
      gb_config_ok = 1;
    }
    vector_index = GetVectorIndex();
    timestep_index = GetTimestepIndex();
    //CDCOUT(sc_time_stamp()  << name() << " DUT - vector_index: " << vector_index << " and timestep_index: " << timestep_index << endl, kDebugLevel);

    if (layernorm_config.PopNB(layernorm_config_tmp)) {
      base_gamma = layernorm_config_tmp.base_gamma;
      base_beta = layernorm_config_tmp.base_beta;
      adpbias_gamma = layernorm_config_tmp.adpbias_gamma;
      adpbias_beta = layernorm_config_tmp.adpbias_beta;
      ln_config_ok = 1;
    }

    if (input_buffer_config.PopNB(input_buffer_config_tmp)) {
       base_input_reg = input_buffer_config_tmp.base_input[0];
       input_ok = 1;
    }

    //CDCOUT(sc_time_stamp()  << name() << " gb_config_ok: " << gb_config_ok << " and ln_config_ok: " << ln_config_ok << " input_ok: " << input_ok << endl, kDebugLevel);

    switch (state) {
      case IDLE: {
        break;
      }
      case MEAN: {
        CDCOUT(sc_time_stamp()  << name() << " case MEAN" << endl, kDebugLevel);
        // one pass of Large Buffer read to set mean amd sqmean
        // E[x^2] - [E[x]]^2
        // Send Req
        MaskMemType::mask_req_t mask_reg;
        mask_reg.type.val = CLITYPE_T::LOAD;
        mask_reg.addr[0] = base_input_reg + vector_index + timestep_index*num_vector;
        mask_reg.valids[0] = 1;
        mask_rd_req.Push(mask_reg);                
        break;
      }
      case MEAN2: {
        CDCOUT(sc_time_stamp()  << name() << " case MEAN2" << endl, kDebugLevel);
        // Receive Rsp        
        spec::VectorType act_reg;
        act_reg = act_rsp.Pop();
        CDCOUT(sc_time_stamp()  << name() << " MEAN2 popped activation -----------" << endl, kDebugLevel);
        spec::ActVectorType x_vector;
        spec::ActVectorType sq_vector;
        spec::ActScalarType tmp_sum, tmp_sqsum;
        
        // Get activation vector in ActScalarType
        Adpfloat2Fixed(act_reg,  x_vector, adpbias_act1);       
        // Get x^2
        EMul(x_vector, x_vector, sq_vector);
        // sum[x]
        VSum(x_vector, tmp_sum);
        // sum[x^2]
        VSum(sq_vector, tmp_sqsum);        
        
        sum += tmp_sum;
        sqsum += tmp_sqsum;
        break;
      }
      case VAR: {
        CDCOUT(sc_time_stamp()  << name() << " case VAR" << endl, kDebugLevel);
        // calculate the mean of  sum[X]/N, sum[X^2]/N, and E[X]^2
        spec::ActScalarType sqmean, meansq;
        NVINTW(spec::kActWordWidth+1) meansq_tmp;
        NVUINT14 dividsor;
        dividsor = num_vector; 
        dividsor = dividsor << nvhls::log2_floor<spec::kVectorSize>::val; // this val == 4
        // E[X], E[X^2]
        mean = sum / dividsor;
        sqmean = sqsum / dividsor;
        
        // E[X]^2 
        meansq_tmp = mean * mean;
        meansq_tmp = meansq_tmp >> spec::kActNumFrac;
        meansq = meansq_tmp;
        
        // VAR[X] = E[X^2] - E[X]^2
        spec::ActScalarType var = sqmean - meansq;
        // We use inv_std = 1/sqrt(VAR[X])
        SInvSqrt(var, inv_std);
        break;
      }
      case NORM: {
        CDCOUT(sc_time_stamp()  << name() << " case NORM" << endl, kDebugLevel);
        
        #pragma hls_unroll
        for (int i = 0; i < spec::kVectorSize; i++) {
          negmean_vector[i] = -mean; // minus mean
          inv_std_vector[i] = inv_std;
        }
        
        // Send Req
        MaskMemType::mask_req_t mask_reg;
        mask_reg.type.val = CLITYPE_T::LOAD;
        mask_reg.addr[0] = base_input_reg + vector_index + timestep_index*num_vector;
        mask_reg.valids[0] = 1;
        mask_rd_req.Push(mask_reg);          
        break;
      }
      case NORM2: {  
        CDCOUT(sc_time_stamp()  << name() << " case NORM2" << endl, kDebugLevel);
        // Receive Rsp  
        spec::VectorType act_reg;
        //wait(10);
        act_reg = act_rsp.Pop();
        CDCOUT(sc_time_stamp()  << name() << " NORM2 popped activation -----------" << endl, kDebugLevel);

        spec::ActVectorType x_vector;
        Adpfloat2Fixed(act_reg,  x_vector, adpbias_act1);
        
        // Normalize = (X - E[X])*(Inverse Stdev)
        EAdd (x_vector, negmean_vector, x_vector);
        EMul (x_vector, inv_std_vector, x_vector);
        
        // out_data temporary storage
        out_data = 0;
        out_data = x_vector;
        break;
      }
      case GAMMA: {
        CDCOUT(sc_time_stamp()  << name() << " case GAMMA" << endl, kDebugLevel);
        // Get gamma vector
        AuxMemType::aux_req_t  layernorm_param_reg;

        #pragma hls_unroll yes
        for (unsigned i = 0; i < N; i++) { 
          layernorm_param_reg.addr[i] = base_gamma + vector_index*N + i;
          layernorm_param_reg.valids[i] = 1;
        }
        layernorm_param_reg.type.val = CLITYPE_T::LOAD;
        layernorm_param_req.Push(layernorm_param_reg);
        break;
      }
      case GAMMA2: { 
        CDCOUT(sc_time_stamp()  << name() << " case GAMMA2" << endl, kDebugLevel);
        spec::VectorType  layernorm_param_rsp_reg;
        layernorm_param_rsp_reg  = layernorm_param_rsp.Pop();        
        CDCOUT(sc_time_stamp()  << name() << "--Popped GAMMA variable" << endl, kDebugLevel);
 
        spec::ActVectorType gamma_vector, vtmp;

        Adpfloat2Fixed(layernorm_param_rsp_reg, gamma_vector, adpbias_gamma);       
      
        // Mul gamma
        EMul (out_data, gamma_vector, vtmp);        
        out_data = vtmp;
        break;
      }
      case BETA: {
        CDCOUT(sc_time_stamp()  << name() << " case BETA" << endl, kDebugLevel);
        // Get Beta vector 
        AuxMemType::aux_req_t  layernorm_param_reg;

        #pragma hls_unroll yes
        for (unsigned i = 0; i < N; i++) { 
          layernorm_param_reg.addr[i] = base_gamma + vector_index*N + i;
          layernorm_param_reg.valids[i] = 1;
        }
        layernorm_param_reg.type.val = CLITYPE_T::LOAD;
        layernorm_param_req.Push(layernorm_param_reg);
        break;
      } 
      case BETA2: {
        CDCOUT(sc_time_stamp()  << name() << " case BETA2" << endl, kDebugLevel);
        spec::VectorType layernorm_param_rsp_reg;
        layernorm_param_rsp_reg  = layernorm_param_rsp.Pop(); 
        CDCOUT(sc_time_stamp()  << name() << "--Popped BETA variable" << endl, kDebugLevel);

        spec::ActVectorType beta_vector, vtmp;
        Adpfloat2Fixed(layernorm_param_rsp_reg, beta_vector, adpbias_beta);       
       
        // Add Beta
        EAdd (out_data, beta_vector, vtmp);  
        out_data = vtmp;
        break;
     }
     case BETA3: {
        CDCOUT(sc_time_stamp()  << name() << " case BETA3" << endl, kDebugLevel);
       
        spec::VectorType out_reg; 
        Fixed2Adpfloat (out_data, out_reg, adpbias_act1);
        act_out_vec.Push(out_reg);               
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
          ResetCounter();
          ResetMeanVar();
          next_state = MEAN;
        }
        else {
          next_state = IDLE;
        }
        break;
      }
      case MEAN: {
        next_state = MEAN2;
        break;
      }
      case MEAN2: { // first data read
        bool is_end = 0;
        UpdateVectorCounter(is_end);
        if (is_end) {
          next_state = VAR;
        }
        else {
          next_state = MEAN;
        }
        break;
      }
      case VAR: {
        next_state = NORM;
        break;        
      }
      case NORM: { // second data read NORM, GAMMA BETA
        next_state = NORM2;
        break;
      }
      case NORM2: { 
        next_state = GAMMA;
        break;
      }
      case GAMMA: {
        next_state = GAMMA2;
        break;
      }
      case GAMMA2: {
        next_state = BETA; 
        break;
      }
      case BETA: {
        next_state = BETA2;
        break;
      }
      case BETA2: {
        next_state = BETA3;
        break;
      }
      case BETA3: { 
        bool is_end = 0;
        UpdateVectorCounter(is_end);
        if (is_end) {
          next_state = NEXT;
        }
        else {
          next_state = NORM;
        }
        break;
      }
      case NEXT: {
        // Reset Temporary state during calculation 
        ResetMeanVar();
        // Move to next timestep
        bool is_end = 0;
        UpdateTimestepCounter(is_end);
        if (is_end) {
          is_start = 0;
          next_state = IDLE;
          CDCOUT(sc_time_stamp()  <<  name() << " LayerNorm Finish" << endl, kDebugLevel);
          done.Push(1);    
        }
        else {
          next_state = MEAN;
        }
        break;
      }
      default: {
        next_state = IDLE;
        break;
      }
      
    }      
    state = next_state;
  }
  

  void LayerNormRun() {
    Reset();

    #pragma hls_pipeline_init_interval 3 
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


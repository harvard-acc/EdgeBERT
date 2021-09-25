/*
 */

#include "PUModule.h"
#include "../include/Spec.h"
#include "../include/utils.h"
#include "../include/helper.h"


#include <systemc.h>
#include <mc_scverify.h>
#include <nvhls_int.h>
#include <vector>

#define NVHLS_VERIFY_BLOCKS (PUModule)
#include <nvhls_verify.h>
using namespace::std;
const int TestNumTiles = 256;

SC_MODULE (Source) {
  sc_in<bool> clk;
  sc_in<bool> rst;

  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;
  typedef spec::IndexType IndexType;

  //Mux selectors
  Connections::Out<bool> flip_mem;
  Connections::Out<bool> use_axi;
  Connections::Out<bool> use_gb;
  Connections::Out<NVUINT12> reset_mode;
  Connections::Out<spec::GBControlConfig> gbcontrol_config;


  //From/To InputAXI and MaskAXI
  Connections::Out<input_req_t> input_mem_wr_req;
  Connections::Out<input_req_t> input_mem_rd_req;  
  Connections::Out<mask_req_t> mask_mem_wr_req;
  Connections::Out<mask_req_t> mask_mem_rd_req;

  Connections::In<mask_rsp_t>   axi_mask_rsp_out;
  Connections::In<input_rsp_t>  axi_input_rsp_out;

  //From/To GB
  Connections::Out<spec::VectorType> activation_from_gb_to_encoder;
  Connections::Out<mask_req_t> mask_req_out_from_gb_to_maskmem;

  Connections::In<spec::VectorType> activation_from_mem_to_gb;

  Connections::Out<IndexType> base_output;
  Connections::Out<spec::AccelConfig> accel_config;

  //From/To InputSetup
  Connections::Out<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations
  Connections::Out<spec::MatrixConfig> mat_config;  // matrix configurations. Also used as start signal!!!

  Connections::In<bool>  com_IRQ;      // computation IRQ

  Connections::Out<spec::InputBufferBaseOffsetConfig> offset_config;

  Connections::Combinational<bool> trigger_wr1;
  Connections::Combinational<bool> trigger_wr2;

  NVUINT8 test_matrix[TestNumTiles][spec::kVectorSize];
  NVUINT8 test_matrix_1[TestNumTiles][spec::kVectorSize];
  NVUINT1 test_mask[TestNumTiles][spec::kVectorSize];
  NVUINTW(spec::kVectorSize) maskdata;

  SC_CTOR(Source) {
    SC_THREAD(InRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(OutRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(BaseRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void BaseRun() {
    trigger_wr1.ResetRead();
    trigger_wr2.ResetRead();
    base_output.Reset();
    accel_config.Reset();
    gbcontrol_config.Reset();
    input_buffer_config.Reset();
    offset_config.Reset();
    use_gb.Reset();
    use_axi.Reset();
    reset_mode.Reset();
    mat_config.Reset();

    spec::AccelConfig config_reg;
    config_reg.is_relu = 1; 
    config_reg.is_bias = 1; 
    config_reg.weight_bias = 3; 
    config_reg.adf_accum_bias = 2; 
    config_reg.accum_right_shift = 8; 

    spec::GBControlConfig gbcontrol_config_reg;
    gbcontrol_config_reg.num_vector = 2;
    gbcontrol_config_reg.num_timestep = 2;
    gbcontrol_config_reg.adpbias_act1 = 2; 
    gbcontrol_config_reg.adpbias_act2 = 2; 
    gbcontrol_config_reg.adpbias_act3 = 2; 

    spec::InputBufferConfig input_buffer_config_reg;
    input_buffer_config_reg.base_input[0] = 1024;
    input_buffer_config_reg.base_input[1] = 1024;

    spec::InputBufferBaseOffsetConfig offset_config_reg;
    offset_config_reg.base_input_offset[0] = 0;
    offset_config_reg.base_input_offset[1] = 0;

    spec::MatrixConfig mat_config_reg; 
    mat_config_reg.N0 = 64;
    mat_config_reg.N1 = 64;
    mat_config_reg.M = 64;

     while (1) {
      bool reg;
      if (trigger_wr1.PopNB(reg)) {
         use_gb.Push(0);
         use_axi.Push(0);
         reset_mode.Push(129); //reset_mode_wr[1] = 1, reset_mode_wr[0]=0

         /*base_output.Push(1024); 
         accel_config.Push(config_reg);   
         gbcontrol_config.Push(gbcontrol_config_reg);
         input_buffer_config.Push(input_buffer_config_reg);
         offset_config.Push(offset_config_reg); */
      }
      bool reg2;
      if (trigger_wr2.PopNB(reg2)) {
         base_output.Push(1024); 
         accel_config.Push(config_reg);   
         gbcontrol_config.Push(gbcontrol_config_reg);
         input_buffer_config.Push(input_buffer_config_reg);
         offset_config.Push(offset_config_reg); 
         mat_config.Push(mat_config_reg);
      }
      wait();
     }
  }

  void InRun() {
    trigger_wr1.ResetWrite();
    trigger_wr2.ResetWrite();
    flip_mem.Reset();

    input_mem_wr_req.Reset();
    input_mem_rd_req.Reset();
    mask_mem_wr_req.Reset();
    mask_mem_rd_req.Reset();

    activation_from_gb_to_encoder.Reset();
    mask_req_out_from_gb_to_maskmem.Reset();

    //mat_config.Reset();


    wait();
    flip_mem.Push(0);    
    
    NVUINT12 base_addr = 1024;
    wait(10);
    cout  << sc_time_stamp() << "  ------Storing Mask into Mask Mem0--------" << endl;
    for (int i = 0; i < TestNumTiles; i++) {
      mask_req_t mask_reg;
      mask_reg.type.val = CLITYPE_T::STORE;
      for (int j = 0; j < spec::kVectorSize; j++) {
        maskdata.set_slc(j, test_mask[i][j]);
        //cout << "test_mask: " << test_mask[i][j] << endl; 
      }
      mask_reg.data[0] = maskdata;
      mask_reg.addr[0] = base_addr + i;
      mask_reg.valids[0] = 1;
      //cout << sc_time_stamp() << " Testbench Storing mask_reg = " << mask_reg.data[0] << " at address: " << mask_reg.addr[0] << endl;
      mask_mem_wr_req.Push(mask_reg);
      wait(10);
    }

    wait(10);
    cout  << sc_time_stamp() << "--------Storing Data via Innput mem0 --------" << endl;
    for (int i = 0; i < TestNumTiles; i++) {
      input_req_t input_reg;
      input_reg.type.val = CLITYPE_T::STORE;
      for (int j = 0; j < spec::kVectorSize; j++) {
        input_reg.data[j] = test_matrix[i][j];
        input_reg.addr[j] = base_addr + j + i*(spec::kVectorSize);
        input_reg.valids[j] = 1;
        //cout << sc_time_stamp() << " Testbench Storing input_reg.data[i] = " << input_reg.data[j] << " at address: " << input_reg.addr[j] << endl;
      }
      input_mem_wr_req.Push(input_reg);
      wait(100);
    }
    wait(100);

    flip_mem.Push(1);    

    wait(10);
    cout  << sc_time_stamp() << "  ------Storing Mask into Mask Mem1--------" << endl;
    for (int i = 0; i < TestNumTiles; i++) {
      mask_req_t mask_reg;
      mask_reg.type.val = CLITYPE_T::STORE;
      for (int j = 0; j < spec::kVectorSize; j++) {
        maskdata.set_slc(j, test_mask[i][j]);
        //cout << "test_mask: " << test_mask[i][j] << endl; 
      }
      mask_reg.data[0] = maskdata;
      mask_reg.addr[0] = base_addr + i;
      mask_reg.valids[0] = 1;
      //cout << sc_time_stamp() << " Testbench Storing mask_reg = " << mask_reg.data[0] << " at address: " << mask_reg.addr[0] << endl;
      mask_mem_wr_req.Push(mask_reg);
      wait(10);
    }

    wait(10);
    cout  << sc_time_stamp() << "--------Storing Data via Input mem1--------" << endl;
    for (int i = 0; i < TestNumTiles; i++) {
      input_req_t input_reg;
      input_reg.type.val = CLITYPE_T::STORE;
      for (int j = 0; j < spec::kVectorSize; j++) {
        input_reg.data[j] = test_matrix[i][j];
        input_reg.addr[j] = base_addr + j + i*(spec::kVectorSize);
        input_reg.valids[j] = 1;
        //cout << sc_time_stamp() << " Testbench Storing input_reg.data[i] = " << input_reg.data[j] << " at address: " << input_reg.addr[j] << endl;
      }
      input_mem_wr_req.Push(input_reg);
      wait(100);
    }

    wait(10);

    trigger_wr1.Push(1);

    wait(100);
    trigger_wr2.Push(1);

    /*spec::MatrixConfig mat_config_reg; 
    mat_config_reg.N0 = 32;
    mat_config_reg.N1 = 32;
    mat_config_reg.M = 32;
    mat_config.Push(mat_config_reg); */
    cout  << sc_time_stamp() << " TB: pushed start!" << endl;
    wait(); 
  }


  void OutRun() {
    axi_mask_rsp_out.Reset();
    axi_input_rsp_out.Reset();
    activation_from_mem_to_gb.Reset();
    com_IRQ.Reset();

    wait();

    while(1) {
      mask_rsp_t axi_mask_rsp_out_reg;
      input_rsp_t axi_input_rsp_out_reg;
      spec::VectorType activation_from_mem_to_gb_reg;
      bool com_IRQ_reg;

      if (activation_from_mem_to_gb.PopNB(activation_from_mem_to_gb_reg)) {
         cout << hex << sc_time_stamp() << " TB - Popped activation_from_mem_to_gb_reg data!" << endl;
      }

      if (axi_input_rsp_out.PopNB(axi_input_rsp_out_reg)) {
         cout << hex << sc_time_stamp() << " TB - Popped axi_input_rsp_out signal = " << axi_input_rsp_out_reg << endl;
      }

      if (axi_mask_rsp_out.PopNB(axi_mask_rsp_out_reg)) {
         cout << hex << sc_time_stamp() << " TB - Popped axi_mask_rsp_out data = " << endl;
      }

      if (com_IRQ.PopNB(com_IRQ_reg)) {
         cout << hex << sc_time_stamp() << " TB - Popped done/com_IRQ signal = " << com_IRQ_reg << endl;
         wait(1000);
         sc_stop();
      }

      wait();
    }
  }


};

SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);
	sc_clock clk;
  sc_signal<bool> rst;

  typedef spec::input_req_t input_req_t;
  typedef spec::input_rsp_t input_rsp_t;
  typedef spec::mask_req_t mask_req_t;
  typedef spec::mask_rsp_t mask_rsp_t;
  typedef spec::IndexType IndexType;

  Connections::Combinational<bool> flip_mem;
  Connections::Combinational<bool> use_axi;
  Connections::Combinational<bool> use_gb;
  Connections::Combinational<input_req_t> input_mem_wr_req;
  Connections::Combinational<input_req_t> input_mem_rd_req;  
  Connections::Combinational<mask_req_t> mask_mem_wr_req;
  Connections::Combinational<mask_req_t> mask_mem_rd_req;
  Connections::Combinational<mask_rsp_t>   axi_mask_rsp_out;
  Connections::Combinational<input_rsp_t>  axi_input_rsp_out;
  Connections::Combinational<spec::VectorType> activation_from_gb_to_encoder;
  Connections::Combinational<mask_req_t> mask_req_out_from_gb_to_maskmem;
  Connections::Combinational<spec::VectorType> activation_from_mem_to_gb;
  Connections::Combinational<IndexType> base_output;
  Connections::Combinational<spec::AccelConfig> accel_config;
  Connections::Combinational<spec::InputBufferConfig> input_buffer_config;  // input buffer configurations
  Connections::Combinational<spec::MatrixConfig> mat_config;  // matrix configurations. Also used as start signal!!!
  Connections::Combinational<bool>  com_IRQ;      // computation IRQ
  Connections::Combinational<spec::InputBufferBaseOffsetConfig> offset_config;
  Connections::Combinational<NVUINT12> reset_mode;
  Connections::Combinational<spec::GBControlConfig> gbcontrol_config;

  Source src;
  NVHLS_DESIGN(PUModule) dut;

  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
    rst("rst"),
    src("src"),
    dut("dut")
  {
    dut.clk(clk);
    dut.rst(rst);
    dut.flip_mem(flip_mem);
    dut.use_axi(use_axi);
    dut.use_gb(use_gb);
    dut.input_mem_wr_req(input_mem_wr_req);
    dut.input_mem_rd_req(input_mem_rd_req);
    dut.mask_mem_wr_req(mask_mem_wr_req);
    dut.mask_mem_rd_req(mask_mem_rd_req);
    dut.axi_mask_rsp_out(axi_mask_rsp_out);
    dut.axi_input_rsp_out(axi_input_rsp_out);
    dut.activation_from_gb_to_encoder(activation_from_gb_to_encoder);
    dut.mask_req_out_from_gb_to_maskmem(mask_req_out_from_gb_to_maskmem);
    dut.activation_from_mem_to_gb(activation_from_mem_to_gb);
    dut.base_output(base_output);
    dut.accel_config(accel_config);
    dut.input_buffer_config(input_buffer_config);
    dut.mat_config(mat_config);
    dut.com_IRQ(com_IRQ);
    dut.offset_config(offset_config);
    dut.reset_mode(reset_mode);
    dut.gbcontrol_config(gbcontrol_config);

    src.clk(clk);
    src.rst(rst);
    src.flip_mem(flip_mem);
    src.use_axi(use_axi);
    src.use_gb(use_gb);
    src.input_mem_wr_req(input_mem_wr_req);
    src.input_mem_rd_req(input_mem_rd_req);
    src.mask_mem_wr_req(mask_mem_wr_req);
    src.mask_mem_rd_req(mask_mem_rd_req);
    src.axi_mask_rsp_out(axi_mask_rsp_out);
    src.axi_input_rsp_out(axi_input_rsp_out);
    src.activation_from_gb_to_encoder(activation_from_gb_to_encoder);
    src.mask_req_out_from_gb_to_maskmem(mask_req_out_from_gb_to_maskmem);
    src.activation_from_mem_to_gb(activation_from_mem_to_gb);
    src.base_output(base_output);
    src.accel_config(accel_config);
    src.input_buffer_config(input_buffer_config);
    src.mat_config(mat_config);
    src.com_IRQ(com_IRQ);
    src.offset_config(offset_config);
    src.reset_mode(reset_mode);
    src.gbcontrol_config(gbcontrol_config);

    SC_THREAD(run); 
  }
  
  void run(){
    for (int i = 0; i < TestNumTiles; i++) {
      for (int j = 0; j < spec::kVectorSize; j++) {
        //src.test_matrix[i][j] = 11;
        src.test_matrix[i][j] = j + i*(spec::kVectorSize);
        //src.test_matrix_1[i][j] = j + i*(spec::kVectorSize) + 1;
        src.test_mask[i][j] = 1;
      }
    }
    /*src.test_mask[0][0] = 1;//15
    src.test_mask[1][0] = 0;//14 
    src.test_mask[2][1] = 0;//13 
    src.test_mask[3][2] = 0;//11
    src.test_mask[4][3] = 0;//7
    src.test_mask[5][0] = 0;
    src.test_mask[5][3] = 0;//6
    src.test_mask[6][1] = 0;
    src.test_mask[6][3] = 0;//5
    src.test_mask[7][3] = 0;
    src.test_mask[7][1] = 0;
    src.test_mask[7][0] = 0;//4 */ 

    
	  wait(2, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" Asserting reset" << std::endl;
    rst.write(false);
    wait(2, SC_NS );
    rst.write(true);
    std::cout << "@" << sc_time_stamp() <<" De-Asserting reset" << std::endl;
    wait(150000, SC_NS);
    std::cout << "@" << sc_time_stamp() <<" sc_stop" << std::endl;
    sc_stop();
  }
};


int sc_main(int argc, char *argv[]) {
  
  cout << "Vector Size = " << spec::kVectorSize << endl;
  testbench tb("tb");
  sc_report_handler::set_actions(SC_ERROR, SC_DISPLAY);
  sc_start();

  bool rc = (sc_report_handler::get_count(SC_ERROR) > 0);
  if (rc)
    DCOUT("TESTBENCH FAIL" << endl);
  else
    DCOUT("TESTBENCH PASS" << endl);
  return rc;
}

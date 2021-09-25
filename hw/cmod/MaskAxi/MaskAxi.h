/*
 * Mask Axi Master Port 
 *  
 */

#ifndef __MASKAXI_H__
#define __MASKAXI_H__

#include <systemc.h>
#include <nvhls_int.h>
#include <nvhls_connections.h>
#include <nvhls_vector.h>

#include <string>
#include "../include/Spec.h"
#include "../include/AxiSpec.h"
#include "../DecodeTop/DecodeTop.h"

SC_MODULE(MaskAxi)
{
  // Customized data type for triggering read request to Memory run 
  //   for master write task 
  class WriteReqTrig: public nvhls_message{
   public:
    spec::MaskAddrType addr;
    NVUINT8 len;               // len - 1, i.e. 1~256     
    static const unsigned int width = spec::MaskAddrType::width + 8;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& addr;
      m& len;
    }
  };


 public: 
  sc_in_clk     clk;
  sc_in<bool>   rst;

  //typedef spec::InputType InputType;  
  typedef DecodeTop MaskMemType;  
  static const int N = spec::N;     // # banks = N
  static const int log2_vectorsize = nvhls::nbits<N - 1>::val;
  static const int Packet = 64 >> log2_vectorsize;

  // Customized data type master axi read/write trigger 
  class MasterTrig: public nvhls_message{
   public:
    NVUINT32              base_addr;
    spec::IndexType M_1;         // column M = M_1 + 1
    static const unsigned int width = 32 + spec::IndexBits;

    template <unsigned int Size>
    void Marshall(Marshaller<Size>& m) {
      m& base_addr;
      m& M_1;
    }
  };
  
  // Master read request interfaces
  typename spec::AxiData::axi4_data::read::template master<> if_data_rd;
  typename spec::AxiData::axi4_data::write::template master<> if_data_wr;

  // IRQs
  Connections::Out<bool> rd_IRQ;
  Connections::Out<bool> wr_IRQ;

  // Trigger for Control.h
  Connections::In<MasterTrig> master_read;
  Connections::In<MasterTrig> master_write;
  Connections::In<spec::IndexType> base_output;

  // Memory req for master read operation
  Connections::Out<MaskMemType::mask_req_t> mem_wr_req;
  // Memory req/rsp for master write operation
  Connections::Out<MaskMemType::mask_req_t> mem_rd_req;
  Connections::In<MaskMemType::mask_rsp_t> mem_rd_rsp; 
  
  // Interconnect
  Connections::Combinational<WriteReqTrig> write_req_trig;
  
  SC_HAS_PROCESS(MaskAxi);
  MaskAxi(sc_module_name name_) : 
    sc_module(name_),
    if_data_rd("if_data_rd"),
    if_data_wr("if_data_wr")
  {

    SC_THREAD (MasterRead);
      sensitive << clk.pos();
      NVHLS_NEG_RESET_SIGNAL_IS(rst);
    
    SC_THREAD (MasterWriteReq);
      sensitive << clk.pos();
      NVHLS_NEG_RESET_SIGNAL_IS(rst);

    SC_THREAD (MasterWrite);
      sensitive << clk.pos();
      NVHLS_NEG_RESET_SIGNAL_IS(rst);
  
  }

  // Load data from outside
  void MasterRead() {
    // Axi Read master
    if_data_rd.reset();
    // Trigger
    master_read.Reset();
    // Send write request to DecodeTop.h 
    mem_wr_req.Reset();
    rd_IRQ.Reset();
    while(1) {
      MaskMemType::mask_req_t req_reg;
      NVUINT12 num_words = 0;
      NVUINT32 base_addr = 0;
      req_reg.type.val = CLITYPE_T::STORE;
      //req_reg.opcode = STORE;

      // Master read trigger (base_addr, and M-1)
      MasterTrig master_trig = master_read.Pop();
      spec::IndexType M_1 = master_trig.M_1;

      // Use base_addr and M_1 to determine 
      //   number of 64-bit words
      base_addr = master_trig.base_addr;
      // mask_matrix size = N*(M_1+1), num_words = mask_matrix_size / 64
      num_words = (N*(M_1+1)) >> 6;

      // For each burst
      NVUINT4 burst_count = 0;
      while (num_words > 0) {
        // 1. Push addr_pld
        typename spec::AxiData::axi4_data::AddrPayload  rd_addr_pld;
        if (num_words > 256) {
          rd_addr_pld.len = 255; // num data in this = len +1 
	        num_words -= 256;
        }
	      else {
          NVHLS_ASSERT (num_words > 0); 
	        rd_addr_pld.len = num_words-1;
          num_words = 0;
        }

        // Push read address and len
        rd_addr_pld.addr = base_addr + burst_count*Packet*(rd_addr_pld.len+1);
        if_data_rd.ar.Push(rd_addr_pld);

        // 2. Recieve Data (don't need to check is last)
        //   Only this part needs to be II=1
        typename spec::AxiData::axi4_data::ReadPayload rd_data_pld;
        #pragma hls_pipeline_init_interval 1
        for (int i = 0; i <= rd_addr_pld.len; i++) {
          rd_data_pld = if_data_rd.r.Pop();  // for each data payload
          // organize write quest to DecodeTop.h, only 8 channels are used
          //   note that address is sequential
          //cout << hex << "rd_data_pld.data: " << rd_data_pld.data << endl;
          
          #pragma hls_pipeline_init_interval 1
          for (int j = 0; j < Packet; j++) {
              req_reg.data[0] = nvhls::get_slc<N>(rd_data_pld.data, N*j);
              //cout << dec << j << "\t" << req_reg.data[j] << endl;
              req_reg.addr[0] = j + Packet*i + burst_count*Packet*(rd_addr_pld.len+1);
	            req_reg.valids[0] = 1;

              mem_wr_req.Push(req_reg);
              wait();
          }
	      }
        // Finish one burst
        burst_count += 1;
      	wait();
      }
      // Finish all bursts for this master read task
      rd_IRQ.Push(1);
      wait();
    }
  }


  // This SC_THREAD is handling memory read req 
  //   during a burst in master write task
  void MasterWriteReq() {
    write_req_trig.ResetRead();
    mem_rd_req.Reset();
    while(1) {
      MaskMemType::mask_req_t req_reg;
      req_reg.type.val = CLITYPE_T::LOAD;  // Read from DecodeTop.h
      //req_reg.opcode = LOAD;  // Read from DecodeTop.h

      WriteReqTrig w_req_trig_tmp = write_req_trig.Pop(); // Pop trigger

      spec::MaskAddrType addr = w_req_trig_tmp.addr;    // Base addr of this burst
      NVUINT8               len = w_req_trig_tmp.len; // (len+1) 64-bit words
      
      //cout << "trig.addr: " << addr << "\t" << "trig.len: " << len << endl;
      //cout << "addr bit: " << spec::MaskAddrType::width << endl;
      #pragma hls_pipeline_init_interval 1
      for (int i = 0; i <= len; i++) {
        #pragma hls_pipeline_init_interval 1
        for (int j = 0; j < Packet; j++) {
	        req_reg.data[0] = 0;
	        req_reg.addr[0] = j + i*Packet + addr;
          req_reg.valids[0] = 1;
          //cout << "\t\t\t" << req_reg.addr[0] << endl;
          mem_rd_req.Push(req_reg);
          wait();
        }
      }
      wait();
    }
  }

   
  // Read Memory, send data to slave
  //   No write response from slave 
  void MasterWrite() {
    // Memory interface
    mem_rd_rsp.Reset(); 
    // Req trigger
    write_req_trig.ResetWrite();
    // AXI master write interface
    if_data_wr.reset();
    // Trigger
    master_write.Reset();
    // IRQ
    wr_IRQ.Reset();
    base_output.Reset();
    bool valid = 0;
    while(1) {
      NVUINT12 num_words = 0;
      NVUINT32 base_addr = 0;

      spec::IndexType base_output_reg;
      if (base_output.PopNB(base_output_reg)) {
         valid = 1;
      }
      if (valid == 1) { 
        // Master Write trigger (base_addr, and num_words)
        MasterTrig master_trig = master_write.Pop();
        spec::IndexType M_1 = master_trig.M_1;
       
        // Use base_addr and num_words to determine 
        //     number of requests (64-bit)
        base_addr = master_trig.base_addr;
        num_words = (N*(M_1+1)) >> 6;
       
        // For each AXI burst (assume at most 7 bursts)
        NVUINT3 burst_count = 0;
        while (num_words > 0) {
          // 1. Push addr_pld
          typename spec::AxiData::axi4_data::AddrPayload  wr_addr_pld;
          if (num_words > 256) {
            wr_addr_pld.len = 255;
            num_words -= 256;
          }
          else {
            wr_addr_pld.len = num_words-1;
            num_words = 0;
          }
          wr_addr_pld.addr = base_addr + burst_count*Packet*(wr_addr_pld.len+1);
          if_data_wr.aw.Push(wr_addr_pld);
   
          // 2. Push req trigger
          WriteReqTrig w_req_trig_tmp;
          w_req_trig_tmp.addr = (wr_addr_pld.len+1)*Packet*burst_count + base_output_reg; // base addr of read req
          w_req_trig_tmp.len = wr_addr_pld.len; // (len+1) 64-bit words
          write_req_trig.Push(w_req_trig_tmp);

          // 3. Send rsp to master write, only this step requires II=1 
          MaskMemType::mask_rsp_t rsp_reg;
          typename spec::AxiData::axi4_data::WritePayload wr_data_pld;
          #pragma hls_pipeline_init_interval 1
          for (int i = 0; i <= wr_addr_pld.len; i++) {
            //rsp_reg = mem_rd_rsp.Pop();
            //cout << "master write mem rsp pop " << rsp_reg.data[0] << endl;
            // only use 8 channels
            #pragma hls_pipeline_init_interval 1
            for (int j = 0; j < Packet; j++) {
              rsp_reg = mem_rd_rsp.Pop();
              wr_data_pld.data.set_slc<N>(N*j, rsp_reg.data[0]);
              NVHLS_ASSERT(rsp_reg.valids[0] == true)
            }
            wr_data_pld.wstrb = ~0;
            wr_data_pld.last = (i == wr_addr_pld.len);
            
            if_data_wr.w.Push(wr_data_pld);
            wait();
          }
          // FIXME pop write response from slave?
          if_data_wr.b.Pop();
          
          burst_count += 1;
          wait();
        } // while(num_words > 0)
        // TODO push done signal to controller
        wr_IRQ.Push(1);
        CDCOUT(sc_time_stamp()  << name() << " Pushed MaskAxi Interrupt " << endl, 0);
      } // if valid == 1
      wait();
    }
  }
};

#endif


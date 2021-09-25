#ifndef __AXI_SPEC__
#define __AXI_SPEC__

/* For EdgeBERT connections, we have two sets of AXI
 * 64-bit Axi Master "axi_data" for master data requests
 * 32-bit Axi Slave "axi_conf" for axi config
*/

#include <nvhls_int.h>
#include <nvhls_types.h>

#include "axi/AxiSplitter.h"
#include "axi/AxiArbiter.h"
#include "axi/AxiSlaveToReadyValid.h"

namespace spec {
  
  // FIXME: 64-bit master port 
  //        need to make sure the interface matches Esp-Noc
  //        set useWriteResponses = 0? 
  namespace AxiData {
    struct axiCfg {
      enum {
        dataWidth = 64,
        useVariableBeatSize = 0,
        useMisalignedAddresses = 0,
        useLast = 1,
        useWriteStrobes = 1,
        useBurst = 1, useFixedBurst = 0, useWrapBurst = 0, maxBurstSize = 256,
        useQoS = 0, useLock = 0, useProt = 0, useCache = 0, useRegion = 0,
        aUserWidth = 0, wUserWidth = 0, bUserWidth = 0, rUserWidth = 0,
        addrWidth = 32,
        idWidth = 10,
        useWriteResponses = 1,
      };
    };
    typedef typename axi::axi4<axiCfg> axi4_data;

    // AxiArbiter template <typename axiCfg, int numMasters, int maxOutstandingRequests>
    typedef AxiArbiter<axiCfg, 3, 1> ArbiterData;
  }
  
  // FIXME: 
  //   * Need to enable burst to avoid compile error in SlaveRV
  //   * Set maxBurstSize = 2 to avoid compile error
  //   * Write strobes enabled though not used to avoid error
  //   * set useWriteResponses = 0? 
  //   * set WriteStrobes = 0? 
  namespace AxiConf {
    struct axiCfg {
      enum {
        dataWidth = 32,
        useVariableBeatSize = 0,
        useMisalignedAddresses = 0,
        useLast = 1,
        useWriteStrobes = 1,
        useBurst = 1, useFixedBurst = 0, useWrapBurst = 0, maxBurstSize = 2,
        useQoS = 0, useLock = 0, useProt = 0, useCache = 0, useRegion = 0,
        aUserWidth = 0, wUserWidth = 0, bUserWidth = 0, rUserWidth = 0,
        addrWidth = 32,
        idWidth = 10,
        useWriteResponses = 1,
      };
    };

    typedef typename axi::axi4<axiCfg> axi4_conf;

    // Slave to RV
    struct rvCfg {
      enum {
        dataWidth = 32, 
        addrWidth = 8,
        wstrbWidth = (dataWidth >> 3),
      };
    };
    typedef AxiSlaveToReadyValid<axiCfg, rvCfg> SlaveToRV;
  }
}



#endif 

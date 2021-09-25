## Tool versions and environment setup

C++ simulation and HLS of the EdgeBERT top-level hardware accelerator and children modules have been verified to work with the following tool versions:

* `gcc` - 4.9.3 (with C++11)
* `systemc` - 2.3.1
* `boost` - 1.55.0 
* `catapult` - 10.5a or newer

In the cmod/cmod_Makefile, please provide the correct tool installation paths for BOOST_HOME, SYSTEMC_HOME and CATAPULT_HOME 


## Build and run

### C++ compile and simulation of SystemC module

The following commands run C++ compilation and simulation of the EdgeBERT accelerator Top-level, executing memory storage, followed by sparse PE execution, layer normalization, softmax, entropy assessment and finally DVFS control.

    git clone --recursive https://github.com/harvard-acc/EdgeBERT.git
    cd EdgeBERT/hw/cmod/TopAccel
    make
    make run

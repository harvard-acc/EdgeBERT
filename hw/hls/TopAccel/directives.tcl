#------------------------
# TopAccel HLS script
# _________________________
#
#
set sfd [file dirname [info script]]

# Reset the options to the factory defaults
options defaults

options set /Input/CppStandard c++11
options set /Output/GenerateCycleNetlist false
solution options set /Input/CppStandard c++11
solution options set /Output/GenerateCycleNetlist false
options set /Flows/ModelSim/MSIM_DOFILE remove_waves.do
solution options set /Flows/ModelSim/MSIM_DOFILE remove_waves.do

# Start a new project to have the above change take effect
project new 

flow package require /ModelSim
flow package option set /ModelSim/VOPT_ARGS {}
flow package option set /ModelSim/SCCOM_OPTS {-g -x c++ -Wall -Wno-unused-label -Wno-unknown-pragmas -O3}
flow package require /SCVerify
flow package option set /SCVerify/USE_CCS_BLOCK true
flow package option set /SCVerify/TB_STACKSIZE 864000000
flow package option set /SCVerify/INVOKE_ARGS [list [file join $sfd kernel.txt] [file join $sfd data.txt]]

flow package require /OSCI
flow package option set /OSCI/COMP_FLAGS {-Wall -Wno-unknown-pragmas -Wno-unused-label -O3}

solution file add [file join $sfd TopAccel.h] -type CHEADER
solution file add [file join $sfd testbench.cpp] -type C++ -exclude true

directive set -PIPELINE_RAMP_UP true
go new 

# Verilog/VHDL
solution options set Output OutputVerilog true
solution options set Output/OutputVHDL false
# Reset FFs
solution options set Architectural/DefaultResetClearsAllRegs yes

# General constrains. Please refer to tool ref manual for detailed descriptions.
directive set -DESIGN_GOAL latency
directive set -SPECULATE true
directive set -MERGEABLE true
# originally 256, 32
directive set -REGISTER_THRESHOLD 2048
directive set -MEM_MAP_THRESHOLD 2048
directive set -FSM_ENCODING binary
directive set -REG_MAX_FANOUT 0
directive set -NO_X_ASSIGNMENTS true
directive set -SAFE_FSM false
directive set -REGISTER_SHARING_LIMIT 0
directive set -ASSIGN_OVERHEAD 0
directive set -TIMING_CHECKS true
directive set -MUXPATH true
directive set -REALLOC true
directive set -UNROLL no
directive set -IO_MODE super
directive set -REGISTER_IDLE_SIGNAL false
directive set -IDLE_SIGNAL {}
directive set -TRANSACTION_DONE_SIGNAL true
directive set -DONE_FLAG {}
directive set -START_FLAG {}
directive set -BLOCK_SYNC none
directive set -TRANSACTION_SYNC ready
directive set -DATA_SYNC none
directive set -RESET_CLEARS_ALL_REGS yes
directive set -CLOCK_OVERHEAD 20.000000
directive set -OPT_CONST_MULTS use_library
directive set -CHARACTERIZE_ROM false
directive set -PROTOTYPE_ROM true
directive set -ROM_THRESHOLD 64
directive set -CLUSTER_ADDTREE_IN_WIDTH_THRESHOLD 0
directive set -CLUSTER_OPT_CONSTANT_INPUTS true
directive set -CLUSTER_RTL_SYN false
directive set -CLUSTER_FAST_MODE false
directive set -CLUSTER_TYPE combinational
directive set -COMPGRADE fast
directive set -PIPELINE_RAMP_UP true
directive set -REGISTER_THRESHOLD 256
directive set -MEM_MAP_THRESHOLD 256
#directive set -BUILTIN_MODULARIO false
#directive set -EFFORT_LEVEL {high}

# Enable clock Gating
directive set -GATE_REGISTERS {true}
directive set -GATE_EFFORT {high}
directive set -GATE_MIN_WIDTH {4}
directive set -GATE_EXPAND_MIN_WIDTH {4}

# Don't branch a new solution during synthesis when source code is changed.
options set General/BranchOnChange {Never branch}
solution options set ComponentLibs/SearchPath [exec readlink -f ../DecMemCore/Catapult/] -append
solution library add "\[Block\] DecMemCore.v1"
go analyze
options set /ComponentLibs/TechLibSearchPath /<path_to_catapult>/2021.1-950854/Mgc_home/pkgs/ccs_xilinx/ -append
options set /ComponentLibs/TechLibSearchPath /<path_to_catapult>/2021.1-950854/Mgc_home/pkgs/ccs_xilinx/ -append
solution library add mgc_Xilinx-VIRTEX-7-1_beh -file {/<path_to_catapult>/2021.1-950854/Mgc_home/pkgs/ccs_xilinx/mgc_Xilinx-VIRTEX-7-1_beh.lib} -- -rtlsyntool Vivado -vendor Xilinx -technology VIRTEX-7
solution library add Xilinx_RAMS
setup_libs
setup_clock 2.0
setup_hier {TopAccel}
go compile
directive set /TopAcce/DecMemCore -MAP_TO_MODULE "\[Block\] DecMemCore.v1"
go libraries
go assembly
go architect
go allocate
go schedule
go dpfsm
go extract
project save


proc setup_clocks {period} {
   set name clk
   set CLK_PERIODby2 [expr $period/2]
   directive set -CLOCKS "$name \"-CLOCK_PERIOD $period -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME $CLK_PERIODby2 -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND sync -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high\"    "
   directive set -CLOCK_NAME $name
}

proc setup_hier {TOP_NAME} {
     directive set -DESIGN_HIERARCHY "$TOP_NAME"
}


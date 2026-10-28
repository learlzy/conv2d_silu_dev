OPP_CUSTOM_PATH ?= ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api
CATLASS_INCLUDE   ?= ../catlass/include
CATLASS_EX_COMMON ?= ../catlass/examples/common


test_performance_conv2d_silu_aclnn: test_performance_conv2d_silu_aclnn.cpp
	g++ -std=c++17 test_performance_conv2d_silu_aclnn.cpp \
		-I$(CATLASS_INCLUDE) \
		-I$(CATLASS_EX_COMMON) \
	    -I$(ASCEND_HOME_PATH)/include \
	    -I$(OPP_CUSTOM_PATH)/include \
	    -L$(ASCEND_HOME_PATH)/lib64 \
	    -L$(OPP_CUSTOM_PATH)/lib \
	    -lascendcl -lnnopbase -lopapi -lcust_opapi \
	    -o test_performance_conv2d_silu_aclnn
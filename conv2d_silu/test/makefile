OPP_CUSTOM_PATH ?= ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api

test_conv2d_silu_aclnn: test_conv2d_silu_aclnn.cpp
	g++ -std=c++17 test_conv2d_silu_aclnn.cpp \
	    -I$(ASCEND_HOME_PATH)/include \
	    -I$(OPP_CUSTOM_PATH)/include \
	    -L$(ASCEND_HOME_PATH)/lib64 \
	    -L$(OPP_CUSTOM_PATH)/lib \
	    -lascendcl -lnnopbase -lopapi -lcust_opapi \
	    -o test_conv2d_silu_aclnn
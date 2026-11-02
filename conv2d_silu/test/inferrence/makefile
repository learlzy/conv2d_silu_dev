OPP_CUSTOM_PATH ?= ${ASCEND_HOME_PATH}/opp/vendors/customize/op_api
CATLASS_INCLUDE   ?= ../../catlass/include
CATLASS_EX_COMMON ?= ../../catlass/examples/common


infer_and_save : infer_and_save.cpp
	g++ -std=c++17 infer_and_save.cpp \
		-I$(CATLASS_INCLUDE) \
		-I$(CATLASS_EX_COMMON) \
	    -I$(ASCEND_HOME_PATH)/include \
	    -I$(OPP_CUSTOM_PATH)/include \
	    -L$(ASCEND_HOME_PATH)/lib64 \
	    -L$(OPP_CUSTOM_PATH)/lib \
	    -lascendcl -lnnopbase -lopapi -lcust_opapi \
	    -o infer_and_save

# 使用msopgen步骤
- msopgen教程：
https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/devaids/optool/docs/zh/user_guide/msopgen_user_guide.md

- 生成算子工程：
###msopgen gen -i conv2d_silu.json -c ai_core-Ascend910B3 -lan cpp -out ./

- 加入catlass：
 - 在op_kernel/CMakeLists.txt加入如下代码
set(CATLASS_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/catlass/include)
npu_op_kernel_options(ascendc_kernels ALL OPTIONS -I ${CATLASS_INCLUDE_PATH})


# 编译构建步骤
- 修改conv2d_silu/CMakePresets.json
把"ASCEND_CANN_PACKAGE_PATH"的"value"改成CANN目录，可以通过echo ${ASCEND_HOME_PATH}查到CANN目录

- 编译：
./build.sh

- 安装算子：
./build_out/custom_opp*.run

- 设置环境变量：
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}

- 运行测试代码：
cd test
make
./test_conv2d_silu_aclnn


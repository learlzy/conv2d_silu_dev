1) 把conv2d_silu_dev/conv2d_silu/build_out/autogen目录下的op_proto.cc和op_proto.h复制到conv2d_silu_dev/fusion_pass_cpp/proto

2) fusion_pass_cpp/gen_es_api/CMakeLists.txt中的文件路径改成
    add_library(custom_op_proto SHARED
        ../proto/op_proto.cc)

3) 在fusion_pass_cpp/src中编写conv2d_silu_pass.cpp

4) 执行以下命令编译、安装：
mkdir build && cd build
cmake ..
make -j$(nproc)
make install

5) 删除已安装的pass：
cd build
make clean_custom_pass

6) 声明算子kernel路径：
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH} 

7) conv2d_silu.py软链接
cd ${ASCEND_HOME_PATH}/opp/vendors/customize/op_impl/ai_core/tbe/customize_impl
touch __init__.py
touch dynamic/__init__.py
ln -sf dynamic/conv2d_silu.py conv2d_silu.py

8) 验证pass是否写入了GE：
atc --model=你的模型.onnx --framework=5 \
    --soc_version=Ascend910B3 --output=./out --log=debug
message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

###########  generate eager style api using generate_es_package.cmake  ######################
set(CMAKE_CURRENT_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)

message(STATUS "Binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

# 设置输出路径
set(ES_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/es_output)
message(STATUS "ES_OUTPUT_PATH: ${ES_OUTPUT_PATH}")

# 尝试使用 find_package 方式（适用于 run 包环境）
if (EXISTS "${ASCEND_PATH}/include/ge/cmake/generate_es_package.cmake")
    list(APPEND CMAKE_MODULE_PATH "${ASCEND_PATH}/include/ge/cmake")
    find_package(GenerateEsPackage REQUIRED)
    message(STATUS "Using GenerateEsPackage from run package")
else ()
    message(FATAL_ERROR "Cannot find generate_es_package.cmake. Please ensure:\n"
            "  Run package is installed and ASCEND_PATH is set correctly")
endif ()

include_directories(
        ${ASCEND_PATH}/include
        ${ASCEND_PATH}/include/graph/
)

#生成自定义算子API
#先将proto包装成so target
add_library(custom_op_proto SHARED
        ../proto/op_proto.cc)
target_compile_definitions(custom_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(custom_op_proto PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(custom_op_proto PRIVATE
            ${CMAKE_COMPILE_COMPILER_LIBRARY}
            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
add_es_library(
        ES_LINKABLE_AND_ALL_TARGET es_custom
        OPP_PROTO_TARGET custom_op_proto
        OUTPUT_PATH ${ES_OUTPUT_PATH}
)

#生成内置算子API
add_library(opgraph_all INTERFACE)
set_target_properties(opgraph_all PROPERTIES
        INTERFACE_LIBRARY_OUTPUT_DIRECTORY "${ASCEND_PATH}/opp/built-in/op_proto"
)
add_es_library(
        ES_LINKABLE_AND_ALL_TARGET es_all
        OPP_PROTO_TARGET opgraph_all
        OUTPUT_PATH ${ES_OUTPUT_PATH}
)
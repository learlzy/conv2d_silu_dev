#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(Conv2dSilu)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(filter, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(pads, ListInt, {0,0,0,0})
    .ATTR(dilations, ListInt, {1,1,1,1})
    .ATTR(groups, Int, 1)
    .OP_END_FACTORY_REG(Conv2dSilu);

}

#endif

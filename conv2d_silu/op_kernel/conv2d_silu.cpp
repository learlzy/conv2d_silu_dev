#include "conv2d_silu.h"
#include "conv2d_silu_tiling_key.h"


template <uint32_t schMode>
__global__ __aicore__ void conv2d_silu(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Conv2dSiluTilingData);
    GET_TILING_DATA_WITH_STRUCT(Conv2dSiluTilingData, tilingData, tiling);

    if constexpr (schMode == CONV2D_SILU_SCH_MODE_FP16) {
        NsConv2dSilu::Conv2dSiluKernel<half, half> op;
        op.Init(x, filter, bias, y, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == CONV2D_SILU_SCH_MODE_FP32) {
        NsConv2dSilu::Conv2dSiluKernel<float, float> op;
        op.Init(x, filter, bias, y, workspace, &tilingData);
        op.Process();
    }
}
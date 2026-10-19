#include "conv2d_silu.h"
#include "conv2d_silu_tiling_key.h"
#include "print_info.h"
#include "lib/matmul_intf.h"


template <uint32_t schMode>
__global__ __aicore__ void conv2d_silu(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Conv2dSiluTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    GET_TILING_DATA_WITH_STRUCT(Conv2dSiluTilingData, tilingData, tiling);

    if constexpr (schMode == CONV2D_SILU_SCH_MODE_FP16) {
        // ElementAccumulator 必须为 float：L0C 累加器 half 会导致 mad intrinsic 编译失败
        NsConv2dSilu::Conv2dSiluKernel<half, float> op;
        op.Init(x, filter, bias, y, workspace, &tilingData);
        op.Process();
    } else if constexpr (schMode == CONV2D_SILU_SCH_MODE_FP32 && g_coreType != AscendC::AIC) {
        NsConv2dSilu::Conv2dSiluKernel<float, float> op;
        op.Init(x, filter, bias, y, workspace, &tilingData);
        op.Process();
    }
}
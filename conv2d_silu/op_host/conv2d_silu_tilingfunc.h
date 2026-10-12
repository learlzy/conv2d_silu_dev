#include "../op_kernel/conv2d_silu_tiling.h"
#include "../op_kernel/conv2d_silu_tiling_key.h"
#include "tiling/platform/platform_ascendc.h"
#include "conv2d_silu_utils.h"

namespace optiling {


static size_t CalcUsrWorkspaceSize(uint32_t batch, uint32_t ho, uint32_t wo,
                                   uint32_t cout1, uint32_t c0, size_t elementBytes)
{
    size_t lenWorkspace = static_cast<size_t>(batch) * cout1 * ho * wo * c0;
    size_t result = lenWorkspace * elementBytes;
    return result;
}   

static uint64_t GetTilingKeyByDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return static_cast<uint64_t>(CONV2D_SILU_SCH_MODE_FP32);
    }
    return static_cast<uint64_t>(CONV2D_SILU_SCH_MODE_FP16);
}


ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    auto* xDesc = context->GetInputDesc(0);
    ge::DataType dtype = xDesc->GetDataType();

    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* filterShape = context->GetInputShape(1);

    const auto* attrs = context->GetAttrs();

    const gert::ContinuousVector* stridesPtr = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const gert::ContinuousVector* padsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const gert::ContinuousVector* dilationsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(2);

    const int64_t* strides = reinterpret_cast<const int64_t*>(stridesPtr->GetData());
    const int64_t* pads = reinterpret_cast<const int64_t*>(padsPtr->GetData());
    const int64_t* dilations = reinterpret_cast<const int64_t*>(dilationsPtr->GetData());

    int64_t strideH = (stridesPtr->GetSize() >= 3) ? strides[2] : strides[0];
    int64_t strideW = (stridesPtr->GetSize() >= 4) ? strides[3] : strides[1];
    int64_t dilH = (dilationsPtr->GetSize() >= 3) ? dilations[2] : dilations[0];
    int64_t dilW = (dilationsPtr->GetSize() >= 4) ? dilations[3] : dilations[1];

    int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
    if (padsPtr->GetSize() >= 4) {
        padTop = pads[0];
        padBottom = pads[1];
        padLeft = pads[2];
        padRight = pads[3];
    }

    const auto& xDims = xShape->GetStorageShape();
    const auto& fDims = filterShape->GetStorageShape();

    // 输入约定 5-D {N, Cin1, H, W, C0}，filter 约定 5-D {Cin1, Kh, Kw, Cout, C0}
    uint32_t batch = static_cast<uint32_t>(xDims.GetDim(0));
    uint32_t cin1  = static_cast<uint32_t>(xDims.GetDim(1));
    uint32_t hi   = static_cast<uint32_t>(xDims.GetDim(2));
    uint32_t wi   = static_cast<uint32_t>(xDims.GetDim(3));
    uint32_t c0   = static_cast<uint32_t>(xDims.GetDim(4));
    uint32_t cout = static_cast<uint32_t>(fDims.GetDim(3));
    uint32_t kh   = static_cast<uint32_t>(fDims.GetDim(1));
    uint32_t kw   = static_cast<uint32_t>(fDims.GetDim(2));
    uint32_t cout1 = (cout + c0 - 1) / c0;
    uint32_t ho = (hi + padTop + padBottom - dilH * (kh - 1) - 1) / strideH + 1;
    uint32_t wo = (wi + padLeft + padRight - dilW * (kw - 1) - 1) / strideW + 1;

    // 元素字节数
    size_t elementBytes = (dtype == ge::DT_FLOAT16) ? sizeof(uint16_t) : sizeof(float);

    // 填充 tiling data
    Conv2dSiluTilingData* tilingData = context->GetTilingData<Conv2dSiluTilingData>();
    // Conv2dParams所需参数
    tilingData->batch = batch;
    tilingData->hi = hi;
    tilingData->wi = wi;
    tilingData->cin = cin1 * c0;
    tilingData->cout = cout;
    tilingData->kh = kh;
    tilingData->kw = kw;
    tilingData->padLeft = static_cast<uint32_t>(padLeft);
    tilingData->padRight = static_cast<uint32_t>(padRight);
    tilingData->padTop = static_cast<uint32_t>(padTop);
    tilingData->padBottom = static_cast<uint32_t>(padBottom);
    tilingData->strideH = static_cast<uint32_t>(strideH);
    tilingData->strideW = static_cast<uint32_t>(strideW);
    tilingData->dilationH = static_cast<uint32_t>(dilH);
    tilingData->dilationW = static_cast<uint32_t>(dilW);
    // 额外参数
    tilingData->cin1 = cin1;
    tilingData->cout1 = cout1;
    tilingData->c0 = c0;
    // 计算用户 workspace（手工复现 GetWorkspaceSize）
    size_t usrWorkspaceSize = CalcUsrWorkspaceSize(batch, ho, wo, cout1, c0, elementBytes);
    tilingData->workspaceSize = static_cast<uint64_t>(usrWorkspaceSize);

    // tiling key
    context->SetTilingKey(GetTilingKeyByDtype(dtype));

    // block dim：按 AIC 核数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);

    // workspace = 系统 workspace + 用户 workspace
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = sysWorkspaceSize + usrWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
}

#include "../op_kernel/conv2d_silu_tiling.h"
#include "../op_kernel/conv2d_silu_tiling_key.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {

// Ascend C0 字节数（与 Catlass BYTE_PER_C0 一致）
static constexpr uint32_t BYTE_PER_C0 = 32;

static uint64_t GetTilingKeyByDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT16) {
        return static_cast<uint64_t>(CONV2D_SILU_SCH_MODE_FP16);
    }
    return static_cast<uint64_t>(CONV2D_SILU_SCH_MODE_FP32);
}

/**
 * @brief 计算用户 workspace 大小（字节）
 *        复现 Catlass::Conv::Kernel::Conv2dEpilogue::GetWorkspaceSize
 *        workspace 存 AIC 写出的中间结果，布局为 NC1HWC0，元素类型 = 输出类型
 *
 * @param batch, ho, wo, cout1  已由 problem shape 推导
 * @param elementBytes          sizeof(ElementOutput)：FP16=2, FP32=4
 */
static size_t CalcUsrWorkspaceSize(uint32_t batch, uint32_t ho, uint32_t wo,
                                   uint32_t cout1, size_t elementBytes)
{
    // C0_WORKSPACE = BYTE_PER_C0 / sizeof(ElementWorkspace)
    const uint32_t c0Workspace = static_cast<uint32_t>(BYTE_PER_C0 / elementBytes);
    const size_t lenWorkspace =
        static_cast<size_t>(batch) * cout1 * ho * wo * c0Workspace;
    return lenWorkspace * elementBytes;
}

/**
 * @brief 根据 NCHW 输入与卷积参数推导 ho/wo/cin1/cout1/c0 等
 *        逻辑对齐 Catlass::Conv2dParams::MakeConv2dParams
 */
static void DeriveConvShape(uint32_t hi, uint32_t wi, uint32_t cin, uint32_t cout,
                            uint32_t kh, uint32_t kw,
                            uint32_t padTop, uint32_t padBottom,
                            uint32_t padLeft, uint32_t padRight,
                            uint32_t strideH, uint32_t strideW,
                            uint32_t dilH, uint32_t dilW,
                            size_t elementBytes,
                            uint32_t& ho, uint32_t& wo,
                            uint32_t& cin1, uint32_t& cout1,
                            uint32_t& coutRound, uint32_t& c0)
{
    c0 = static_cast<uint32_t>(BYTE_PER_C0 / elementBytes);
    cin1 = (cin + c0 - 1) / c0;
    cout1 = (cout + c0 - 1) / c0;
    coutRound = cout1 * c0;

    // 标准卷积输出尺寸公式
    const int64_t effectiveKh = static_cast<int64_t>(dilH) * (kh - 1) + 1;
    const int64_t effectiveKw = static_cast<int64_t>(dilW) * (kw - 1) + 1;
    ho = static_cast<uint32_t>(
        (static_cast<int64_t>(hi) + padTop + padBottom - effectiveKh) / strideH + 1);
    wo = static_cast<uint32_t>(
        (static_cast<int64_t>(wi) + padLeft + padRight - effectiveKw) / strideW + 1);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
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

    // 兼容 [N,C,H,W] 与 [H,W] 两种 attr 长度
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

    // 输入约定 NCHW，filter 约定 NCHW (cout, cin, kh, kw)
    uint32_t batch = static_cast<uint32_t>(xDims.GetDim(0));
    uint32_t cin  = static_cast<uint32_t>(xDims.GetDim(1));
    uint32_t hi   = static_cast<uint32_t>(xDims.GetDim(2));
    uint32_t wi   = static_cast<uint32_t>(xDims.GetDim(3));
    uint32_t cout = static_cast<uint32_t>(fDims.GetDim(0));
    uint32_t kh   = static_cast<uint32_t>(fDims.GetDim(2));
    uint32_t kw   = static_cast<uint32_t>(fDims.GetDim(3));

    // 元素字节数
    size_t elementBytes = (dtype == ge::DT_FLOAT16) ? sizeof(uint16_t) : sizeof(float);

    // 推导输出与 C0 相关维度（不依赖 Catlass）
    uint32_t ho = 0, wo = 0, cin1 = 0, cout1 = 0, coutRound = 0, c0 = 0;
    DeriveConvShape(hi, wi, cin, cout, kh, kw,
                    static_cast<uint32_t>(padTop), static_cast<uint32_t>(padBottom),
                    static_cast<uint32_t>(padLeft), static_cast<uint32_t>(padRight),
                    static_cast<uint32_t>(strideH), static_cast<uint32_t>(strideW),
                    static_cast<uint32_t>(dilH), static_cast<uint32_t>(dilW),
                    elementBytes,
                    ho, wo, cin1, cout1, coutRound, c0);

    // 简单合法性检查（对齐 Catlass CanImplement 的一部分）
    if (strideH == 0 || strideW == 0 || dilH == 0 || dilW == 0 ||
        ho == 0 || wo == 0) {
        return ge::GRAPH_FAILED;
    }

    // 填充 tiling data
    Conv2dSiluTilingData* tilingData = context->GetTilingData<Conv2dSiluTilingData>();

    tilingData->batch = batch;
    tilingData->hi = hi;
    tilingData->wi = wi;
    tilingData->cin = cin;
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

    tilingData->ho = ho;
    tilingData->wo = wo;
    tilingData->cin1 = cin1;
    tilingData->cout1 = cout1;
    tilingData->coutRound = coutRound;
    tilingData->c0 = c0;

    // 计算用户 workspace（手工复现 GetWorkspaceSize）
    size_t usrWorkspaceSize = CalcUsrWorkspaceSize(batch, ho, wo, cout1, elementBytes);
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


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    using optiling::BYTE_PER_C0;
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* filterShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || filterShape == nullptr || yShape == nullptr) {
        return GRAPH_FAILED;
    }

    // NC1HWC0 必须是 5 维
    if (xShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }

    // 获取 dtype，用于计算 C0（与 TilingFunc / BYTE_PER_C0 一致）
    auto* xDesc = context->GetInputDesc(0);
    if (xDesc == nullptr) {
        return GRAPH_FAILED;
    }
    ge::DataType dtype = xDesc->GetDataType();
    size_t elementBytes = (dtype == ge::DT_FLOAT16) ? sizeof(uint16_t) : sizeof(float);
    const int64_t c0 = static_cast<int64_t>(BYTE_PER_C0 / elementBytes);  // FP16->16, FP32->8

    // x: [N, Cin1, Hin, Win, C0]
    int64_t N    = xShape->GetDim(0);
    int64_t Cin1 = xShape->GetDim(1);
    int64_t Hin  = xShape->GetDim(2);
    int64_t Win  = xShape->GetDim(3);
    // xShape->GetDim(4) 应为 c0，可做校验
    if (xShape->GetDim(4) != c0) {
        return GRAPH_FAILED;
    }

    // filter 在当前仓库 Tiling 中按 4-D NCHW (Cout, Cin, Kh, Kw) 解析
    // 若实际传入的是 FRACTAL_Z 5/4 维，需按真实布局改下面几行
    if (filterShape->GetDimNum() < 4) {
        return GRAPH_FAILED;
    }
    int64_t Cout = filterShape->GetDim(0);   // 与现有 TilingFunc 保持一致
    int64_t Kh   = filterShape->GetDim(2);
    int64_t Kw   = filterShape->GetDim(3);

    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::ContinuousVector* stridesPtr   = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const gert::ContinuousVector* padsPtr      = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const gert::ContinuousVector* dilationsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    if (stridesPtr == nullptr || padsPtr == nullptr || dilationsPtr == nullptr) {
        return GRAPH_FAILED;
    }

    const int64_t* strides   = reinterpret_cast<const int64_t*>(stridesPtr->GetData());
    const int64_t* pads      = reinterpret_cast<const int64_t*>(padsPtr->GetData());
    const int64_t* dilations = reinterpret_cast<const int64_t*>(dilationsPtr->GetData());

    // 兼容 [N,C,H,W] 与 [H,W] 两种 attr 长度（与 TilingFunc 一致）
    int64_t strideH = (stridesPtr->GetSize() >= 3) ? strides[2] : strides[0];
    int64_t strideW = (stridesPtr->GetSize() >= 4) ? strides[3] : strides[1];
    int64_t dilH    = (dilationsPtr->GetSize() >= 3) ? dilations[2] : dilations[0];
    int64_t dilW    = (dilationsPtr->GetSize() >= 4) ? dilations[3] : dilations[1];

    int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
    if (padsPtr->GetSize() >= 4) {
        padTop    = pads[0];
        padBottom = pads[1];
        padLeft   = pads[2];
        padRight  = pads[3];
    }

    // 标准卷积输出尺寸
    int64_t effectiveKh = dilH * (Kh - 1) + 1;
    int64_t effectiveKw = dilW * (Kw - 1) + 1;
    int64_t Hout = (Hin + padTop + padBottom - effectiveKh) / strideH + 1;
    int64_t Wout = (Win + padLeft + padRight - effectiveKw) / strideW + 1;

    if (strideH <= 0 || strideW <= 0 || dilH <= 0 || dilW <= 0 ||
        Hout <= 0 || Wout <= 0) {
        return GRAPH_FAILED;
    }

    // 输出通道对齐到 C0
    int64_t Cout1 = (Cout + c0 - 1) / c0;

    // y: [N, Cout1, Hout, Wout, C0]
    yShape->SetDimNum(5);
    yShape->SetDim(0, N);
    yShape->SetDim(1, Cout1);
    yShape->SetDim(2, Hout);
    yShape->SetDim(3, Wout);
    yShape->SetDim(4, c0);

    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Conv2dSilu : public OpDef {
public:
    explicit Conv2dSilu(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("strides").ListInt();
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0});
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dSilu);
}

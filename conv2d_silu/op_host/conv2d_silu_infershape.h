#include "conv2d_silu_utils.h"



namespace ge {
ge::graphStatus InferShape(gert::InferShapeContext* context)
{
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
    
    // x: [N, Cin1, Hin, Win, C0]
    int64_t N    = xShape->GetDim(0);
    int64_t Cin1 = xShape->GetDim(1);
    int64_t Hin  = xShape->GetDim(2);
    int64_t Win  = xShape->GetDim(3);
    int64_t C0   = xShape->GetDim(4);

    // filter 在当前仓库 Tiling 中按 5-D {Cin1, Kh, Kw, Cout, C0} 解析
    if (filterShape->GetDimNum() < 5) {
        return GRAPH_FAILED;
    }
    int64_t Cout = filterShape->GetDim(3);
    int64_t Kh   = filterShape->GetDim(1);
    int64_t Kw   = filterShape->GetDim(2);

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
    
    // 计算输出尺寸 Hout, Wout
    int64_t Hout = (Hin + padTop + padBottom - dilH * (Kh - 1) - 1) / strideH + 1;
    int64_t Wout = (Win + padLeft + padRight - dilW * (Kw - 1) - 1) / strideW + 1;

    if (strideH <= 0 || strideW <= 0 || dilH <= 0 || dilW <= 0 ||
        Hout <= 0 || Wout <= 0) {
        return GRAPH_FAILED;
    }

    // 输出通道对齐到 C0
    int64_t Cout1 = (Cout + C0 - 1) / C0;

    // y: [N, Cout1, Hout, Wout, C0]
    yShape->SetDimNum(5);
    yShape->SetDim(0, N);
    yShape->SetDim(1, Cout1);
    yShape->SetDim(2, Hout);
    yShape->SetDim(3, Wout);
    yShape->SetDim(4, C0);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

}
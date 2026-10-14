/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv2d_silu.h
 * \brief Catlass-based Conv2d + Bias + SiLU kernel (Atlas A2)
 *        Support Element type: half / float
 *        直接调用 Conv2dEpilogue，不使用 DeviceConv Adapter，不处理 hardwareSyncAddr
 */

#ifndef __CONV2D_SILU_H__
#define __CONV2D_SILU_H__


#include "conv2d_silu_tiling.h"
#include "conv2d_silu_kernel_template.h"


namespace NsConv2dSilu {

using namespace AscendC;
using namespace Catlass;
using namespace Catlass::Conv::Kernel;

template <typename ElementType, typename ElementAccumulator>
class Conv2dSiluKernel {
public:
    using Traits = Conv2dSiluKernelTraits<ElementType, ElementAccumulator>;
    using Conv2dKernel = typename Traits::Conv2dKernel;
    using Arguments = typename Conv2dKernel::Arguments;
    using Params = typename Conv2dKernel::Params;
    using BlockEpilogue = typename Conv2dKernel::BlockEpilogue;
    static constexpr uint16_t C0_WORKSPACE = BYTE_PER_C0 / sizeof(Traits::ElementOutput);

    __aicore__ inline Conv2dSiluKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
        GM_ADDR workspace, const Conv2dSiluTilingData* tiling)
    {
        // 用户自己的中间结果区域（AIC 写 / AIV 读）
        userWorkspace_ = GetUserWorkspace(workspace);

        tilingData_ = tiling;

        // 构造 Conv2dParams
        problemParams_ = MakeProblemParams(tilingData_);

        // 构造 Arguments
        args_.problemShape = problemParams_;
        args_.ptrFmap = x;
        args_.ptrFilter = filter;
        args_.ptrBias = bias;
        args_.ptrOutput = y;

        // 转成底层 Params（内部会填充 layout 和 epilogueParams）
        params_ = ToUnderlyingArguments(args_, userWorkspace_); // 这行代码报错！
    }

    __aicore__ inline void Process()
    {
        // 直接实例化并调用 Conv2dEpilogue
        Conv2dKernel kernel;
        kernel(params_);
    }

private:
    __aicore__ inline Catlass::Conv2dParams MakeProblemParams(const Conv2dSiluTilingData* t)
    {
        uint32_t dataSizes[5] = {t->batch, t->hi, t->wi, t->cin, t->cout};
        uint8_t filterSizes[2] = {
            static_cast<uint8_t>(t->kh),
            static_cast<uint8_t>(t->kw)
        };
        uint8_t pads[4] = {
            static_cast<uint8_t>(t->padLeft),
            static_cast<uint8_t>(t->padRight),
            static_cast<uint8_t>(t->padTop),
            static_cast<uint8_t>(t->padBottom)
        };
        uint8_t strides[2] = {
            static_cast<uint8_t>(t->strideH),
            static_cast<uint8_t>(t->strideW)
        };
        uint8_t dilations[2] = {
            static_cast<uint8_t>(t->dilationH),
            static_cast<uint8_t>(t->dilationW)
        };
        return Catlass::Conv2dParams::MakeConv2dParams(dataSizes, filterSizes, pads, strides, dilations);
    }

    __aicore__ inline Params ToUnderlyingArguments(const Arguments& args, GM_ADDR workspace)
    {
        auto layoutFmap = tla::MakeLayoutFmap<Traits::ElementFmap>(
            args.problemShape.batch(), args.problemShape.cin1(), args.problemShape.hi(), args.problemShape.wi());
        auto layoutFilter = tla::MakeLayoutFilter<Traits::ElementFilter, Arch::PositionGM>(
            args.problemShape.cin1(), args.problemShape.kh(), args.problemShape.kw(), args.problemShape.cout());
        auto layoutOutput = tla::MakeLayoutFmap<Traits::ElementOutput>(
            args.problemShape.batch(), args.problemShape.cout1(), args.problemShape.ho(), args.problemShape.wo());

        layout::NC1HWC0 layoutWorkspaceEpilogue{
            args.problemShape.batch(), args.problemShape.cout1(), args.problemShape.ho(), args.problemShape.wo(),
            C0_WORKSPACE};

        typename BlockEpilogue::Params epilogueParams(
            workspace, layoutWorkspaceEpilogue, args.ptrBias, args.problemShape.cout(), args.ptrOutput,
            layoutWorkspaceEpilogue);

        Params params{args.problemShape, args.ptrFmap,       layoutFmap,     args.ptrFilter, layoutFilter,
                      args.ptrBias, workspace, args.ptrOutput, layoutOutput,   epilogueParams};
        return params;
    }

private:
    const Conv2dSiluTilingData* tilingData_{nullptr};
    GM_ADDR userWorkspace_{nullptr};
    Catlass::Conv2dParams problemParams_{};
    Arguments args_{};
    Params params_{};
};

} // namespace NsConv2dSilu

#endif // __CONV2D_SILU_H__
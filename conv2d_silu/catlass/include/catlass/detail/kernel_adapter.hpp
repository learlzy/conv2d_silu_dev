/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_DETAIL_KERNEL_ADAPTER_HPP
#define CATLASS_DETAIL_KERNEL_ADAPTER_HPP

#include "catlass/catlass.hpp"

namespace Catlass {
/// Generic Catlass kernel template
template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params)
{
    Operator op;
    op(params);
}

template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, uint64_t hardwareSyncAddr)
{
    AscendC::SetSyncBaseAddr(hardwareSyncAddr);
    Operator op;
    op(params);
}

/// Mix kernel adapter for Ascend950 (3510) platform with 1 AIC : 1 AIV ratio
/// Used when cross-core sync requires 1:1 mapping (e.g., Conv2d+SiLU epilogue)
template <class Operator>
CATLASS_GLOBAL __mix__(1, 1) void MixKernelAdapter(typename Operator::Params params)
{
    Operator op;
    op(params);
}
} // namespace Catlass
#endif

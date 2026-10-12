/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */


#ifndef CONV2D_SILU_TILING_H
#define CONV2D_SILU_TILING_H
#include <cstdint>

struct Conv2dSiluTilingData {
    // Conv2dParams所需参数
    uint32_t batch;
    uint32_t hi;
    uint32_t wi;
    uint32_t cin;
    uint32_t cout;
    uint32_t kh;
    uint32_t kw;
    uint32_t padLeft;
    uint32_t padRight;
    uint32_t padTop;
    uint32_t padBottom;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t dilationH;
    uint32_t dilationW;
    // 额外参数
    uint32_t cin1;
    uint32_t cout1;
    uint32_t c0;
    // workspace size (user workspace, bytes)
    uint64_t workspaceSize;
};

#endif // CONV2D_SILU_TILING_H

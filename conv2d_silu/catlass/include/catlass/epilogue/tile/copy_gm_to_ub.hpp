/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it
 * and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the
 * "License").
 * Please refer to the License for details. You may not use this file except in compliance with the
 * License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_COPY_GM_TO_UB_HPP
#define CATLASS_EPILOGUE_TILE_TILE_COPY_GM_TO_UB_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/conv_coord.hpp"

namespace Catlass::Epilogue::Tile {

template <class ArchTag, class GmType>
struct CopyGm2Ub {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub, can not find the specialization.");
};

template <typename Element>
struct CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyGm2Ub() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        layout::RowMajor const& layoutDst, layout::RowMajor const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutSrc.shape(0), layoutSrc.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) * sizeof(Element),
            (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK, 0);
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    };
};

template <typename Element>
struct CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::VectorLayout>> {
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyGm2Ub() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        layout::VectorLayout const& layoutDst, layout::VectorLayout const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, layoutSrc.shape(0) * sizeof(Element), 0, 0, 0);
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    };
};

/// @brief This copy instruction used to copy per token scale from GM to UB.
/// Copy the scale of shape (m,1) on GM to the first column of shape (m,n) on UB,
/// and pad the first block of each row (i.e. pad to shape (m,8) when element type is float).
/// @tparam ArchTag: Architecture tag.
/// @tparam GmType: Type of data on GM.
template <class ArchTag, class GmType>
struct CopyPerTokenScale2Ub {
    static_assert(
        std::is_same_v<typename GmType::Layout, layout::ColumnMajor>, "Unsupported layout for CopyPerTokenScale2Ub.");

    using Element = typename GmType::Element;
    using LayoutSrc = typename GmType::Layout;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyPerTokenScale2Ub() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        LayoutDst const& layoutDst, LayoutSrc const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams;
        AscendC::DataCopyPadExtParams<Element> padParams;

        dataCopyParams.blockCount = layoutSrc.shape(0);
        dataCopyParams.blockLen = layoutSrc.shape(1) * sizeof(Element); // per token scale has only one column
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;
        // Pad the data to the complete block
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;

        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    }
};

template <class ArchTag, class GmType>
struct CopyGm2UbAligned {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub aligned, can not find the specialization.");
};

template <typename Element>
struct CopyGm2UbAligned<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;
    static constexpr uint32_t STRIDE_LIMIT = 65536;

    CATLASS_DEVICE
    CopyGm2UbAligned() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        layout::RowMajor const& layoutDst, layout::RowMajor const& layoutSrc)
    {
        uint32_t rows = layoutSrc.shape(0);
        uint32_t cols = layoutSrc.shape(1);
        uint32_t srcStride = (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

        if ((layoutSrc.shape(1) == layoutSrc.stride(0)) && (layoutDst.shape(1) == layoutDst.stride(0))) {
            DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride);
                DataCopy(
                    dstTensor[i * MAX_REPEAT * layoutDst.stride(0)], srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)],
                    dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], cols);
            }
        }
    };
};

//////////////////////////// CopyGm2Ub(Ascend950, No TLA) ////////////////////////////
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
// Partial specialization for CopyGm2Ub(Ascend950, No TLA), RowMajor in and RowMajor out.
template <class Element>
struct CopyGm2Ub<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyGm2Ub() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        layout::RowMajor const& layoutDst, layout::RowMajor const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutSrc.shape(0), layoutSrc.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) * sizeof(Element),
            (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK, 0);
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    }
};

template <class Element>
struct CopyGm2Ub<Arch::Ascend950, Gemm::GemmType<Element, layout::VectorLayout>> {
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    CATLASS_DEVICE
    CopyGm2Ub() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        layout::VectorLayout const& layoutDst, layout::VectorLayout const& layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(1, layoutSrc.shape(0) * sizeof(Element), 0, 0, 0);
        if constexpr (AscendC::Std::is_one_of_v<
                          Element, float8_e4m3_t, float8_e5m2_t, float8_e8m0_t, float4_e2m1x2_t, float4_e1m2x2_t>) {
            AscendC::DataCopyPadExtParams<uint8_t> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(
                dstTensor.template ReinterpretCast<uint8_t>(), srcTensor.template ReinterpretCast<uint8_t>(),
                dataCopyParams, padParams);
        } else {
            AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
        }
    };
};
#endif // CATLASS_ARCH == 3510 || __NPU_ARCH__ == 3510

template <class Element>
struct CopyGm2UbAligned<Arch::Ascend950, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;
    static constexpr uint32_t STRIDE_LIMIT = 65536;

    CATLASS_DEVICE
    CopyGm2UbAligned() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        layout::RowMajor const& layoutDst, layout::RowMajor const& layoutSrc)
    {
        uint32_t rows = layoutSrc.shape(0);
        uint32_t cols = layoutSrc.shape(1);
        uint32_t srcStride = (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

        if ((cols == layoutSrc.stride(0)) && (layoutDst.shape(1) == layoutDst.stride(0))) {
            // continuous case
            DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            // "small" stride
            uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride);
                DataCopy(
                    dstTensor[i * MAX_REPEAT * layoutDst.stride(0)], srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)],
                    dataCopyParams);
            }
        } else {
            // long stride
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], cols);
            }
        }
    }
};

////////////////////////////////////////////////////////////
// NC1HWC0 specialized copies for Conv2d epilogue
////////////////////////////////////////////////////////////

template <typename Element>
struct CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::NC1HWC0>> {
    using LayoutSrc = layout::NC1HWC0;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t C0_SIZE = BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyGm2Ub() = default;

    // CATLASS_DEVICE
    // void operator()(
    //     AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
    //     layout::RowMajor const& layoutDst, layout::NC1HWC0 const& layoutSrc, int64_t tileBaseOffset = 0,
    //     uint32_t localStart = 0)
    // {
    //     uint32_t batch = layoutSrc.shape(0);
    //     uint32_t c1 = layoutSrc.shape(1);
    //     uint32_t h = layoutSrc.shape(2);
    //     uint32_t w = layoutSrc.shape(3);
    //     uint32_t c0 = layoutSrc.shape(4);

    //     uint32_t totalElements = batch * c1 * h * w * c0;
    //     uint32_t dstRows = layoutDst.shape(0);
    //     uint32_t dstCols = layoutDst.shape(1);
    //     uint32_t actualLen = dstRows * dstCols;

    //     if (actualLen >= totalElements) {
    //         actualLen = totalElements;
    //     }

    //     int64_t strideB = layoutSrc.stride(0);
    //     int64_t strideC1 = layoutSrc.stride(1);
    //     int64_t strideH = layoutSrc.stride(2);
    //     int64_t strideW = layoutSrc.stride(3);
    //     int64_t strideC0 = layoutSrc.stride(4);

    //     int64_t localFlat = localStart;
    //     uint32_t remainingElements = actualLen;
    //     uint32_t dstIdx = 0;

    //     while (remainingElements > 0) {
    //         uint32_t chunkSize = (remainingElements > C0_SIZE) ? C0_SIZE : remainingElements;

    //         uint32_t localC0 = localFlat % c0;
    //         uint32_t remIdx = localFlat / c0;
    //         uint32_t localW = remIdx % w;
    //         remIdx /= w;
    //         uint32_t localH = remIdx % h;
    //         remIdx /= h;
    //         uint32_t localC1 = remIdx % c1;
    //         uint32_t localB = remIdx / c1;

    //         int64_t gmOffset = tileBaseOffset + localB * strideB + localC1 * strideC1 + localH * strideH +
    //                            localW * strideW + localC0 * strideC0;

    //         AscendC::DataCopyExtParams copyParams(1, chunkSize * sizeof(Element), 0, 0, 0);
    //         AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
    //         AscendC::DataCopyPad(dstTensor[dstIdx], srcTensor[gmOffset], copyParams, padParams);

    //         localFlat += chunkSize;
    //         dstIdx += chunkSize;
    //         remainingElements -= chunkSize;
    //     }
    // };

    // CATLASS_DEVICE
    // void operator()(
    //     AscendC::LocalTensor<Element> const& dstTensor,  // UB 中的目标 buffer（ubC），我们要把数据连续写进去
    //     AscendC::GlobalTensor<Element> const& srcTensor,  // GM 上的 workspace
    //     layout::RowMajor const& layoutDst,  // 描述 UB 侧形状，通常是 RowMajor(ubRows, c0)
    //     layout::NC1HWC0 const& layoutSrc, // shape 是当前 block 的局部大小，stride 是全局真实 stride
    //     int64_t tileBaseOffset = 0,  // 当前 block 第一个元素在 GM 中的绝对偏移
    //     uint32_t localStart = 0   // 本次要搬的数据在 block 内的线性起始下标
    // ) {
    //     using namespace AscendC;

    //     uint32_t batch = layoutSrc.shape(0);  // 当前 block 的 batch（通常 1）
    //     uint32_t c1    = layoutSrc.shape(1);  // 当前 block 的 Cout1
    //     uint32_t h     = layoutSrc.shape(2);  // 当前 block 的实际 Ho
    //     uint32_t w     = layoutSrc.shape(3);  // 当前 block 的实际 Wo
    //     uint32_t c0    = layoutSrc.shape(4);  // 通常是 16

    //     uint32_t totalElements = batch * c1 * h * w * c0; // 整个 block 的总元素数
    //     uint32_t dstRows = layoutDst.shape(0);  // UB 行数 = CeilDiv(actualLen, c0)
    //     uint32_t dstCols = layoutDst.shape(1);  // 通常 = c0
    //     uint32_t actualLen = dstRows * dstCols; // 本次实际要搬的元素个数
    //     if (actualLen > totalElements) {
    //         actualLen = totalElements;
    //     }

    //     int64_t strideB  = layoutSrc.stride(0);  // batch 方向的全局步长
    //     int64_t strideC1 = layoutSrc.stride(1);  // C1 方向的全局步长
    //     int64_t strideH  = layoutSrc.stride(2);  // H 方向的全局(GM)步长
    //     int64_t strideW  = layoutSrc.stride(3);  // W 方向的全局(GM)步长
    //     int64_t strideC0 = layoutSrc.stride(4);  // 通常是 1

    //     uint32_t elementsLeft = actualLen;
    //     uint32_t dstIdx       = 0;              // 当前数据起点在 UB 物理空间 中的写入位置（连续递增）
    //     uint32_t localFlat    = localStart;     // 当前数据起点在 block 逻辑空间 中的线性编号（用来还原坐标、计算 GM 地址）

    //     while (elementsLeft > 0) {
    //         // 1. 当前数据起点在block内的线性下标 → 当前数据起点在block内的 NC1HWC0 坐标
    //         uint32_t localC0 = localFlat % c0;
    //         uint32_t rem     = localFlat / c0;
    //         uint32_t localW  = rem % w;   rem /= w;
    //         uint32_t localH  = rem % h;   rem /= h;
    //         uint32_t localC1 = rem % c1;
    //         uint32_t localB  = rem / c1;

    //         // 2. 计算当前数据起点在 GM 的地址
    //         int64_t gmOffset = tileBaseOffset
    //                          + localB  * strideB
    //                          + localC1 * strideC1
    //                          + localH  * strideH
    //                          + localW  * strideW
    //                          + localC0 * strideC0;

    //         // 3. 处理非 C0 对齐的起始点（只可能出现在第一个 chunk）
    //         if (localC0 != 0) {
    //             uint32_t chunk = min(c0 - localC0, elementsLeft);
    //             DataCopyExtParams p{1, static_cast<uint32_t>(chunk * sizeof(Element)), 0, 0, 0};
    //             DataCopyPadExtParams<Element> pad{false, 0, 0, 0};
    //             DataCopyPad(dstTensor[dstIdx], srcTensor[gmOffset], p, pad);

    //             localFlat    += chunk;
    //             dstIdx       += chunk;
    //             elementsLeft -= chunk;
    //             continue;
    //         }

    //         // 4. 计算同一行内还能连续搬多少个完整的 C0
    //         uint32_t maxContiguousW = w - localW;               // 当前 H 行内剩余的 W 数
    //         uint32_t maxBlocksByLen = elementsLeft / c0;        // 剩余元素能支撑的块数
    //         uint32_t blocks = min(maxContiguousW, maxBlocksByLen);  // 本次实际能一次搬的块数（取两者较小值，一块就是 c0 个元素）

    //         if (blocks == 0) {
    //             // 理论上不会走到这里，保护一下
    //             break;
    //         }

    //         // 5. 一次搬 blocks 个连续的 C0（使用 srcStride）
    //         DataCopyExtParams p;
    //         p.blockCount = static_cast<uint16_t>(blocks);
    //         p.blockLen   = static_cast<uint32_t>(c0 * sizeof(Element));
    //         // srcStride：前一个块的尾 → 后一个块的头 的间隔（字节）
    //         // 相邻 W 的间隔是 strideW，每个块本身占 c0 个元素，所以间隔是 (strideW - c0)
    //         p.srcStride  = static_cast<uint32_t>((strideW - c0) * sizeof(Element));
    //         p.dstStride  = 0;   // UB 侧连续存放
    //         p.rsv        = 0;

    //         DataCopyPadExtParams<Element> pad{false, 0, 0, 0};
    //         DataCopyPad(dstTensor[dstIdx], srcTensor[gmOffset], p, pad);

    //         uint32_t moved = blocks * c0;
    //         localFlat    += moved;
    //         dstIdx       += moved;
    //         elementsLeft -= moved;
    //     }
    // }

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, // UB 中的目标 buffer（ubC），我们要把数据连续写进去
        AscendC::GlobalTensor<Element> const& srcTensor,  // GM 上的 workspace
        layout::RowMajor const& layoutDst,  // 描述 UB 侧形状，通常是 RowMajor(ubRows, c0)
        layout::NC1HWC0 const& layoutSrc,  // shape 是当前 block 的局部大小，stride 是全局真实 stride
        int64_t tileBaseOffset = 0,  // 当前 block 第一个元素在 GM 中的绝对偏移
        uint32_t localStart = 0   // 本次要搬的数据在 block 内的线性起始下标
    ) {
        using namespace AscendC;

        uint32_t batch = layoutSrc.shape(0);  // 当前 block 的 batch（通常 1）
        uint32_t c1    = layoutSrc.shape(1);  // 当前 block 的 Cout1
        uint32_t h     = layoutSrc.shape(2);  // 当前 block 的实际 Ho
        uint32_t w     = layoutSrc.shape(3);  // 当前 block 的实际 Wo
        uint32_t c0    = layoutSrc.shape(4);  // 通常是 16

        uint32_t totalElements = batch * c1 * h * w * c0;  // 整个 block 的总元素数
        uint32_t dstRows = layoutDst.shape(0);     // UB 行数 = CeilDiv(actualLen, c0)
        uint32_t dstCols = layoutDst.shape(1);     // 通常 = c0
        uint32_t actualLen = dstRows * dstCols;    // 本次实际要搬的元素个数
        if (actualLen > totalElements) {
            actualLen = totalElements;
        }

        int64_t strideB  = layoutSrc.stride(0);   // batch 方向的全局步长
        int64_t strideC1 = layoutSrc.stride(1);   // C1 方向的全局步长
        int64_t strideH  = layoutSrc.stride(2);   // H 方向的全局(GM)步长
        int64_t strideW  = layoutSrc.stride(3);   // W 方向的全局(GM)步长
        int64_t strideC0 = layoutSrc.stride(4);   // 通常是 1

        uint32_t elementsLeft = actualLen;
        uint32_t dstIdx       = 0;           // 当前数据起点在 UB 物理空间 中的写入位置（连续递增）
        uint32_t localFlat    = localStart;  // 当前数据起点在 block 逻辑空间 中的线性编号（用来还原坐标、计算 GM 地址）

        // 新增分支：actualLen 是完整行（w*c0）的整数倍时，按行搬运
        uint32_t rowElements = w * c0;   // 一行的元素个数
        bool rowFlag = rowElements > 0 && 
            (actualLen % rowElements == 0) && 
            (localStart % rowElements == 0);
        if (rowFlag) {
            // 满足条件：从完整行边界开始，且长度是整行的倍数
            // 此时可以按「行」为粒度，用 strideH 一次搬多行
            while (elementsLeft > 0) {
                // 1. 当前数据起点在block内的线性下标 → 当前数据起点在block内的 NC1HWC0 坐
                uint32_t localC0 = localFlat % c0;
                uint32_t rem     = localFlat / c0;
                uint32_t localW  = rem % w;   rem /= w;
                uint32_t localH  = rem % h;   rem /= h;
                uint32_t localC1 = rem % c1;
                uint32_t localB  = rem / c1;
                // 2. 计算当前数据起点在 GM 的地址
                int64_t gmOffset = tileBaseOffset
                                 + localB  * strideB
                                 + localC1 * strideC1
                                 + localH  * strideH
                                 + localW  * strideW
                                 + localC0 * strideC0;

                // 计算同一 C1 内还能连续搬多少完整行
                uint32_t maxContiguousH = h - localH;
                uint32_t maxRowsByLen   = elementsLeft / rowElements;
                uint32_t rows = min(maxContiguousH, maxRowsByLen);

                if (rows == 0) break;

                // 一次搬 rows 行
                DataCopyExtParams p;
                p.blockCount = static_cast<uint16_t>(rows);
                p.blockLen   = static_cast<uint32_t>(rowElements * sizeof(Element));
                p.srcStride  = static_cast<uint32_t>((strideH - rowElements) * sizeof(Element));
                p.dstStride  = 0;
                p.rsv        = 0;

                DataCopyPadExtParams<Element> pad{false, 0, 0, 0};
                DataCopyPad(dstTensor[dstIdx], srcTensor[gmOffset], p, pad);

                uint32_t moved = rows * rowElements;
                localFlat    += moved;
                dstIdx       += moved;
                elementsLeft -= moved;
            }
        } else { // 原有通用分支：按 C0 粒度
            while (elementsLeft > 0) {
                uint32_t chunkSize = (elementsLeft > C0_SIZE) ? C0_SIZE : elementsLeft;

                uint32_t localC0 = localFlat % c0;
                uint32_t remIdx = localFlat / c0;
                uint32_t localW = remIdx % w;
                remIdx /= w;
                uint32_t localH = remIdx % h;
                remIdx /= h;
                uint32_t localC1 = remIdx % c1;
                uint32_t localB = remIdx / c1;

                int64_t gmOffset = tileBaseOffset + localB * strideB + localC1 * strideC1 + localH * strideH +
                                   localW * strideW + localC0 * strideC0;

                AscendC::DataCopyExtParams copyParams(1, chunkSize * sizeof(Element), 0, 0, 0);
                AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
                AscendC::DataCopyPad(dstTensor[dstIdx], srcTensor[gmOffset], copyParams, padParams);

                localFlat += chunkSize;
                dstIdx += chunkSize;
                elementsLeft -= chunkSize;
            }
        }
    }
};

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
template <typename Element>
struct CopyGm2Ub<Arch::Ascend950, Gemm::GemmType<Element, layout::NC1HWC0>> {
    using LayoutSrc = layout::NC1HWC0;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t C0_SIZE = BYTE_PER_C0 / sizeof(Element);

    CATLASS_DEVICE
    CopyGm2Ub() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const& dstTensor, AscendC::GlobalTensor<Element> const& srcTensor,
        layout::RowMajor const& layoutDst, layout::NC1HWC0 const& layoutSrc, int64_t tileBaseOffset = 0,
        uint32_t localStart = 0)
    {
        uint32_t batch = layoutSrc.shape(0);
        uint32_t c1 = layoutSrc.shape(1);
        uint32_t h = layoutSrc.shape(2);
        uint32_t w = layoutSrc.shape(3);
        uint32_t c0 = layoutSrc.shape(4);

        uint32_t totalElements = batch * c1 * h * w * c0;
        uint32_t dstRows = layoutDst.shape(0);
        uint32_t dstCols = layoutDst.shape(1);
        uint32_t actualLen = dstRows * dstCols;

        if (actualLen >= totalElements) {
            actualLen = totalElements;
        }

        // tileBaseOffset is the absolute GM offset for the tile's local index 0
        // localStart is the starting local index within the tile
        int64_t strideB = layoutSrc.stride(0);
        int64_t strideC1 = layoutSrc.stride(1);
        int64_t strideH = layoutSrc.stride(2);
        int64_t strideW = layoutSrc.stride(3);
        int64_t strideC0 = layoutSrc.stride(4);

        int64_t localFlat = localStart;
        uint32_t remainingElements = actualLen;
        uint32_t dstIdx = 0;

        while (remainingElements > 0) {
            uint32_t chunkSize = (remainingElements > C0_SIZE) ? C0_SIZE : remainingElements;

            // Convert localFlat to tile-local NC1HWC0 coordinate
            uint32_t localC0 = localFlat % c0;
            uint32_t remIdx = localFlat / c0;
            uint32_t localW = remIdx % w;
            remIdx /= w;
            uint32_t localH = remIdx % h;
            remIdx /= h;
            uint32_t localC1 = remIdx % c1;
            uint32_t localB = remIdx / c1;

            // Compute absolute GM offset: tileBaseOffset + local coord * global strides
            int64_t gmOffset = tileBaseOffset + localB * strideB + localC1 * strideC1 + localH * strideH +
                               localW * strideW + localC0 * strideC0;

            AscendC::DataCopyExtParams copyParams(1, chunkSize * sizeof(Element), 0, 0, 0);
            AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
            AscendC::DataCopyPad(dstTensor[dstIdx], srcTensor[gmOffset], copyParams, padParams);

            localFlat += chunkSize;
            dstIdx += chunkSize;
            remainingElements -= chunkSize;
        }
    };
};
#endif
} // namespace Catlass::Epilogue::Tile

#endif

/**************************************************************************************************
 * block_conv2d_shared_filter_bias.hpp
 *
 * 基于 block_conv2d_pingpong_tla.hpp / block_mmad_pingpong_bias.hpp 改写
 * 针对 Ascend 910B (Atlas A2)
 *
 * 关键点：
 *   1. filter 全量加载并长驻 L0B
 *   2. bias 长驻 BiasTable (btBuf)，通过 TileMmad 的 bias overload 初始化 L0C
 *   3. L1 只按 Cin1 切分
 *   4. L0 K = Cin1_tile * kh * kw（可再按 L0TileShape::K 切 kPart）
 *   5. 每次 L0 Mad 按 kPart 从长驻 L0B 中取对应 K 切片
 *************************************************************************************************/
#ifndef CATLASS_CONV_BLOCK_BLOCK_CONV2D_SHARED_FILTER_BIAS_HPP
#define CATLASS_CONV_BLOCK_BLOCK_CONV2D_SHARED_FILTER_BIAS_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"
#include "catlass/conv/tile/tile_copy.hpp"

namespace Catlass::Conv::Block {

template <
    class ArchTag_,
    uint32_t L1A_STAGES_,
    uint32_t L0A_STAGES_,
    uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_,
    class FmapL1TileShape_,   // (Ho, Wo, Cin1Tile)
    class L0TileShape_,       // (M, N, K)  K 语义 = Cin1_tile * kh * kw 的分组数
    class ElementFmap_,
    class ElementFilter_,
    class ElementOutput_,
    class ElementAccumulator_,
    class ElementBias_,
    class TileCopy_,
    class TileMmad_
>
class BlockConv2dSharedFilterBias {
public:
    using ArchTag = ArchTag_;
    using FmapL1TileShape = FmapL1TileShape_;
    using L0TileShape = L0TileShape_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;

    using ElementFmap = ElementFmap_;
    using ElementFilter = ElementFilter_;
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementBias = ElementBias_;

    using LayoutFmap = typename TileCopy::LayoutFmap;
    using LayoutFilter = typename TileCopy::LayoutFilter;
    using LayoutOutput = typename TileCopy::LayoutOutput;

    using CopyGmToL1A = typename TileCopy::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy::CopyL0CToGm;
    // Bias: GM→L1 再 L1→BT（与 block_mmad_pingpong_bias 一致）
    using CopyGmToL1Bias = typename TileCopy::CopyGmToL1Bias;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;

    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;

    static constexpr uint32_t HO_BLOCK  = FmapL1TileShape::Ho;
    static constexpr uint32_t WO_BLOCK  = FmapL1TileShape::Wo;
    static constexpr uint32_t CIN1_TILE = FmapL1TileShape::Cin1;

    static constexpr uint32_t L0_TILE_M = L0TileShape::M;
    static constexpr uint32_t L0_TILE_N = L0TileShape::N;
    static constexpr uint32_t L0_TILE_K = L0TileShape::K;  // 单位：Cin1*kh*kw 方向上的 group 数

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementFmap);

    CATLASS_DEVICE
    BlockConv2dSharedFilterBias(
        Arch::Resource<ArchTag>& resource,
        const Conv2dFilterParams& filterParams_,
        uint32_t l1BufAddrStart = 0)
        : filterParams(filterParams_),
          copyL1ToL0A(filterParams_)
    {
        // 与原生一致的 hiBlock / wiBlock / l1A_size
        hiBlock =
            (FmapL1TileShape::Ho - 1) * filterParams.strideH() +
            (filterParams.kh() - 1) * filterParams.dilationH() + 1;
        wiBlock =
            (FmapL1TileShape::Wo - 1) * filterParams.strideW() +
            (filterParams.kw() - 1) * filterParams.dilationW() + 1;
        l1A_size = FmapL1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0;

        coutRound_ = RoundUp(filterParams.cout(), C0_NUM_PER_FRACTAL);
        // filter 全量字节数（长驻 L0B）
        // layout CI1KHKWCOCI0: cin1 * kh * kw * coutRound * BYTE_PER_C0
        filterFullSize_ =
            filterParams.cin1() * filterParams.kh() * filterParams.kw() *
            coutRound_ * BYTE_PER_C0;
        // 全量 K group 数 = cin1 * kh * kw
        kFullGroups_ = filterParams.cin1() * filterParams.kh() * filterParams.kw();

        biasNum_ = coutRound_;  // BiasTable 按 cout 对齐

        // ---------- L1A multi-stage ----------
        uint32_t l1AOffset = l1BufAddrStart;
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFmap>(
                l1AOffset + l1A_size * i);
            l1AEventList[i] = static_cast<int32_t>(i);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }

        // L1 上临时区域：放 filter / bias（仅 resident 加载时用一次）
        // 放在 L1A stages 之后
        uint32_t l1TmpOffset = l1AOffset + l1A_size * L1A_STAGES;
        l1FilterTmp_ = resource.l1Buf.template GetBufferByByte<ElementFilter>(l1TmpOffset);
        l1BiasTmp_ = resource.l1Buf.template GetBufferByByte<ElementBias>(
            l1TmpOffset + filterFullSize_);

        // ---------- L0A multi-stage ----------
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementFmap>(
                L0A_PINGPONG_BUF_SIZE * i);
            l0AEventList[i] = static_cast<int32_t>(i);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }

        // ---------- L0B 长驻（全量 filter）----------
        l0BTensor_ = resource.l0BBuf.template GetBufferByByte<ElementFilter>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);

        // ---------- L0C ----------
        l0CTensor_ = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);

        // ---------- BiasTable (btBuf) ----------
        // 与 block_mmad_pingpong_bias 一致：btBuf 上的 bias 用 ElementAccumulator 视图
        l0BiasTensor_ = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);

        // bias 独立 event
        EVENT_ID_BIAS = static_cast<int32_t>(L1A_STAGES + L0A_STAGES);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID_BIAS);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID_BIAS);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    CATLASS_DEVICE
    ~BlockConv2dSharedFilterBias()
    {
        for (uint32_t i = 0; i < L1A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID_BIAS);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID_BIAS);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    // ------------------------------------------------------------------
    // filter + bias 一次性长驻加载
    // ------------------------------------------------------------------
    CATLASS_DEVICE
    void LoadFilterAndBiasResident(
        AscendC::GlobalTensor<ElementFilter> const& gmFilter,
        AscendC::GlobalTensor<ElementBias> const& gmBias)
    {
        // ===== 1. Bias: GM → L1 → BiasTable =====
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID_BIAS);

        // GM → L1
        layout::VectorLayout layoutBiasGm(biasNum_);
        layout::VectorLayout layoutBiasL1(biasNum_);
        copyGmToL1Bias(l1BiasTmp_, gmBias, layoutBiasL1, layoutBiasGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID_BIAS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID_BIAS);

        // L1 → BT
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID_BIAS);
        layout::VectorLayout layoutBiasBT(biasNum_);
        copyL1ToBT(l0BiasTensor_, l1BiasTmp_, layoutBiasBT, layoutBiasL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID_BIAS);
        // 注意：第一次 Mad(initC=true) 时再 Wait 这个 flag

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID_BIAS);

        // ===== 2. Filter: GM → L1 → L0B（全量长驻）=====
        // 复用 L1A stage0 的 event 做一次同步（resident 只做一次，不进主循环）
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[0]);

        // GM → L1（全量 filter）
        // layout: CI1KHKWCOCI0
        auto layoutFilterL1 = layout::CI1KHKWCOCI0::template MakeLayout<ElementFilter>(
            filterParams.cin1(), filterParams.kh(), filterParams.kw(), coutRound_);
        auto layoutFilterGm = layoutFilterL1;  // GM 与 L1 同 layout
        copyGmToL1B(l1FilterTmp_, gmFilter, layoutFilterL1, layoutFilterGm);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[0]);

        // L1 → L0B（全量，nZ）
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        auto layoutFilterL0 = layout::nZ::template MakeLayout<ElementFilter>(
            kFullGroups_ * ELE_NUM_PER_C0, coutRound_);
        copyL1ToL0B(l0BTensor_, l1FilterTmp_, layoutFilterL0, layoutFilterL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

        // 归还 L1A stage0 的 MTE1_MTE2，供主循环使用
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[0]);
    }

    // ------------------------------------------------------------------
    // 主计算
    // ------------------------------------------------------------------
    template <class TensorFmap, class TensorOutput>
    CATLASS_DEVICE void operator()(
        TensorFmap& tensorFmap,
        TensorOutput& tensorOutput,
        const Conv2dCoord& actualBlockShape,
        const uint8_t blockPadList[4])
    {
        uint32_t howoActual = actualBlockShape.h() * actualBlockShape.w();
        uint32_t coutActual = actualBlockShape.cout();
        uint32_t cin1Actual = actualBlockShape.cin1();

        if (howoActual == 0 || coutActual == 0) {
            return;
        }

        uint32_t howoRound = RoundUp(howoActual, C0_NUM_PER_FRACTAL);
        uint32_t coutRound = RoundUp(coutActual, C0_NUM_PER_FRACTAL);

        // L0 上 M/N 方向循环
        uint32_t mPartLoop = CeilDiv(howoRound, L0_TILE_M);
        uint32_t nPartLoop = CeilDiv(coutRound, L0_TILE_N);

        // Cin1 方向 L1 tile 循环
        uint32_t cin1TileCnt = CeilDiv(cin1Actual, CIN1_TILE);

        uint32_t l1AListId = 0;
        uint32_t l0AListId = 0;

        for (uint32_t cin1LoopIdx = 0; cin1LoopIdx < cin1TileCnt; ++cin1LoopIdx) {
            uint32_t cin1Start = cin1LoopIdx * CIN1_TILE;
            uint32_t cin1This =
                (cin1Start + CIN1_TILE > cin1Actual) ? (cin1Actual - cin1Start) : CIN1_TILE;

            // 当前 L1 tile 对应的全量 K group 数
            uint32_t kGroupsThis = cin1This * filterParams.kh() * filterParams.kw();

            // ---------- 1. L1A 加载（只切 Cin1）----------
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);

            auto tensorL1A = l1ATensorList[l1AListId];
            copyGmToL1A(
                tensorL1A, tensorFmap,
                cin1Start, cin1This,
                actualBlockShape, blockPadList, filterParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

            // ---------- 2. L0 级循环：按 L0_TILE_K 切 K ----------
            uint32_t kPartLoop = CeilDiv(kGroupsThis, L0_TILE_K);

            for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; ++mPartIdx) {
                uint32_t mPartActual =
                    (mPartIdx + 1 < mPartLoop) ? L0_TILE_M
                                               : (howoActual - mPartIdx * L0_TILE_M);

                for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                    uint32_t kPartActual =
                        (kPartIdx + 1 < kPartLoop) ? L0_TILE_K
                                                   : (kGroupsThis - kPartIdx * L0_TILE_K);
                    // k 对 Mmad 是元素数：group * C0
                    uint32_t kPartElems = kPartActual * ELE_NUM_PER_C0;

                    // ----- L0A: im2col + 加载当前 kPart -----
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);

                    auto tensorL0A = l0ATensorList[l0AListId];
                    // CopyL1ToL0A 需要支持 kPart 偏移；
                    // 这里传入 kPart 起点与长度（与原生 copyL1ToL0A 扩展接口对齐）
                    copyL1ToL0A(
                        tensorL0A, tensorL1A,
                        mPartIdx * L0_TILE_M, mPartActual,
                        kPartIdx * L0_TILE_K, kPartActual,
                        filterParams);

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);

                    for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; ++nPartIdx) {
                        uint32_t nPartActual =
                            (nPartIdx + 1 < nPartLoop) ? L0_TILE_N
                                                       : (coutActual - nPartIdx * L0_TILE_N);

                        // ----- 从长驻 L0B 取当前 (kPart, nPart) 切片 -----
                        // L0B 全量 layout 可视为 [kFullGroups, coutRound] 的 nZ
                        // offset（元素）：kPart 起点 * coutRound * C0 + nPart 起点
                        //
                        // 全局 K 起点 = 当前 cin1 tile 在全量 filter 中的 K 偏移 + kPart 偏移
                        uint32_t kGlobalStart =
                            cin1Start * filterParams.kh() * filterParams.kw() +
                            kPartIdx * L0_TILE_K;

                        // nZ 下 K 方向一步跨过 coutRound 个 fractal 的元素数
                        // 与原生 CopyL1ToL0B 的 nZ layout 一致：
                        //   stride_k ≈ coutRound * ELE_NUM_PER_C0（具体以 layout::nZ 为准）
                        uint32_t l0BOffset =
                            kGlobalStart * coutRound_ * ELE_NUM_PER_C0 +
                            nPartIdx * L0_TILE_N * ELE_NUM_PER_C0;
                        // 注意：若 nZ 的实际 stride 不同，需用 layout::nZ::GetOffset 替换
                        auto l0BTile = l0BTensor_[l0BOffset];

                        // L0C tile
                        auto layoutInL0C = layout::zN::MakeLayoutInL0C(
                            MakeCoord(howoRound, coutRound));
                        MatrixCoord l0COffset{mPartIdx * L0_TILE_M, nPartIdx * L0_TILE_N};
                        auto l0CTile = l0CTensor_[layoutInL0C.GetOffset(l0COffset)];

                        // ----- Mad -----
                        bool initC = (cin1LoopIdx == 0) && (kPartIdx == 0);

                        uint8_t unitFlag = 0;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            bool isLast =
                                (cin1LoopIdx == cin1TileCnt - 1) &&
                                (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) &&
                                (nPartIdx == nPartLoop - 1);
                            unitFlag = isLast ? 0b11 : 0b10;
                        }

                        if (initC) {
                            // 第一次：用 BiasTable 初始化 L0C
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID_BIAS);
                            // bias 按 nPart 切
                            auto l0BiasTile = l0BiasTensor_[nPartIdx * L0_TILE_N];
                            tileMmad(
                                l0CTile, tensorL0A, l0BTile, l0BiasTile,
                                mPartActual, nPartActual, kPartElems,
                                true, unitFlag);
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID_BIAS);
                        } else {
                            tileMmad(
                                l0CTile, tensorL0A, l0BTile,
                                mPartActual, nPartActual, kPartElems,
                                false, unitFlag);
                        }
                    } // nPart

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
                } // kPart
            } // mPart

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
            l1AListId = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
        } // cin1

        // ---------- 写回 ----------
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        }
        copyL0CToDst(tensorOutput, l0CTensor_, howoActual, coutActual);
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
    }

    // ------------------------------------------------------------------
    static bool CanImplement(const Conv2dFilterParams& filterParams)
    {
        uint32_t hiBlock =
            (FmapL1TileShape::Ho - 1) * filterParams.strideH() +
            (filterParams.kh() - 1) * filterParams.dilationH() + 1;
        uint32_t wiBlock =
            (FmapL1TileShape::Wo - 1) * filterParams.strideW() +
            (filterParams.kw() - 1) * filterParams.dilationW() + 1;
        uint32_t l1A_size = FmapL1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0;

        uint32_t coutRound = RoundUp(filterParams.cout(), C0_NUM_PER_FRACTAL);
        uint32_t filterFullSize =
            filterParams.cin1() * filterParams.kh() * filterParams.kw() *
            coutRound * BYTE_PER_C0;

        // L1：L1A stages + filter临时 + bias临时
        uint32_t l1Need = l1A_size * L1A_STAGES + filterFullSize +
                          coutRound * sizeof(ElementBias);
        if (l1Need > ArchTag::L1_SIZE) {
            return false;
        }
        // L0B 需放下全量 filter
        if (filterFullSize > ArchTag::L0B_SIZE) {
            return false;
        }
        // 硬件 kPart 上限
        constexpr uint32_t MAX_KPART_HW = 512;
        if (filterParams.kh() * filterParams.kw() * ELE_NUM_PER_C0 > MAX_KPART_HW) {
            return false;
        }
        return true;
    }

protected:
    Conv2dFilterParams filterParams;
    uint32_t hiBlock{0};
    uint32_t wiBlock{0};
    uint32_t l1A_size{0};
    uint32_t filterFullSize_{0};
    uint32_t kFullGroups_{0};
    uint32_t coutRound_{0};
    uint32_t biasNum_{0};

    // multi-stage
    AscendC::LocalTensor<ElementFmap>        l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementFmap>        l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementFilter>      l0BTensor_;       // 长驻全量 filter
    AscendC::LocalTensor<ElementAccumulator> l0CTensor_;
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor_;    // BiasTable 视图

    // resident 加载用的 L1 临时区
    AscendC::LocalTensor<ElementFilter>      l1FilterTmp_;
    AscendC::LocalTensor<ElementBias>        l1BiasTmp_;

    int32_t l1AEventList[L1A_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t EVENT_ID_BIAS{0};

    TileMmad tileMmad;
    CopyGmToL1A   copyGmToL1A;
    CopyGmToL1B   copyGmToL1B;
    CopyGmToL1Bias copyGmToL1Bias;
    CopyL1ToL0A   copyL1ToL0A;
    CopyL1ToL0B   copyL1ToL0B;
    CopyL1ToBT    copyL1ToBT;
    CopyL0CToGm   copyL0CToDst;
};

} // namespace Catlass::Conv::Block

#endif // CATLASS_CONV_BLOCK_BLOCK_CONV2D_SHARED_FILTER_BIAS_HPP
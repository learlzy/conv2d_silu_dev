/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * CANN Open Software License Agreement Version 2.0
 */
#ifndef CATLASS_CONV_BLOCK_BLOCK_CONV2D_PINGPONG_BIAS_TLA_HPP
#define CATLASS_CONV_BLOCK_BLOCK_CONV2D_PINGPONG_BIAS_TLA_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Conv::Block {

/**
 * BlockConv2dTla with Cube-side Bias Add.
 *
 * Bias path: GM -> L1 -> BT (Bias Table / C2) -> Mmad(with bias)
 * Bias is applied only on the first K-tile (initC path), same as
 * block_conv3d_pingpong_bias / TileMmadTla bias overload.
 */
template <
    class ArchTag_, uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_,
    class FmapL1TileShape_,   // (Ho, Wo, Cin1_small)
    class FilterL1TileShape_, // (Cout, Cin1_big)
    class L0TileShape_,       // (M, N, K)
    class ElementFmap_, class ElementFilter_, class ElementOutput_, class ElementAccumulator_, class ElementBias_,
    class TileCopy_, class TileMmad_>
struct BlockConv2dTla<
    ConvPingpong<ArchTag_, L1A_STAGES_, L1B_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_>,
    FmapL1TileShape_, FilterL1TileShape_, L0TileShape_, ElementFmap_, ElementFilter_, ElementOutput_,
    ElementAccumulator_, ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy =
        ConvPingpong<ArchTag_, L1A_STAGES_, L1B_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using FmapL1TileShape = FmapL1TileShape_;
    using FilterL1TileShape = FilterL1TileShape_;
    using L0TileShape = L0TileShape_;

    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;

    using ElementFmap = ElementFmap_;
    using LayoutFmap = typename TileCopy::LayoutFmap;
    using ElementFilter = ElementFilter_;
    using LayoutFilter = typename TileCopy::LayoutFilter;
    using ElementOutput = ElementOutput_;
    using LayoutOutput = typename TileCopy::LayoutOutput;
    using ElementBias = ElementBias_;

    using CopyGmToL1A = typename TileCopy::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    // Bias copies (must be provided by TileCopy when ElementBias != void)
    using CopyGmToL1Bias = typename TileCopy::CopyGmToL1Bias;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;

    using ElementAccumulator = std::conditional_t<
        !std::is_void_v<ElementAccumulator_>, ElementAccumulator_,
        typename Gemm::helper::ElementAccumulatorSelector<ElementFmap, ElementFilter>::ElementAccumulator>;

    using LayoutTagFmap = typename TileCopy::LayoutTagFmap;
    using LayoutTagFilter = typename TileCopy::LayoutTagFilter;
    using LayoutTagOutput = typename TileCopy::LayoutTagOutput;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t MAX_STAGES = 2;
    static constexpr uint32_t L1A_STAGES =
        (DispatchPolicy::L1A_STAGES < MAX_STAGES) ? DispatchPolicy::L1A_STAGES : MAX_STAGES;
    static constexpr uint32_t L1B_STAGES =
        (DispatchPolicy::L1B_STAGES < MAX_STAGES) ? DispatchPolicy::L1B_STAGES : MAX_STAGES;
    static constexpr uint32_t L0A_STAGES =
        (DispatchPolicy::L0A_STAGES < MAX_STAGES) ? DispatchPolicy::L0A_STAGES : MAX_STAGES;
    static constexpr uint32_t L0B_STAGES =
        (DispatchPolicy::L0B_STAGES < MAX_STAGES) ? DispatchPolicy::L0B_STAGES : MAX_STAGES;

    static constexpr uint32_t L0_TILE_M = L0TileShape::M;
    static constexpr uint32_t L0_TILE_N = L0TileShape::N;
    static constexpr uint32_t L0_TILE_K = L0TileShape::K;

    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0A_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0B_STAGES;

    // Bias buffer size check: BT holds one L0 N-tile of accumulator
    static constexpr uint32_t BIAS_BUF_SIZE = L0_TILE_N * sizeof(ElementAccumulator);
    static_assert(BIAS_BUF_SIZE <= ArchTag::BIAS_SIZE, "BIAS_BUF_SIZE exceeds BT space! Reduce L0TileShape::N");

    static_assert(std::is_same_v<LayoutTagOutput, layout::NC1HWC0>, "LayoutTagOutput only support NC1HWC0 yet!");
    static_assert(!ENABLE_UNIT_FLAG, "UnitFlag only support false yet!");

    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementFilter);
    static_assert((L0B_TILE_SIZE * L0B_STAGES) <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");

    static constexpr uint32_t ELE_NUM_A_PER_C0 = BYTE_PER_C0 / sizeof(ElementFmap);
    static constexpr uint32_t ELE_NUM_B_PER_C0 = BYTE_PER_C0 / sizeof(ElementFilter);

    static_assert(
        (FilterL1TileShape::Cin1 * ELE_NUM_B_PER_C0 >= FmapL1TileShape::Cin1 * ELE_NUM_A_PER_C0) &&
            ((FilterL1TileShape::Cin1 * ELE_NUM_B_PER_C0) % (FmapL1TileShape::Cin1 * ELE_NUM_A_PER_C0) == 0),
        "FilterL1TileShape::Cin must be a multiple of FmapL1TileShape::Cin");

    static constexpr uint32_t K_FMAP_PER_FILTER = FilterL1TileShape::Cin1 / FmapL1TileShape::Cin1;

    // Event ids for bias (reuse after L1A/L1B stages)
    static constexpr int32_t L1_BIAS_EVENT = L1A_STAGES + L1B_STAGES;
    static constexpr int32_t L0_BIAS_EVENT = L0A_STAGES + L0B_STAGES;

    /// Construct
    CATLASS_DEVICE
    BlockConv2dTla(
        Arch::Resource<ArchTag>& resource, const Conv2dFilterParams& filterParams_, uint32_t l1BufAddrStart = 0)
        : filterParams(filterParams_), copyL1ToL0A(filterParams_)
    {
        hiBlock =
            (FmapL1TileShape::Ho - 1) * filterParams.strideH() + (filterParams.kh() - 1) * filterParams.dilationH() + 1;
        wiBlock =
            (FmapL1TileShape::Wo - 1) * filterParams.strideW() + (filterParams.kw() - 1) * filterParams.dilationW() + 1;
        l1A_size = FmapL1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0;
        uint32_t coutRound = RoundUp(FilterL1TileShape::Cout, C0_NUM_PER_FRACTAL);
        l1B_size = FilterL1TileShape::Cin1 * filterParams.kh() * filterParams.kw() * coutRound * BYTE_PER_C0;
        // L1 bias: one L0-N tile of ElementBias (pad to C0 alignment)
        l1Bias_size = RoundUp(L0_TILE_N * sizeof(ElementBias), BYTE_PER_C0);

        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + l1A_size * L1A_STAGES;
        uint32_t l1BiasOffset = l1BOffset + l1B_size * L1B_STAGES;

        for (uint32_t i = 0; i < MAX_STAGES; i++) {
            if (i < L1A_STAGES) {
                l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFmap>(l1AOffset + l1A_size * i);
                l1AEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            if (i < L1B_STAGES) {
                l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFilter>(l1BOffset + l1B_size * i);
                l1BEventList[i] = i + L1A_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            if (i < L0A_STAGES) {
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementFmap>(L0A_PINGPONG_BUF_SIZE * i);
                l0AEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            if (i < L0B_STAGES) {
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementFilter>(L0B_PINGPONG_BUF_SIZE * i);
                l0BEventList[i] = i + L0A_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

        // Bias buffers
        l1BiasTensor = resource.l1Buf.template GetBufferByByte<ElementBias>(l1BiasOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BIAS_EVENT);
        l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_BIAS_EVENT);
    }

    CATLASS_DEVICE
    ~BlockConv2dTla()
    {
        for (uint32_t i = 0; i < MAX_STAGES; i++) {
            if (i < L1A_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            if (i < L1B_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            if (i < L0A_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            if (i < L0B_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BIAS_EVENT);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_BIAS_EVENT);
    }

    /**
     * @param tensorBias  bias GM tensor, layout = VectorLayout(cout), element = ElementBias
     *                    若无 bias，可传空 GlobalTensor（data()==nullptr），此时走无 bias 路径
     */
    template <class TensorFmap, class TensorFilter, class TensorOutput, class TensorBias>
    CATLASS_DEVICE void operator()(
        TensorFmap const& tensorFmap, TensorFilter const& tensorFilter, TensorOutput const& tensorOutput,
        TensorBias const& tensorBias, Conv2dCoord const& actualShape, uint8_t* blockPadList)
    {
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
        using CopyL0CToGm = typename TileCopy::template CopyL0CToGm<TensorOutput>;
        CopyL0CToGm copyL0CToDst;
#endif
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorOutput>;
        CopyL0CToDst copyL0CToDst;
#endif

        const bool hasBias = (tensorBias.data().GetPhyAddr() != 0);

        uint8_t blockPadLeft = blockPadList[0];
        uint8_t blockPadRight = blockPadList[1];
        uint8_t blockPadTop = blockPadList[2];
        uint8_t blockPadBottom = blockPadList[3];
        uint32_t wiActual = actualShape.w();
        uint32_t hiActual = actualShape.h();
        uint32_t coutActual = actualShape.cout();
        int32_t wiActualOrg = wiActual + blockPadLeft + blockPadRight;
        int32_t hiActualOrg = hiActual + blockPadTop + blockPadBottom;

        uint32_t hoActual =
            min((hiActualOrg - 1 - (filterParams.kh() - 1) * filterParams.dilationH()) / filterParams.strideH() + 1,
                FmapL1TileShape::Ho);
        uint32_t woActual =
            min((wiActualOrg - 1 - (filterParams.kw() - 1) * filterParams.dilationW()) / filterParams.strideW() + 1,
                FmapL1TileShape::Wo);
        uint32_t howoActual = hoActual * woActual;

        uint32_t nPartLoop = CeilDiv(coutActual, L0_TILE_N);
        uint32_t filterKwKhC0 = filterParams.kh() * filterParams.kw() * ELE_NUM_A_PER_C0;
        uint32_t maxKPartForL0B = L0B_PINGPONG_BUF_SIZE / (L0_TILE_N * sizeof(ElementFilter));
        uint32_t cin1L0Tile = Max(L0_TILE_K / filterKwKhC0, 1U);
        uint32_t kPartMaxSize = cin1L0Tile * filterKwKhC0;
        if (kPartMaxSize > maxKPartForL0B) {
            kPartMaxSize = maxKPartForL0B;
            cin1L0Tile = Max(kPartMaxSize / filterKwKhC0, 1U);
        }

        auto layoutFmapInL1 = tla::MakeLayoutFmap<ElementFmap>((uint32_t)1, FmapL1TileShape::Cin1, hiActual, wiActual);
        auto layoutFilterInL1 = tla::MakeLayoutFilter<ElementFilter, Arch::PositionL1>(
            FilterL1TileShape::Cin1, filterParams.kh(), filterParams.kw(), coutActual);
        auto layoutInL0C = tla::MakeLayoutL0C(howoActual, coutActual);
        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});

        uint32_t cin1Actual = min(actualShape.cin1(), FmapL1TileShape::Cin1);
        uint32_t cin1FilterActual = min(actualShape.cin1(), FilterL1TileShape::Cin1);

        // ---- first Fmap / Filter load (same as non-bias version) ----
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
        auto tensorL1A = tla::MakeTensor(l1ATensorList[l1AListId], layoutFmapInL1, Arch::PositionL1{});
        auto tensorTileFmap = GetTile(
            tensorFmap, tla::MakeCoord(0, 0, 0, 0, 0),
            tla::MakeShape((uint32_t)1, cin1Actual, hiActual, wiActual, ELE_NUM_A_PER_C0));
        copyGmToL1A(tensorL1A, tensorTileFmap);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
        auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], layoutFilterInL1, Arch::PositionL1{});
        auto tensorTileFilter = GetTile(
            tensorFilter, tla::MakeCoord(0, 0, 0, 0, 0),
            tla::MakeShape(
                cin1FilterActual, (uint32_t)filterParams.kh(), (uint32_t)filterParams.kw(), actualShape.cout(),
                ELE_NUM_B_PER_C0));
        copyGmToL1B(tensorL1B, tensorTileFilter);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

        // ---- main K / N loops ----
        uint32_t cin1TileCnt = CeilDiv<FmapL1TileShape::Cin1>(actualShape.cin1());
        uint32_t cin1FilterTileCnt = CeilDiv<FilterL1TileShape::Cin1>(actualShape.cin1());

        for (uint32_t cin1LoopIdx = 0; cin1LoopIdx < cin1TileCnt; cin1LoopIdx++) {
            uint32_t cin1FmapIdx = cin1LoopIdx % K_FMAP_PER_FILTER;

            uint32_t l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
            uint32_t l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;

            uint32_t cin1ActualNext{0};
            uint32_t cin1FilterActualNext{0};

            // preload next A/B (identical to non-bias)
            if (cin1LoopIdx < cin1TileCnt - 1) {
                uint32_t cin1LoopIdxNext = cin1LoopIdx + 1;
                cin1ActualNext = (cin1LoopIdxNext < cin1TileCnt - 1) ?
                                     FmapL1TileShape::Cin1 :
                                     (actualShape.cin1() - cin1LoopIdxNext * FmapL1TileShape::Cin1);

                auto l1ATensor = l1ATensorList[l1AListIdNext];
                auto tensorL1ANext = tla::MakeTensor(l1ATensor, layoutFmapInL1, Arch::PositionL1{});
                auto tensorTileFmapNext = GetTile(
                    tensorFmap, tla::MakeCoord(0, cin1LoopIdxNext * FmapL1TileShape::Cin1, 0, 0, 0),
                    tla::MakeShape((uint32_t)1, cin1ActualNext, hiActual, wiActual, ELE_NUM_A_PER_C0));

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                copyGmToL1A(tensorL1ANext, tensorTileFmapNext);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);

                if (cin1FmapIdx == K_FMAP_PER_FILTER - 1) {
                    uint32_t cin1FilterIdxNext = cin1LoopIdxNext / K_FMAP_PER_FILTER;
                    cin1FilterActualNext = (cin1FilterIdxNext < cin1FilterTileCnt - 1) ?
                                               FilterL1TileShape::Cin1 :
                                               (actualShape.cin1() - cin1FilterIdxNext * FilterL1TileShape::Cin1);

                    auto l1BTensor = l1BTensorList[l1BListIdNext];
                    auto tensorL1BNext = tla::MakeTensor(l1BTensor, layoutFilterInL1, Arch::PositionL1{});
                    auto tensorTileFilterNext = GetTile(
                        tensorFilter, tla::MakeCoord(cin1FilterIdxNext * FilterL1TileShape::Cin1, 0, 0, 0, 0),
                        tla::MakeShape(
                            cin1FilterActualNext, (uint32_t)filterParams.kh(), (uint32_t)filterParams.kw(),
                            actualShape.cout(), ELE_NUM_B_PER_C0));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                    copyGmToL1B(tensorL1BNext, tensorTileFilterNext);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
                }
            }

            auto l1ATensor = l1ATensorList[l1AListId];
            auto l1BTensor = l1BTensorList[l1BListId];
            tensorL1A = tla::MakeTensor(l1ATensor, layoutFmapInL1, Arch::PositionL1{});
            tensorL1B = tla::MakeTensor(l1BTensor, layoutFilterInL1, Arch::PositionL1{});
            uint32_t kPartLoop = CeilDiv(cin1Actual, cin1L0Tile);

            for (int kPartIdx = 0; kPartIdx < static_cast<int>(kPartLoop); kPartIdx++) {
                uint32_t cin1PartActual =
                    (kPartIdx < static_cast<int>(kPartLoop) - 1) ? cin1L0Tile : (cin1Actual - kPartIdx * cin1L0Tile);
                uint32_t kPartActual = cin1PartActual * filterParams.kh() * filterParams.kw() * ELE_NUM_A_PER_C0;

                auto l0ATile = l0ATensorList[l0AListId];
                auto layoutFmapInL0 = tla::MakeLayout<ElementFmap, LayoutTagL0A>(howoActual, kPartActual);
                auto tensorL0A = tla::MakeTensor(l0ATile, layoutFmapInL0, Arch::PositionL0A{});
                auto tensorTileL1A = GetTile(
                    tensorL1A, tla::MakeCoord(0, kPartIdx * cin1L0Tile, 0, 0, 0),
                    tla::MakeShape((uint32_t)1, cin1PartActual, hiActual, wiActual, ELE_NUM_A_PER_C0));

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if (kPartIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                }
                copyL1ToL0A(tensorL0A, tensorTileL1A, blockPadList);
                if (kPartIdx == static_cast<int>(kPartLoop) - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                }

                for (int nPartIdx = 0; nPartIdx < static_cast<int>(nPartLoop); nPartIdx++) {
                    uint32_t nPartActual =
                        (nPartIdx < static_cast<int>(nPartLoop) - 1) ? L0_TILE_N : (coutActual - nPartIdx * L0_TILE_N);

                    auto l0BTile = l0BTensorList[l0BListId];
                    auto layoutFilterInL0 = tla::MakeLayout<ElementFilter, LayoutTagL0B>(kPartActual, nPartActual);
                    auto tensorL0B = tla::MakeTensor(l0BTile, layoutFilterInL0, Arch::PositionL0B{});
                    auto tensorTileL1B = GetTile(
                        tensorL1B,
                        tla::MakeCoord(
                            cin1FmapIdx * FmapL1TileShape::Cin1 + kPartIdx * cin1L0Tile, 0, 0, nPartIdx * L0_TILE_N, 0),
                        tla::MakeShape(
                            cin1PartActual, (uint32_t)filterParams.kh(), (uint32_t)filterParams.kw(), nPartActual,
                            ELE_NUM_B_PER_C0));

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    if ((cin1FmapIdx == 0) && (kPartIdx == 0) && (nPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                    }
                    copyL1ToL0B(tensorL0B, tensorTileL1B);

                    if (((cin1FmapIdx == K_FMAP_PER_FILTER - 1) || (cin1LoopIdx == cin1TileCnt - 1)) &&
                        (kPartIdx == static_cast<int>(kPartLoop) - 1) &&
                        (nPartIdx == static_cast<int>(nPartLoop) - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    auto tensorTileL0C = GetTile(
                        tensorL0C, tla::MakeCoord(0, nPartIdx * L0_TILE_N), tla::MakeShape(howoActual, nPartActual));

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    // first K-tile: reset C (+ optional Cube bias)
                    bool initC = (cin1LoopIdx == 0) && (kPartIdx == 0);

                    auto tensorTileL0CMmad = tla::MakeTensor(
                        tensorTileL0C.data()[tensorTileL0C.layout()(tensorTileL0C.coord())], tensorTileL0C.layout(),
                        tla::MakeCoord(0, 0), Arch::PositionL0C{});

                    if (initC && hasBias) {
                        // ---- Cube bias path: GM -> L1 -> BT -> Mmad(with bias) ----
                        // bias GM offset = nPartIdx * L0_TILE_N  (per-block cout start is already applied by caller)
                        layout::VectorLayout layoutTileBias(coutActual);          // full block cout
                        layout::VectorLayout layoutBiasInL1(nPartActual);          // this N tile
                        layout::VectorLayout layoutBiasInBT(nPartActual);

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BIAS_EVENT);
                        // copy bias slice: biasGm[nPartIdx * L0_TILE_N : nPartIdx * L0_TILE_N + nPartActual]
                        copyGmToL1Bias(
                            l1BiasTensor,
                            tensorBias.data()[nPartIdx * L0_TILE_N],
                            layoutBiasInL1, layoutTileBias);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_BIAS_EVENT);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_BIAS_EVENT);

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_BIAS_EVENT);
                        copyL1ToBT(l0BiasTensor, l1BiasTensor, layoutBiasInBT, layoutBiasInL1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0_BIAS_EVENT);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0_BIAS_EVENT);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_BIAS_EVENT);

                        // TileMmadTla bias overload: cmatrixInitVal forced to false inside,
                        // bias is added by Cube Mmad hardware.
                        auto tensorL0Bias = tla::MakeTensor(
                            l0BiasTensor, layoutBiasInBT, Arch::PositionBias{});
                        tileMmad(
                            tensorTileL0CMmad, tensorL0A, tensorL0B, tensorL0Bias,
                            howoActual, nPartActual, kPartActual, /*initC=*/true);
                    } else {
                        tileMmad(
                            tensorTileL0CMmad, tensorL0A, tensorL0B,
                            howoActual, nPartActual, kPartActual, initC);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
            l1AListId = l1AListIdNext;
            cin1Actual = cin1ActualNext;
            if (cin1FmapIdx == K_FMAP_PER_FILTER - 1) {
                l1BListId = l1BListIdNext;
                cin1FilterActual = cin1FilterActualNext;
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        copyL0CToDst(tensorOutput, tensorL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    static bool CanImplement(const Conv2dFilterParams& filterParams)
    {
        uint32_t hiBlock =
            (FmapL1TileShape::Ho - 1) * filterParams.strideH() + (filterParams.kh() - 1) * filterParams.dilationH() + 1;
        uint32_t wiBlock =
            (FmapL1TileShape::Wo - 1) * filterParams.strideW() + (filterParams.kw() - 1) * filterParams.dilationW() + 1;
        uint32_t l1A_size = FmapL1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0;
        uint32_t coutRound = RoundUp(FilterL1TileShape::Cout, C0_NUM_PER_FRACTAL);
        uint32_t l1B_size = FilterL1TileShape::Cin1 * filterParams.kh() * filterParams.kw() * coutRound * BYTE_PER_C0;
        uint32_t l1Bias_size = RoundUp(L0_TILE_N * sizeof(ElementBias), BYTE_PER_C0);
        if (l1A_size * L1A_STAGES + l1B_size * L1B_STAGES + l1Bias_size > ArchTag::L1_SIZE) {
            return false;
        }
        uint32_t filterKwKhC0 = filterParams.kh() * filterParams.kw() * ELE_NUM_A_PER_C0;
        constexpr uint32_t MAX_KPART_HW = 512;
        if (filterKwKhC0 > MAX_KPART_HW) {
            return false;
        }
        if (BIAS_BUF_SIZE > ArchTag::BIAS_SIZE) {
            return false;
        }
        return true;
    }

protected:
    Conv2dFilterParams filterParams;
    uint32_t hiBlock, wiBlock;
    uint32_t l1A_size, l1B_size, l1Bias_size;

    AscendC::LocalTensor<ElementFmap> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementFilter> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementFmap> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementFilter> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;
    AscendC::LocalTensor<ElementBias> l1BiasTensor;
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;

    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];

    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyGmToL1Bias copyGmToL1Bias;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
};

} // namespace Catlass::Conv::Block

#endif // CATLASS_CONV_BLOCK_BLOCK_CONV2D_PINGPONG_BIAS_TLA_HPP
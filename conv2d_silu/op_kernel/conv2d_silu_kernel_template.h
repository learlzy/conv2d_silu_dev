#ifndef __CONV2D_SILU_KERNEL_TEMPLATE_H__
#define __CONV2D_SILU_KERNEL_TEMPLATE_H__

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/conv/block/block_conv.hpp"
#include "catlass/conv/block/block_swizzle.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/conv/kernel/conv2d_epilogue.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_silu.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

namespace NsConv2dSilu {

using namespace Catlass;
using namespace Catlass::Conv::Kernel;
using namespace Catlass::Epilogue;

template <typename ElementType, typename ElementAccumulator>
struct Conv2dSiluKernelTraits {
    using ElementFmap = ElementType;
    using ElementFilter = ElementType;
    using ElementBias = ElementType;
    using ElementOutput = ElementType;

    using ArchTag = Arch::AtlasA2;
    // static constexpr uint32_t L1A_STAGES = 2;
    // static constexpr uint32_t L1B_STAGES = 2;
    // static constexpr uint32_t L0A_STAGES = 2;
    // static constexpr uint32_t L0B_STAGES = 2;
    // static constexpr uint32_t L0C_STAGES = 1;
    // static constexpr bool ENABLE_UNIT_FLAG = false;

    static constexpr uint32_t L1A_STAGES = 1;
    static constexpr uint32_t L1B_STAGES = 1;
    static constexpr uint32_t L0A_STAGES = 1;
    static constexpr uint32_t L0B_STAGES = 1;
    static constexpr uint32_t L0C_STAGES = 1;
    static constexpr bool ENABLE_UNIT_FLAG = false;

    using DispatchPolicy =
        Conv::ConvPingpong<ArchTag, L1A_STAGES, L1B_STAGES, L0A_STAGES, L0B_STAGES, L0C_STAGES, ENABLE_UNIT_FLAG>;

    // using FmapL1TileShape = Catlass::Conv2dFmapL1Shape<4, 8, 8>;    // <Ho, Wo, Cin1_small>
    // using FilterL1TileShape = Catlass::Conv2dFilterL1Shape<96, 8>;  // <Cout, Cin1_big>
    // using L0TileShape = Catlass::Conv2dL0Shape<16, 96, 16>;         // <M, N, K>

    using FmapL1TileShape = Catlass::Conv2dFmapL1Shape<14, 16, 1>;    // <Ho, Wo, Cin1_small>
    using FilterL1TileShape = Catlass::Conv2dFilterL1Shape<32, 1>;  // <Cout, Cin1_big>
    using L0TileShape = Catlass::Conv2dL0Shape<16, 32, 16>;         // <M, N, K>

    using BlockConv2d = Conv::Block::BlockConv2dTla<
        DispatchPolicy, FmapL1TileShape, FilterL1TileShape, L0TileShape,
        ElementFmap, ElementFilter, ElementOutput, ElementAccumulator>;

    using BlockScheduler = typename Conv::Block::Conv2dIdentityBlockSwizzle<3, 0>;

    // 存储类型始终为 ElementOutput（匹配 AIC 写入 workspace 的精度）
    using StorageType = Gemm::GemmType<ElementOutput, layout::NC1HWC0>;
    // SiLU 计算精度（默认用 float 做高精度计算，也可改为 half）
    using ComputeType = Gemm::GemmType<float, layout::NC1HWC0>;
    using FinalOutputType = Gemm::GemmType<ElementOutput, layout::NC1HWC0>;

    using TileEpilogue = Tile::TileElemWiseSilu<ArchTag, ComputeType, 256>;
    using TileCopy = Tile::TileCopy<ArchTag, StorageType, FinalOutputType>;

    using BlockEpilogue = Block::BlockEpilogue<
        EpilogueAtlasA2Conv2dElemWise,
        StorageType,
        FinalOutputType,
        TileEpilogue,
        TileCopy,
        std::integral_constant<uint32_t, FmapL1TileShape::Ho>,
        std::integral_constant<uint32_t, FmapL1TileShape::Wo>,
        std::integral_constant<uint32_t, FilterL1TileShape::Cout>>;

    using Conv2dKernel = Conv2dEpilogue<BlockConv2d, BlockEpilogue, BlockScheduler>;
};

} // namespace NsConv2dSilu

#endif // __CONV2D_SILU_KERNEL_TEMPLATE_H__
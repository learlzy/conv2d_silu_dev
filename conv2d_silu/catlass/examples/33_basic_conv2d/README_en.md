# Basic Conv2d Example Readme

## Code Organization

```text
├── 33_basic_conv2d
│   ├── CMakeLists.txt       # CMake build file
│   ├── README.md
│   ├── basic_conv2d.cpp     # Basic 2D convolution main file
│ └── conv2d_bias_silu.cpp   # Conv2D+Bias+SiLU cross-core fusion main file
```

## Description

### basic_conv2d (Basic 2D Convolution)

- Function: performs basic convolution computation.
- Input feature map layout (Ascend-friendly): `(N, C1, H, W, C0)`, where:
  - N: Batch size
  - C1: `C1 = CeilDiv(C, C0)`, where `C` is the number of input channels and `C0` is 16
  - H: Feature map height
  - W: Feature map width
  - C0: `C0` is 16
- Convolution kernel layout (Ascend-friendly): `(Cin, Kh, Kw, Cout, C0)`, where:
  - Cin: `C1 = CeilDiv(C, C0)`
  - Kh: Kernel height
  - Kw: Kernel width
  - Cout: Number of output channels
  - C0: `C0` is 16
- Basic constraints:
  - Dilation coefficients `dilations` and convolution kernel strides `strides` cannot be zero.
  - The effective receptive field size of the convolution kernel cannot exceed the input feature map size:
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`

    where `hi` and `wi` are the height and width of the input feature map, `kh` and `kw` are the height and width of the convolution kernel, `dilationH` and `dilationW` are the dilation coefficients in the two directions, and `padTop`, `padBottom`, `padLeft`, `padRight` are the top, bottom, left, and right padding sizes.

- On-chip buffer constraints (lowercase symbols in the following formulas are runtime constants; others are compile-time constants):
  - `L1_STAGES * FmapSize + L1_STAGES * FilterSize <= L1_SIZE`
    where `L1_STAGES` is 2 when double-buffering is enabled and 1 otherwise, L1_SIZE is 512K (AtlasA2/A3). FmapSize and FilterSize are calculated as follows:
    - `FmapSize = Cin1 * hi * wi * C0 * sizeof(ElementFmap)`, where `Cin1` is the tiling constant (number of input channels) under `FmapL1TileShape`; hi and wi can be derived from `Ho` and `Wo` (output feature map sizes) under `FmapL1TileShape`:
      1. `hi = (Ho - 1) * strideH + dilationH * (kh - 1) + 1`
      2. `wi = (Wo - 1) * strideW + dilationW * (kw - 1) + 1`

    - `FilterSize = Cin1 * kh * kw * Cout * C0`, where `Cin1` is the tiling constant (number of input channels) under `FilterL1TileShape`, and `Cout` is the value of `Cout` (number of output channels) under `FilterL1TileShape` aligned to `C0`

  - `L0A_STAGES * FmapL0ASize <= L0A_SIZE`, where `FmapL0ASize` is `Ho * Wo * max(L0K, kh * kw * C0) * sizeof(ElementFmap)`, `L0K` is the tiling constant under `L0TileShape`, and `L0A_SIZE` is 64K (AtlasA2/A3)
  - `L0B_STAGES * FilterL0BSize <= L0B_SIZE`, where `FilterL0BSize` is `max(L0K, kh * kw * C0) * CoutL0 * sizeof(ElementFilter)`, `L0K` is the tiling constant under `L0TileShape`, `CoutL0` is the size of `L0N` aligned to `C0`, and `L0B_SIZE` is also 64K (AtlasA2/A3)
  - `Ho * Wo * Cout * sizeof(ElementOut) <= L0C_SIZE`, where `L0C_SIZE` is 128K (AtlasA2/A3), and `Ho`, `Wo`, `Cout` are all tiling constants under `FilterL1TileShape`.
  (ElementFmap, ElementFilter, and ElementOutput are all of the fp16 type in this example.)

### conv2d_bias_silu (Conv2D+Bias+SiLU Cross-Core Fusion)

- Function: performs **Conv2D + Bias + SiLU activation fusion computation**. The implementation is taken from the AtlasA2 (2201) branch of the original `conv2d_bias_silu` example; only the 2201 platform implementation is kept in this directory.
- Adopts an AIC (Cube Core) + AIV (Vector Core) cross-core collaborative architecture to avoid GM memory read/write of intermediate results and minimize memory access overhead.
- Operator pipeline: **AIC core executes Conv2D (Matrix Multiply) → Cross-core synchronization → AIV core executes BiasAdd + SiLU activation (Element-wise)**.

#### Input/Output Layout

- **Input feature map**: `(N, C1, H, W, C0)` corresponding to `layout::NC1HWC0`, where:
  - N: Batch size
  - C1: `C1 = CeilDiv(Cin, C0)`, Cin is the input channel count
  - H: Input feature map height
  - W: Input feature map width
  - C0: `C0 = 16` (byte_per_c0 / sizeof(fp16))
- **Convolution kernel**: `(Cin1, Kh, Kw, Cout, C0)` corresponding to `layout::CI1KHKWCOCI0`
- **Bias**: `(CoutRound,)`, indexed by output channel
- **Output feature map**: `(N, Cout1, Ho, Wo, C0)` corresponding to `layout::NC1HWC0`

#### Data Types

- Feature map, convolution kernel, bias, and output are all `fp16_t` (half precision).
- Convolution accumulation and SiLU intermediate computation use float (FP32) precision by default to ensure numerical stability.
- Optional specialization macros (both disabled by default):
  - `ENABLE_FP16_ACCUMULATOR=1`: convolution accumulation in fp16 precision
  - `ENABLE_FP16_SILU=1`: SiLU computed in fp16 precision (corresponding to the extra build target `33_conv2d_bias_silu_fp16_silu`)

#### Basic Constraints

- Dilation coefficients `dilations` and convolution kernel strides `strides` cannot be zero.
- The effective receptive field size of the convolution kernel cannot exceed the input feature map size:
  - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
  - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`
- L1/L0 cache space constraints must be satisfied (same as the [basic_conv2d constraints](#basic_conv2d-basic-2d-convolution)). Tile Shape:
  - `FmapL1TileShape<8, 12, 8>`, `FilterL1TileShape<96, 8>`, `L0TileShape<16, 96, 16>`
  - Note: the L1 requirement grows with `strideH/strideW` (`hi`/`wi` scale with the stride). With the default tile config, a 3x3 kernel only supports stride=1 (about 98% L1 usage); for stride=2 a smaller kernel is required (for example 2x2, about 75% L1 usage), otherwise the case is rejected by `CanImplement`.

#### Cross-Core Synchronization Mechanism (AtlasA2 pipeline synchronization model)

- After an AIC core completes the Conv2D computation of a block, it sets the completion flag via the CANN native API `CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>`.
- AIV cores wait for AIC completion via `CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>`, then execute BiasAdd + SiLU and write out the results.
- Workspace is used for intermediate result transfer from AIC to AIV. AIC writes in half precision, with size `batch * ho * wo * coutRound * sizeof(fp16)`.
- The AtlasA2 platform requires `hardwareSyncAddr` (obtained via `aclrtGetHardwareSyncAddr`; if the call fails, cross-core fusion cannot be enabled and the operator exits directly).

## Example

### basic_conv2d

- After obtaining the code, build the operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#build-and-execution).
- Execute the operator.

```bash
# Build a specified test case.
bash scripts/build.sh 33_basic_conv2d
cd ./output/bin
# Executable file name |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# The device ID is optional. The default value is 0.
./33_basic_conv2d 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

If the following result is displayed, the accuracy verification is successful.

```text
Compare success.
```

### conv2d_bias_silu

- Execute the operator (default float accumulation and SiLU precision).

```bash
# Build a specified test case.
bash scripts/build.sh 33_conv2d_bias_silu
cd ./output/bin
# Executable file name |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# The device ID is optional. The default value is 0.
./33_conv2d_bias_silu 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

If the following result is displayed, the accuracy verification is successful (strong validation: the AIC workspace conv result and the AIV final output are compared separately).

```text
Workspace(AIC conv) vs golden conv comparison: success.
Output(AIV) vs golden SiLU(conv+bias) comparison: success.
```

- To verify the SiLU fp16 specialization path, build the extra target `33_conv2d_bias_silu_fp16_silu` and execute it.

```bash
bash scripts/build.sh 33_conv2d_bias_silu_fp16_silu
cd ./output/bin
./33_conv2d_bias_silu_fp16_silu 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

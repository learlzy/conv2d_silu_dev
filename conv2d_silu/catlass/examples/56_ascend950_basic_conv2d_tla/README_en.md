# BasicConv2dTla Example Readme

## Code Organization

```text
├── 56_ascend950_basic_conv2d_tla
│   ├── CMakeLists.txt        # CMake build file
│   ├── README.md
│   ├── basic_conv2d_tla.cpp  # Basic 2D convolution main file
│   └── conv2d_bias_silu.cpp  # Conv2D+Bias+SiLU cross-core fusion main file
```

## Description

### basic_conv2d_tla (Basic 2D Convolution)

- Function: performs basic convolution computation (using [`TLA`](../../docs/en/2_Design/02_tla/01_layout.md) semantics).

### conv2d_bias_silu (Conv2D+Bias+SiLU Cross-Core Fusion)

- Function: performs **Conv2D + Bias + SiLU activation fusion computation**. The implementation is taken from the Ascend950 (3510) branch of the original `conv2d_bias_silu` example; only the 3510 platform implementation is kept in this directory.
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
- SiLU intermediate computation uses float (FP32) precision to ensure numerical stability of the activation function.

#### Basic Constraints

- Dilation coefficients `dilations` and convolution kernel strides `strides` cannot be zero.
- The effective receptive field size of the convolution kernel cannot exceed the input feature map size:
  - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
  - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`
- L1 cache space constraints must be satisfied. Ascend950 uses the TLA architecture; Tile Shape needs to be reduced to accommodate 2-stage pingpong buffering:
  - Tile Shape: `FmapL1TileShape<4, 8, 4>`, `FilterL1TileShape<48, 4>`, `L0TileShape<16, 16, 16>`

#### Cross-Core Synchronization Mechanism (3510 bulk synchronization model)

- After an AIC core completes all Conv2D computation, it waits for all AIC cores via `CrossCoreBarrier`, then sets the completion flag with `CrossCoreSetFlag<0x2, PIPE_FIX>`.
- AIV cores wait for AIC completion via `CrossCoreWaitFlag<0x2, PIPE_MTE2>`, then execute BiasAdd + SiLU and write out the results.
- Workspace is used for intermediate result transfer from AIC to AIV, with size `batch * ho * wo * coutRound * sizeof(fp16)`.
- The 3510 platform does not require `hardwareSyncAddr` (unlike AtlasA2).

## Remarks

The overall design of this test case is the same as that of [_basic_matmul](../33_basic_conv2d/README.md). The difference is that TLA-related abstraction is used. Therefore, related examples are provided for description.

## Example

### basic_conv2d_tla

- After obtaining the code, compile the corresponding operator executable file. For details, see [Template Library Quick Start](../../docs/en/1_Practice/01_quick_start.md#operator-compilation). This test case is an Ascend 950 operator. During compilation, you need to add -DCATLASS_ARCH=3510.
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 56_ascend950_basic_conv2d_tla -DCATLASS_ARCH=3510
cd ./output/bin
# Executable file name |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# The device ID is optional. The default value is 0.
./56_ascend950_basic_conv2d_tla 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

If the following result is displayed, the accuracy verification is successful.

```text
Compare success.
```

### conv2d_bias_silu

- During compilation, you need to add -DCATLASS_ARCH=3510.
- Execute the operator.

```bash
# Compile a specified test case.
bash scripts/build.sh 56_ascend950_conv2d_bias_silu -DCATLASS_ARCH=3510
cd ./output/bin
# Executable file name |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# The device ID is optional. The default value is 0.
./56_ascend950_conv2d_bias_silu 2 33 43 112 80 3 3 2 2 2 2 2 2 1 1 0
```

If the following result is displayed, the accuracy verification is successful (strong validation: the AIC workspace conv result and the AIV final output are compared separately).

```text
Workspace(AIC conv) vs golden conv comparison: success.
Output(AIV) vs golden SiLU(conv+bias) comparison: success.
```

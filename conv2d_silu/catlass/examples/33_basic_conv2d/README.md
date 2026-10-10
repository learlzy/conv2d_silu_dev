# BasicConv2d Example Readme

## 代码组织

```text
├── 33_basic_conv2d
│   ├── CMakeLists.txt        # CMake编译文件
│   ├── README.md
│   ├── basic_conv2d.cpp      # 基础2D卷积主文件
│   └── conv2d_bias_silu.cpp  # Conv2D+Bias+SiLU跨核融合主文件
```

## 功能介绍

### basic_conv2d（基础2D卷积）

- 该算子完成2D版本的卷积计算
- 昇腾亲和的特征图尺寸表达是：`(N, C1, H, W, C0)`，其中:
  - N: 批量Batch大小
  - C1: `C1 = CeilDiv(C, C0)`，其中`C`为输入特征图的通道数，`C0`为16
  - H: 特征图高度
  - W: 特征图宽度
  - C0: `C0`为16
- 昇腾亲和的卷积核尺寸表达是：`(Cin, Kh, Kw, Cout, C0)`，其中:
  - Cin: `C1 = CeilDiv(C, C0)`
  - Kh: 卷积核高度
  - Kw: 卷积核宽度
  - Cout: 输出通道数
  - C0: `C0`为16
- 需满足以下基础约束：
  - 膨胀系数`dilations`和卷积核计算步幅`strides`均不能为零
  - 卷积核的有效感受野大小不能超过输入特征图的大小，需满足：
    - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
    - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`

    其中`hi`，`wi`为输入特征图的高度和宽度，`kh`, `kw`为卷积核的高度和宽度，`dilationH`, `dilationW`为上述两方向上的膨胀系数，`padTop`, `padBottom`, `padLeft`, `padRight`为上、下、左、右填充大小。

- 考虑到空间分配，需满足下述条件（为做区分下述公式中小写的符号为运行时常量，反之是编译期常量）：
  - `L1_STAGES * FmapSize + L1_STAGES * FilterSize <= L1_SIZE`
    其中`L1_STAGES`在开double-buffer的情形下为2，不启用为1，L1_SIZE是512K（AtlasA2/A3），FmapSize和FilterSize的具体计算公式为：
    - `FmapSize = Cin1 * hi * wi * C0 * sizeof(ElementFmap)`, `Cin1`为`FmapL1TileShape`下的Tiling常量（输入通道数），hi和wi由`FmapL1TileShape`下的`Ho`和`Wo`（输出特征尺寸）可反推得到，逆运算为：
      1. `hi = (Ho - 1) * strideH + dilationH * (kh - 1) + 1`
      2. `wi = (Wo - 1) * strideW + dilationW * (kw - 1) + 1`

    - `FilterSize = Cin1 * kh * kw * Cout * C0`，`Cin1`为`FilterL1TileShape`下的Tiling常量（输入通道数）, `Cout`是`FilterL1TileShape`下`Cout`（输出通道数）对齐到`C0`后的值

  - `L0A_STAGES * FmapL0ASize <= L0A_SIZE`, 其中`FmapL1ASize`具体为`Ho * Wo * max(L0K, kh * kw * C0) * sizeof(ElementFmap)`，`L0K`为`L0TileShape`下的Tiling常量, `L0A_SIZE`等于64K（AtlasA2/A3）
  - `L0B_STAGES * FilterL0BSize <= L0B_SIZE`, 其中`FilterL0BSize`具体为`max(L0K, kh * kw * C0) * CoutL0 * sizeof(ElementFilter)`，`L0K`为`L0TileShape`下的Tiling常量, `CoutL0`是`L0N`对齐到`C0`的大小，`L0B_SIZE`也等于64K（AtlasA2/A3）
  - `Ho * Wo * Cout * sizeof(ElementOut) <= L0C_SIZE`, 其中`L0C_SIZE`等于128K（AtlasA2/A3），`Ho`, `Wo`, `Cout`均为`FilterL1TileShape`中的Tiling常量。
  （样例中ElementFmap，ElementFilter和ElementOutput为fp16类型）

### conv2d_bias_silu（Conv2D+Bias+SiLU 跨核融合）

- 算子功能：完成 **Conv2D + Bias + SiLU 激活融合计算**，实现取自原`conv2d_bias_silu`算子的 AtlasA2（2201）分支，本目录内仅保留 2201 平台实现
- 采用 AIC（Cube Core）+ AIV（Vector Core）跨核协同架构，避免中间结果的 GM 显存读写，实现访存开销最小化
- 算子流水线：**AIC 核执行 Conv2D（Matrix Multiply）→ 跨核同步 → AIV 核执行 BiasAdd + SiLU 激活（Element-wise）**

#### 输入输出 Layout

- **输入特征图**：`(N, C1, H, W, C0)` 对应 `layout::NC1HWC0`，其中：
  - N: 批量 Batch 大小
  - C1: `C1 = CeilDiv(Cin, C0)`，Cin 为输入通道数
  - H: 输入特征图高度
  - W: 输入特征图宽度
  - C0: `C0 = 16`（byte_per_c0 / sizeof(fp16)）
- **卷积核**：`(Cin1, Kh, Kw, Cout, C0)` 对应 `layout::CI1KHKWCOCI0`
- **偏置 Bias**：`(CoutRound,)`，按输出通道索引
- **输出特征图**：`(N, Cout1, Ho, Wo, C0)` 对应 `layout::NC1HWC0`

#### 数据类型

- 特征图、卷积核、偏置、输出均为 `fp16_t`（half 精度）
- 卷积累加与 SiLU 中间计算默认使用 float（FP32）精度，确保数值稳定性
- 可选特化宏（默认均关闭）：
  - `ENABLE_FP16_ACCUMULATOR=1`：卷积累加按 fp16 精度
  - `ENABLE_FP16_SILU=1`：SiLU 按 fp16 精度计算（对应附加编译目标`33_conv2d_bias_silu_fp16_silu`）

#### 基础约束

- 膨胀系数 `dilations` 和卷积核步幅 `strides` 均不能为零
- 卷积核的有效感受野大小不能超过输入特征图大小，需满足：
  - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
  - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`
- 需满足 L1/L0 缓存空间约束（同[`basic_conv2d` 约束](#basic_conv2d基础2d卷积)），Tile Shape：
  - `FmapL1TileShape<8, 12, 8>`，`FilterL1TileShape<96, 8>`，`L0TileShape<16, 96, 16>`
  - 注意：L1 需求随 `strideH/strideW` 增大而增长（`hi`/`wi` 随 stride 放大）。默认 tile 配置下 3×3 核仅支持 stride=1（L1 占用约 98%）；stride=2 时需使用更小的卷积核（如 2×2，L1 占用约 75%），否则会被 `CanImplement` 拒绝

#### 跨核同步机制（AtlasA2 管线同步模型）

- AIC 核完成单个 block 的 Conv2D 计算后，通过 CANN 原生 API `CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>` 设置完成标志
- AIV 核通过 `CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>` 等待 AIC 完成，随后执行 BiasAdd + SiLU 并写出结果
- Workspace 用于 AIC→AIV 的中间结果传递，AIC 以 half 精度写入，大小为 `batch * ho * wo * coutRound * sizeof(fp16)`
- AtlasA2 平台需要 `hardwareSyncAddr`（通过 `aclrtGetHardwareSyncAddr` 获取；获取失败时算子无法启用跨核融合，将直接退出）

## 使用示例

### basic_conv2d

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#编译执行)
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 33_basic_conv2d
cd ./output/bin
# 可执行文件名 |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# Device ID可选，默认为0
./33_basic_conv2d 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

执行结果如下，表明精度验证通过。

```text
Compare success.
```

### conv2d_bias_silu

- 执行算子（默认 float 累加与 SiLU 精度）

```bash
# 编译指定用例
bash scripts/build.sh 33_conv2d_bias_silu
cd ./output/bin
# 可执行文件名 |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# Device ID可选，默认为0
./33_conv2d_bias_silu 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

执行结果如下，表明精度验证通过（强校验：分别对比 AIC workspace 的纯卷积结果与 AIV 最终输出）。

```text
Workspace(AIC conv) vs golden conv comparison: success.
Output(AIV) vs golden SiLU(conv+bias) comparison: success.
```

- 如需验证 SiLU 按 fp16 计算的特化路径，编译附加目标`33_conv2d_bias_silu_fp16_silu`并执行

```bash
bash scripts/build.sh 33_conv2d_bias_silu_fp16_silu
cd ./output/bin
./33_conv2d_bias_silu_fp16_silu 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

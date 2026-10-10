# BasicConv2dTla Example Readme

## 代码组织

```text
├── 56_ascend950_basic_conv2d_tla
│   ├── CMakeLists.txt        # CMake编译文件
│   ├── README.md
│   ├── basic_conv2d_tla.cpp  # 基础2D卷积主文件
│   └── conv2d_bias_silu.cpp  # Conv2D+Bias+SiLU跨核融合主文件
```

## 功能说明

### basic_conv2d_tla（基础2D卷积）

- 算子功能：完成基础2D版本卷积计算（使用[`TLA`](../../docs/zh/2_Design/02_tla/01_layout.md)语义）
- 该用例总体设计与[`33_basic_conv2d`](../33_basic_conv2d/README.md)相同，区别为使用了TLA相关抽象，条件约束与[`BasicConv2d` 约束](../33_basic_conv2d/README.md#功能说明)一致，注意在Ascend950硬件条件下，`L0C_SIZE`为256K。

### conv2d_bias_silu（Conv2D+Bias+SiLU 跨核融合）

- 算子功能：完成 **Conv2D + Bias + SiLU 激活融合计算**，实现取自原`conv2d_bias_silu`算子的 Ascend950（3510）分支，本目录内仅保留 3510 平台实现
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
- SiLU 中间计算使用 float（FP32）精度，确保激活函数的数值稳定性

#### 基础约束

- 膨胀系数 `dilations` 和卷积核步幅 `strides` 均不能为零
- 卷积核的有效感受野大小不能超过输入特征图大小，需满足：
  - `hi + padTop + padBottom > dilationH * (kh - 1) + 1`
  - `wi + padLeft + padRight > dilationW * (kw - 1) + 1`
- 需满足 L1 缓存空间约束，Ascend950 使用 TLA 架构，需调小 Tile Shape 以适应 2-stage pingpong 缓冲：
  - Tile Shape：`FmapL1TileShape<4, 8, 4>`，`FilterL1TileShape<48, 4>`，`L0TileShape<16, 16, 16>`

#### 跨核同步机制（3510 bulk 同步模型）

- AIC 核完成全部 Conv2D 计算后，通过 `CrossCoreBarrier` 等待所有 AIC 就绪，再以 `CrossCoreSetFlag<0x2, PIPE_FIX>` 设置完成标志
- AIV 核通过 `CrossCoreWaitFlag<0x2, PIPE_MTE2>` 等待 AIC 完成，随后执行 BiasAdd + SiLU 并写出结果
- Workspace 用于 AIC→AIV 的中间结果传递，大小为 `batch * ho * wo * coutRound * sizeof(fp16)`
- 3510 平台无需 `hardwareSyncAddr`（区别于 AtlasA2）

## 使用示例

### basic_conv2d_tla

- 获取代码之后编译相应的算子可执行文件，可参考[quickstart](../../docs/zh/1_Practice/01_quick_start.md#算子编译)，本用例为Ascend 950算子，编译时需加-DCATLASS_ARCH=3510
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 56_ascend950_basic_conv2d_tla -DCATLASS_ARCH=3510
cd ./output/bin
# 可执行文件名 |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# Device ID可选，默认为0
./56_ascend950_basic_conv2d_tla 2 33 43 112 80 3 3 2 2 2 2 1 1 1 1 0
```

执行结果如下，表明精度验证通过。

```text
Compare success.
```

### conv2d_bias_silu

- 编译时需加-DCATLASS_ARCH=3510
- 执行算子

```bash
# 编译指定用例
bash scripts/build.sh 56_ascend950_conv2d_bias_silu -DCATLASS_ARCH=3510
cd ./output/bin
# 可执行文件名 |Batch|Hi|Wi|Cin|Cout|kh|kw|padL|padR|padT|padB|strideH|strideW|dilationH|dilationW|Device ID
# Device ID可选，默认为0
./56_ascend950_conv2d_bias_silu 2 33 43 112 80 3 3 2 2 2 2 2 2 1 1 0
```

执行结果如下，表明精度验证通过（强校验：分别对比 AIC workspace 的纯卷积结果与 AIV 最终输出）。

```text
Workspace(AIC conv) vs golden conv comparison: success.
Output(AIV) vs golden SiLU(conv+bias) comparison: success.
```

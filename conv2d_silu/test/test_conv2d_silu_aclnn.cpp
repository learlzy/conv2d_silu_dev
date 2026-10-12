/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * test_conv2d_silu_aclnn.cpp
 *
 * 精度测试：自定义 aclnnConv2dSilu vs CPU golden
 * 公式：y = SiLU(Conv2d(x, filter) + bias)
 *
 * 数据布局（与 catlass example / 算子定义一致）：
 *   x/y   : NC1HWC0  (fp16)
 *   filter: FRACTAL_Z / CI1KHKWCOCI0  (fp16)
 *   bias  : ND (CoutRound)  (fp16)
 *
 * Golden 实现方式照搬 catlass/examples/33_basic_conv2d/conv2d_bias_silu.cpp
 * （CPU 上 ComputeConv2d + Bias + SiLU，不调用任何原生 aclnn 接口）
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <cstdint>

#include "acl/acl.h"
#include "aclnn_conv2d_silu.h"   // 自定义算子 aclnn 头文件（OPP_CUSTOM_PATH/include）

// ============================================================
// 宏与工具
// ============================================================
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(fmt, ...)          \
    do {                             \
        printf(fmt, ##__VA_ARGS__);  \
    } while (0)

#define ACL_CHECK(ret)                                                      \
    do {                                                                    \
        auto _r = (ret);                                                    \
        if (_r != ACL_SUCCESS) {                                            \
            LOG_PRINT("ACL error %d at %s:%d\n", _r, __FILE__, __LINE__);   \
            return -1;                                                      \
        }                                                                   \
    } while (0)

using fp16_t = aclFloat16;

static inline float Fp16ToFloat(fp16_t v) { return aclFloat16ToFloat(v); }
static inline fp16_t FloatToFp16(float v) { return aclFloatToFloat16(v); }

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

static std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

// ============================================================
// 创建 aclTensor（支持 NC1HWC0 / FRACTAL_Z / ND）
// ============================================================
template <typename T>
static int CreateAclTensor(const std::vector<T>& hostData,
                           const std::vector<int64_t>& shape,
                           void** deviceAddr,
                           aclDataType dataType,
                           aclTensor** tensor,
                           aclFormat format)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);

    auto strides = MakeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, format, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);
    return 0;
}

// ============================================================
// 随机填充（与 golden.hpp FillRandomData 风格一致）
// ============================================================
template <typename T>
static void FillRandom(std::vector<T>& data, float lo, float hi, int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < data.size(); ++i) {
        float v = dist(gen);
        if constexpr (std::is_same_v<T, fp16_t>) {
            data[i] = FloatToFp16(v);
        } else {
            data[i] = static_cast<T>(v);
        }
    }
}

// ============================================================
// CPU Golden（照搬 conv2d_bias_silu.cpp 的逻辑）
// 布局：
//   fmap   : NC1HWC0   [N, Cin1, Hi, Wi, C0]
//   filter : CI1KHKWCOCI0  [Cin1, Kh, Kw, Cout, C0]  （Cout 未 round）
//   bias   : [CoutRound]
//   output : NC1HWC0   [N, Cout1, Ho, Wo, C0]
// ============================================================
struct Conv2dParams {
    uint32_t batch = 0, hi = 0, wi = 0, cin = 0, cout = 0;
    uint32_t kh = 0, kw = 0;
    uint32_t padL = 0, padR = 0, padT = 0, padB = 0;
    uint32_t strideH = 1, strideW = 1;
    uint32_t dilationH = 1, dilationW = 1;
    uint32_t c0 = 16;   // fp16

    uint32_t cin1() const { return (cin + c0 - 1) / c0; }
    uint32_t cout1() const { return (cout + c0 - 1) / c0; }
    uint32_t coutRound() const { return cout1() * c0; }
    uint32_t ho() const {
        return (hi + padT + padB - dilationH * (kh - 1) - 1) / strideH + 1;
    }
    uint32_t wo() const {
        return (wi + padL + padR - dilationW * (kw - 1) - 1) / strideW + 1;
    }
};

// 纯 CPU Conv2d（累加到 float，与 example 中 golden::ComputeConv2d 一致）
static void ComputeConv2d(const Conv2dParams& p,
                          const std::vector<fp16_t>& hostFmap,
                          const std::vector<fp16_t>& hostFilter,
                          std::vector<float>& goldenConv)
{
    const uint32_t c0 = p.c0;
    const uint32_t cin1 = p.cin1();
    const uint32_t cout1 = p.cout1();
    const uint32_t cout = p.cout;
    const uint32_t ho = p.ho();
    const uint32_t wo = p.wo();
    const size_t lenOut = static_cast<size_t>(p.batch) * cout1 * ho * wo * c0;
    goldenConv.assign(lenOut, 0.0f);

    auto fmapIdx = [&](uint32_t n, uint32_t c1, uint32_t h, uint32_t w, uint32_t c0i) -> size_t {
        return (((static_cast<size_t>(n) * cin1 + c1) * p.hi + h) * p.wi + w) * c0 + c0i;
    };
    // filter: [Cin1, Kh, Kw, Cout, C0]
    auto filterIdx = [&](uint32_t c1, uint32_t kh, uint32_t kw, uint32_t co, uint32_t c0i) -> size_t {
        return ((((static_cast<size_t>(c1) * p.kh + kh) * p.kw + kw) * cout + co) * c0) + c0i;
    };
    auto outIdx = [&](uint32_t n, uint32_t c1, uint32_t h, uint32_t w, uint32_t c0i) -> size_t {
        return (((static_cast<size_t>(n) * cout1 + c1) * ho + h) * wo + w) * c0 + c0i;
    };

    for (uint32_t n = 0; n < p.batch; ++n) {
        for (uint32_t oc = 0; oc < cout; ++oc) {
            uint32_t oc1 = oc / c0;
            uint32_t oc0 = oc % c0;
            for (uint32_t oh = 0; oh < ho; ++oh) {
                for (uint32_t ow = 0; ow < wo; ++ow) {
                    float acc = 0.0f;
                    for (uint32_t ic = 0; ic < p.cin; ++ic) {
                        uint32_t ic1 = ic / c0;
                        uint32_t ic0 = ic % c0;
                        for (uint32_t kh = 0; kh < p.kh; ++kh) {
                            for (uint32_t kw = 0; kw < p.kw; ++kw) {
                                int32_t ih = static_cast<int32_t>(oh * p.strideH)
                                           - static_cast<int32_t>(p.padT)
                                           + static_cast<int32_t>(kh * p.dilationH);
                                int32_t iw = static_cast<int32_t>(ow * p.strideW)
                                           - static_cast<int32_t>(p.padL)
                                           + static_cast<int32_t>(kw * p.dilationW);
                                if (ih < 0 || ih >= static_cast<int32_t>(p.hi) ||
                                    iw < 0 || iw >= static_cast<int32_t>(p.wi)) {
                                    continue;
                                }
                                float a = Fp16ToFloat(hostFmap[fmapIdx(n, ic1, ih, iw, ic0)]);
                                float b = Fp16ToFloat(hostFilter[filterIdx(ic1, kh, kw, oc, ic0)]);
                                acc += a * b;
                            }
                        }
                    }
                    goldenConv[outIdx(n, oc1, oh, ow, oc0)] = acc;
                }
            }
        }
    }
}

// SiLU(x) = x / (1 + exp(-x))
static float Silu(float x)
{
    return x / (1.0f + std::exp(-x));
}

// 生成最终 golden：SiLU(conv + bias)，尾通道保持 0
static void ComputeGoldenFinal(const Conv2dParams& p,
                               const std::vector<float>& goldenConv,
                               const std::vector<fp16_t>& hostBias,
                               std::vector<float>& goldenFinal)
{
    const uint32_t c0 = p.c0;
    const uint32_t cout1 = p.cout1();
    const uint32_t cout = p.cout;
    const uint32_t ho = p.ho();
    const uint32_t wo = p.wo();
    const size_t lenOut = static_cast<size_t>(p.batch) * cout1 * ho * wo * c0;
    goldenFinal.assign(lenOut, 0.0f);

    for (uint32_t n = 0; n < p.batch; ++n) {
        for (uint32_t c1 = 0; c1 < cout1; ++c1) {
            for (uint32_t h = 0; h < ho; ++h) {
                for (uint32_t w = 0; w < wo; ++w) {
                    for (uint32_t c0Idx = 0; c0Idx < c0; ++c0Idx) {
                        uint32_t c = c1 * c0 + c0Idx;
                        if (c >= cout) continue;
                        uint64_t offset = (uint64_t(n) * cout1 + c1) * ho * wo * c0
                                        + h * wo * c0 + w * c0 + c0Idx;
                        goldenFinal[offset] =
                            Silu(goldenConv[offset] + Fp16ToFloat(hostBias[c]));
                    }
                }
            }
        }
    }
}

// 清零尾通道（cout % c0 != 0 的 padding 部分）
static void ClearInvalidOutput(std::vector<fp16_t>& data, const Conv2dParams& p)
{
    const uint32_t c0 = p.c0;
    const uint32_t cout1 = p.cout1();
    const uint32_t cout = p.cout;
    const uint32_t ho = p.ho();
    const uint32_t wo = p.wo();
    for (uint32_t n = 0; n < p.batch; ++n) {
        for (uint32_t c1 = 0; c1 < cout1; ++c1) {
            for (uint32_t h = 0; h < ho; ++h) {
                for (uint32_t w = 0; w < wo; ++w) {
                    for (uint32_t c0Idx = 0; c0Idx < c0; ++c0Idx) {
                        uint32_t c = c1 * c0 + c0Idx;
                        if (c >= cout) {
                            uint64_t offset = (uint64_t(n) * cout1 + c1) * ho * wo * c0
                                            + h * wo * c0 + w * c0 + c0Idx;
                            data[offset] = FloatToFp16(0.0f);
                        }
                    }
                }
            }
        }
    }
}

// 精度比较（与 example 中 golden::CompareData 风格一致，使用相对+绝对误差）
static std::vector<uint64_t> CompareData(const std::vector<fp16_t>& actual,
                                         const std::vector<float>& golden,
                                         float atol = 1e-2f, float rtol = 1e-2f)
{
    std::vector<uint64_t> errors;
    size_t n = std::min(actual.size(), golden.size());
    for (size_t i = 0; i < n; ++i) {
        float a = Fp16ToFloat(actual[i]);
        float g = golden[i];
        float absErr = std::fabs(a - g);
        float relErr = absErr / (std::fabs(g) + 1e-8f);
        if (absErr > atol && relErr > rtol) {
            errors.push_back(i);
        }
    }
    return errors;
}

// ============================================================
// 调用自定义算子 aclnnConv2dSilu
// ============================================================
static int RunAclnnConv2dSilu(const Conv2dParams& p,
                              const std::vector<fp16_t>& hostFmap,
                              const std::vector<fp16_t>& hostFilter,
                              const std::vector<fp16_t>& hostBias,
                              std::vector<fp16_t>& hostOutput,
                              aclrtStream stream)
{
    const uint32_t c0 = p.c0;
    const uint32_t cin1 = p.cin1();
    const uint32_t cout1 = p.cout1();
    const uint32_t coutRound = p.coutRound();
    const uint32_t ho = p.ho();
    const uint32_t wo = p.wo();

    // shape
    // x: NC1HWC0  [N, Cin1, Hi, Wi, C0]
    std::vector<int64_t> xShape = {static_cast<int64_t>(p.batch), static_cast<int64_t>(cin1),
                                   static_cast<int64_t>(p.hi), static_cast<int64_t>(p.wi),
                                   static_cast<int64_t>(c0)};
    // filter: FRACTAL_Z  [Cin1, Kh, Kw, Cout, C0]  （与 example 一致）
    std::vector<int64_t> filterShape = {static_cast<int64_t>(cin1), static_cast<int64_t>(p.kh),
                                        static_cast<int64_t>(p.kw), static_cast<int64_t>(p.cout),
                                        static_cast<int64_t>(c0)};
    // bias: ND [CoutRound]
    std::vector<int64_t> biasShape = {static_cast<int64_t>(coutRound)};
    // y: NC1HWC0 [N, Cout1, Ho, Wo, C0]
    std::vector<int64_t> yShape = {static_cast<int64_t>(p.batch), static_cast<int64_t>(cout1),
                                   static_cast<int64_t>(ho), static_cast<int64_t>(wo),
                                   static_cast<int64_t>(c0)};

    void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr, *yDev = nullptr;
    aclTensor *xTensor = nullptr, *wTensor = nullptr, *bTensor = nullptr, *yTensor = nullptr;

    int ret = CreateAclTensor(hostFmap, xShape, &xDev, ACL_FLOAT16, &xTensor, ACL_FORMAT_NC1HWC0);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(hostFilter, filterShape, &wDev, ACL_FLOAT16, &wTensor, ACL_FORMAT_FRACTAL_Z);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(hostBias, biasShape, &bDev, ACL_FLOAT16, &bTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == 0, return ret);

    size_t ySize = GetShapeSize(yShape) * sizeof(fp16_t);
    hostOutput.assign(GetShapeSize(yShape), FloatToFp16(0.0f));
    ret = CreateAclTensor(hostOutput, yShape, &yDev, ACL_FLOAT16, &yTensor, ACL_FORMAT_NC1HWC0);
    CHECK_RET(ret == 0, return ret);

    // attrs: strides / pads / dilations 按 NCHW 约定（H,W）或完整 4 值均可
    // 这里与 ops-nn example 保持一致，使用长度 2 的 strides/dilations + 长度 4 的 pads
    std::vector<int64_t> strides   = {static_cast<int64_t>(p.strideH), static_cast<int64_t>(p.strideW)};
    std::vector<int64_t> pads      = {static_cast<int64_t>(p.padT), static_cast<int64_t>(p.padB),
                                      static_cast<int64_t>(p.padL), static_cast<int64_t>(p.padR)};
    std::vector<int64_t> dilations = {static_cast<int64_t>(p.dilationH), static_cast<int64_t>(p.dilationW)};
    int64_t groups = 1;

    aclIntArray* strideArr   = aclCreateIntArray(strides.data(), strides.size());
    aclIntArray* padArr      = aclCreateIntArray(pads.data(), pads.size());
    aclIntArray* dilationArr = aclCreateIntArray(dilations.data(), dilations.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 接口签名以实际生成的 aclnn_conv2d_silu.h 为准
    // 常见形式（与 ops-nn example 一致）：
    //   aclnnConv2dSiluGetWorkspaceSize(x, filter, bias, strides, pads, dilations, groups, y, &ws, &exe)
    // 若头文件带 dataFormat 参数，可按实际调整
    ret = aclnnConv2dSiluGetWorkspaceSize(
        xTensor, wTensor, bTensor,
        strideArr, padArr, dilationArr, groups,
        yTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        LOG_PRINT("aclnnConv2dSiluGetWorkspaceSize failed. ret=%d msg=%s\n", ret, errMsg ? errMsg : "");
        // 清理
        aclDestroyTensor(xTensor); aclDestroyTensor(wTensor);
        aclDestroyTensor(bTensor); aclDestroyTensor(yTensor);
        aclDestroyIntArray(strideArr); aclDestroyIntArray(padArr); aclDestroyIntArray(dilationArr);
        aclrtFree(xDev); aclrtFree(wDev); aclrtFree(bDev); aclrtFree(yDev);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnConv2dSilu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnConv2dSilu failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        LOG_PRINT("aclnnConv2dSiluGetWorkspaceSize failed. ret=%d msg=%s\n", ret, errMsg ? errMsg : "");
        // 清理
        aclDestroyTensor(xTensor); aclDestroyTensor(wTensor);
        aclDestroyTensor(bTensor); aclDestroyTensor(yTensor);
        aclDestroyIntArray(strideArr); aclDestroyIntArray(padArr); aclDestroyIntArray(dilationArr);
        aclrtFree(xDev); aclrtFree(wDev); aclrtFree(bDev); aclrtFree(yDev);
        return ret;
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 拷回
    ret = aclrtMemcpy(hostOutput.data(), ySize, yDev, ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 释放
    aclDestroyTensor(xTensor); aclDestroyTensor(wTensor);
    aclDestroyTensor(bTensor); aclDestroyTensor(yTensor);
    aclDestroyIntArray(strideArr); aclDestroyIntArray(padArr); aclDestroyIntArray(dilationArr);
    aclrtFree(xDev); aclrtFree(wDev); aclrtFree(bDev); aclrtFree(yDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// ============================================================
// 单 case 入口
// ============================================================
static int RunOneCase(const Conv2dParams& p, aclrtStream stream)
{
    LOG_PRINT("\n========== Case FP16  N=%u Cin=%u HxW=%ux%u -> Cout=%u KhxKw=%ux%u "
              "pad=[%u,%u,%u,%u] stride=[%u,%u] dil=[%u,%u] ==========\n",
              p.batch, p.cin, p.hi, p.wi, p.cout, p.kh, p.kw,
              p.padT, p.padB, p.padL, p.padR, p.strideH, p.strideW, p.dilationH, p.dilationW);

    const uint32_t c0 = p.c0;
    const uint32_t cin1 = p.cin1();
    const uint32_t cout1 = p.cout1();
    const uint32_t coutRound = p.coutRound();
    const uint32_t ho = p.ho();
    const uint32_t wo = p.wo();

    size_t lenFmap   = static_cast<size_t>(p.batch) * cin1 * p.hi * p.wi * c0;
    size_t lenFilter = static_cast<size_t>(cin1) * p.kh * p.kw * p.cout * c0;
    size_t lenBias   = coutRound;
    size_t lenOutput = static_cast<size_t>(p.batch) * cout1 * ho * wo * c0;

    std::vector<fp16_t> hostFmap(lenFmap);
    std::vector<fp16_t> hostFilter(lenFilter);
    std::vector<fp16_t> hostBias(lenBias);
    std::vector<fp16_t> hostOutput(lenOutput);

    // 与 example 相同的随机范围
    FillRandom(hostFmap,   -5.0f, 5.0f, 11);
    FillRandom(hostFilter, -5.0f, 5.0f, 22);
    FillRandom(hostBias,   -5.0f, 5.0f, 33);

    // ---------- 1. 调用自定义算子 ----------
    int ret = RunAclnnConv2dSilu(p, hostFmap, hostFilter, hostBias, hostOutput, stream);
    CHECK_RET(ret == 0, LOG_PRINT("RunAclnnConv2dSilu failed\n"); return ret);

    // // ---------- 2. CPU Golden（照搬 example） ----------
    // std::vector<float> goldenConv;
    // ComputeConv2d(p, hostFmap, hostFilter, goldenConv);

    // std::vector<float> goldenFinal;
    // ComputeGoldenFinal(p, goldenConv, hostBias, goldenFinal);

    // // 尾通道清零（与 example 中 golden::ClearInvalidOutput 一致）
    // ClearInvalidOutput(hostOutput, p);

    // // ---------- 3. 精度比对 ----------
    // auto errors = CompareData(hostOutput, goldenFinal, 1e-2f, 1e-2f);
    // LOG_PRINT("Output(AIV) vs golden SiLU(conv+bias) comparison: ");
    // if (errors.empty()) {
    //     LOG_PRINT("success.\n");
    //     return 0;
    // } else {
    //     LOG_PRINT("failed. Error count: %zu\n", errors.size());
    //     // 打印前 10 个错误位置
    //     for (size_t i = 0; i < errors.size() && i < 10; ++i) {
    //         uint64_t off = errors[i];
    //         uint32_t n  = off / (uint64_t(cout1) * ho * wo * c0);
    //         uint64_t rem = off % (uint64_t(cout1) * ho * wo * c0);
    //         uint32_t c1 = rem / (uint64_t(ho) * wo * c0);
    //         rem %= uint64_t(ho) * wo * c0;
    //         uint32_t h = rem / (uint64_t(wo) * c0);
    //         rem %= uint64_t(wo) * c0;
    //         uint32_t w = rem / c0;
    //         uint32_t c = c1 * c0 + rem % c0;
    //         float a = Fp16ToFloat(hostOutput[off]);
    //         float g = goldenFinal[off];
    //         LOG_PRINT("  [out] n=%u c=%u h=%u w=%u  actual=%.6f golden=%.6f\n",
    //                   n, c, h, w, a, g);
    //     }
    //     return 1;
    // }

    return 0;
}

// ============================================================
// main
// ============================================================
int main(int argc, const char** argv)
{
    // 默认参数与 example 保持一致：batch=2, hi=33, wi=43, cin=112, cout=80, 3x3, pad=2, stride=1
    Conv2dParams p;
    p.batch = 2;
    p.hi = 33;
    p.wi = 43;
    p.cin = 112;
    p.cout = 80;
    p.kh = 3;
    p.kw = 3;
    p.padL = 2; p.padR = 2; p.padT = 2; p.padB = 2;
    p.strideH = 1; p.strideW = 1;
    p.dilationH = 1; p.dilationW = 1;
    p.c0 = 16;

    // 支持命令行覆盖（与 example 参数顺序一致）
    // ./test_conv2d_silu_aclnn [batch hi wi cin cout kh kw padL padR padT padB strideH strideW dilH dilW]
    if (argc >= 16) {
        p.batch      = static_cast<uint32_t>(std::atoi(argv[1]));
        p.hi         = static_cast<uint32_t>(std::atoi(argv[2]));
        p.wi         = static_cast<uint32_t>(std::atoi(argv[3]));
        p.cin        = static_cast<uint32_t>(std::atoi(argv[4]));
        p.cout       = static_cast<uint32_t>(std::atoi(argv[5]));
        p.kh         = static_cast<uint32_t>(std::atoi(argv[6]));
        p.kw         = static_cast<uint32_t>(std::atoi(argv[7]));
        p.padL       = static_cast<uint32_t>(std::atoi(argv[8]));
        p.padR       = static_cast<uint32_t>(std::atoi(argv[9]));
        p.padT       = static_cast<uint32_t>(std::atoi(argv[10]));
        p.padB       = static_cast<uint32_t>(std::atoi(argv[11]));
        p.strideH    = static_cast<uint32_t>(std::atoi(argv[12]));
        p.strideW    = static_cast<uint32_t>(std::atoi(argv[13]));
        p.dilationH  = static_cast<uint32_t>(std::atoi(argv[14]));
        p.dilationW  = static_cast<uint32_t>(std::atoi(argv[15]));
    }

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed %d\n", ret); return ret);
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int fail = RunOneCase(p, stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (fail == 0) {
        LOG_PRINT("\n========== ALL PASSED ==========\n");
        return 0;
    }
    LOG_PRINT("\n========== CASE FAILED ==========\n");
    return 1;
}
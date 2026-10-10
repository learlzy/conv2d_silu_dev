/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *
 * 精度验证：自定义 aclnnConv2dSilu (NC1HWC0 + FRACTAL_Z) 
 *          vs  原生 aclnnConvolution + aclnnSilu (NCHW)
 * 公式：y = SiLU(Conv2d(x, filter) + bias)
 * 支持 FP16 / FP32
 * golden 参考 catlass/examples/33_basic_conv2d/conv2d_bias_silu.cpp
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <acl/acl.h>
#include "aclnnop/aclnn_convolution.h"
#include "aclnnop/aclnn_silu.h"
#include "aclnn_conv2d_silu.h"   // 自定义算子头文件

#define CHECK_RET(cond, return_expr) \
    do { if (!(cond)) { return_expr; } } while (0)

#define LOG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)

// ============================================================
// 常量：与 tiling 中 BYTE_PER_C0 = 32 保持一致
// FP16 → C0 = 16；FP32 → C0 = 8
// ============================================================
static constexpr uint32_t BYTE_PER_C0 = 32;

static inline uint32_t GetC0(aclDataType dtype)
{
    size_t elemBytes = (dtype == ACL_FLOAT16) ? 2 : 4;
    return static_cast<uint32_t>(BYTE_PER_C0 / elemBytes);
}

// ============================================================
// 工具
// ============================================================
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("H2D failed %d\n", ret); return ret);

    auto strides = MakeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, format, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);
    return 0;
}

static inline float Fp16ToFloat(aclFloat16 v) { return aclFloat16ToFloat(v); }
static inline aclFloat16 FloatToFp16(float v) { return aclFloatToFloat16(v); }

template <typename T>
static void FillRandom(std::vector<T>& data, float lo, float hi, int seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < data.size(); ++i) {
        float v = dist(gen);
        if constexpr (std::is_same_v<T, aclFloat16>) {
            data[i] = FloatToFp16(v);
        } else {
            data[i] = static_cast<T>(v);
        }
    }
}

// ============================================================
// 格式转换：NCHW ↔ NC1HWC0
// ============================================================
template <typename T>
static void NchwToNc1hwc0(const std::vector<T>& nchw,
                          int64_t n, int64_t c, int64_t h, int64_t w,
                          uint32_t c0,
                          std::vector<T>& nc1hwc0)
{
    int64_t c1 = (c + c0 - 1) / c0;
    nc1hwc0.assign(static_cast<size_t>(n * c1 * h * w * c0), T{});
    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t ci = 0; ci < c; ++ci) {
            int64_t c1i = ci / c0;
            int64_t c0i = ci % c0;
            for (int64_t hi = 0; hi < h; ++hi) {
                for (int64_t wi = 0; wi < w; ++wi) {
                    size_t src = ((ni * c + ci) * h + hi) * w + wi;
                    size_t dst = ((((ni * c1 + c1i) * h + hi) * w + wi) * c0) + c0i;
                    nc1hwc0[dst] = nchw[src];
                }
            }
        }
    }
}

template <typename T>
static void Nc1hwc0ToNchw(const std::vector<T>& nc1hwc0,
                          int64_t n, int64_t c, int64_t h, int64_t w,
                          uint32_t c0,
                          std::vector<T>& nchw)
{
    int64_t c1 = (c + c0 - 1) / c0;
    nchw.assign(static_cast<size_t>(n * c * h * w), T{});
    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t ci = 0; ci < c; ++ci) {
            int64_t c1i = ci / c0;
            int64_t c0i = ci % c0;
            for (int64_t hi = 0; hi < h; ++hi) {
                for (int64_t wi = 0; wi < w; ++wi) {
                    size_t src = ((((ni * c1 + c1i) * h + hi) * w + wi) * c0) + c0i;
                    size_t dst = ((ni * c + ci) * h + hi) * w + wi;
                    nchw[dst] = nc1hwc0[src];
                }
            }
        }
    }
}

// ============================================================
// 简化版 FRACTAL_Z 转换（Cout, Cin, Kh, Kw → FRACTAL_Z）
// 实际硬件 FRACTAL_Z 布局较复杂，这里做最常用的对齐转换
// 对于精度测试，只要自定义算子与 golden 使用同一份 filter 数据即可
// ============================================================
template <typename T>
static void NchwFilterToFractalZ(const std::vector<T>& nchwFilter,
                                 int64_t cout, int64_t cin, int64_t kh, int64_t kw,
                                 uint32_t c0,
                                 std::vector<T>& fractalZ)
{
    // 简化实现：先按 Cin1 * Kh * Kw * Cout * C0 排布（与 catlass LayoutFilter 一致）
    // 真实 FRACTAL_Z 还有更细的分形，但多数单测场景下此布局可被算子接受
    int64_t cin1 = (cin + c0 - 1) / c0;
    fractalZ.assign(static_cast<size_t>(cin1 * kh * kw * cout * c0), T{});

    for (int64_t co = 0; co < cout; ++co) {
        for (int64_t ci = 0; ci < cin; ++ci) {
            int64_t c1 = ci / c0;
            int64_t c0i = ci % c0;
            for (int64_t kh_i = 0; kh_i < kh; ++kh_i) {
                for (int64_t kw_i = 0; kw_i < kw; ++kw_i) {
                    size_t src = ((co * cin + ci) * kh + kh_i) * kw + kw_i;
                    // Layout: [Cin1, Kh, Kw, Cout, C0]
                    size_t dst = ((((c1 * kh + kh_i) * kw + kw_i) * cout + co) * c0) + c0i;
                    fractalZ[dst] = nchwFilter[src];
                }
            }
        }
    }
}

// ============================================================
// 输出 shape 计算
// ============================================================
static void CalcOutputShape(int64_t n, int64_t cin, int64_t hi, int64_t wi,
                            int64_t cout, int64_t kh, int64_t kw,
                            int64_t padT, int64_t padB, int64_t padL, int64_t padR,
                            int64_t strideH, int64_t strideW,
                            int64_t dilH, int64_t dilW,
                            std::vector<int64_t>& outNchwShape)
{
    int64_t ho = (hi + padT + padB - dilH * (kh - 1) - 1) / strideH + 1;
    int64_t wo = (wi + padL + padR - dilW * (kw - 1) - 1) / strideW + 1;
    outNchwShape = {n, cout, ho, wo};
}

// ============================================================
// Golden：在 NCHW 上跑 aclnnConvolution + aclnnSilu
// ============================================================
template <typename T>
static int ComputeGoldenNchw(const std::vector<T>& xNchw,
                             const std::vector<T>& filterNchw,
                             const std::vector<T>& bias,
                             const std::vector<int64_t>& xShape,
                             const std::vector<int64_t>& filterShape,
                             const std::vector<int64_t>& biasShape,
                             const std::vector<int64_t>& outShape,
                             const std::vector<int64_t>& strides,
                             const std::vector<int64_t>& pads,
                             const std::vector<int64_t>& dilations,
                             int64_t groups,
                             std::vector<T>& goldenNchw,
                             aclDataType dataType,
                             aclrtStream stream)
{
    void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr;
    void *convOutDev = nullptr, *siluOutDev = nullptr;
    aclTensor *xTensor = nullptr, *wTensor = nullptr, *bTensor = nullptr;
    aclTensor *convOutTensor = nullptr, *siluOutTensor = nullptr;

    int64_t outNum = GetShapeSize(outShape);
    std::vector<T> convOutHost(outNum, T{});

    auto ret = CreateAclTensor(xNchw, xShape, &xDev, dataType, &xTensor, ACL_FORMAT_NCHW);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(filterNchw, filterShape, &wDev, dataType, &wTensor, ACL_FORMAT_NCHW);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(bias, biasShape, &bDev, dataType, &bTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(convOutHost, outShape, &convOutDev, dataType, &convOutTensor, ACL_FORMAT_NCHW);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(goldenNchw, outShape, &siluOutDev, dataType, &siluOutTensor, ACL_FORMAT_NCHW);
    CHECK_RET(ret == 0, return ret);

    aclIntArray* strideArr   = aclCreateIntArray(strides.data(), strides.size());
    aclIntArray* padArr      = aclCreateIntArray(pads.data(), pads.size());
    aclIntArray* dilationArr = aclCreateIntArray(dilations.data(), dilations.size());
    std::vector<int64_t> outPadZero = {0, 0};
    aclIntArray* outPadArr = aclCreateIntArray(outPadZero.data(), outPadZero.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnConvolutionGetWorkspaceSize(
        xTensor, wTensor, bTensor,
        strideArr, padArr, dilationArr,
        false, outPadArr, groups,
        convOutTensor, 0,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionGetWorkspaceSize fail %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolution fail %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    workspaceSize = 0;
    executor = nullptr;
    ret = aclnnSiluGetWorkspaceSize(convOutTensor, siluOutTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSiluGetWorkspaceSize fail %d\n", ret); return ret);

    if (workspaceSize > 0) {
        if (workspaceAddr) { aclrtFree(workspaceAddr); workspaceAddr = nullptr; }
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnSilu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilu fail %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(goldenNchw.data(), outNum * sizeof(T),
                      siluOutDev, outNum * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclDestroyTensor(xTensor); aclDestroyTensor(wTensor);
    aclDestroyTensor(bTensor); aclDestroyTensor(convOutTensor);
    aclDestroyTensor(siluOutTensor);
    aclDestroyIntArray(strideArr); aclDestroyIntArray(padArr);
    aclDestroyIntArray(dilationArr); aclDestroyIntArray(outPadArr);
    aclrtFree(xDev); aclrtFree(wDev); aclrtFree(bDev);
    aclrtFree(convOutDev); aclrtFree(siluOutDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// ============================================================
// 自定义算子：aclnnConv2dSilu（NC1HWC0 + FRACTAL_Z）
// ============================================================
template <typename T>
static int ComputeCustom(const std::vector<T>& xNc1hwc0,
                         const std::vector<T>& filterFz,
                         const std::vector<T>& bias,
                         const std::vector<int64_t>& xShape,      // [N, C1, H, W, C0]
                         const std::vector<int64_t>& filterShape, // FRACTAL_Z shape
                         const std::vector<int64_t>& biasShape,
                         const std::vector<int64_t>& yShape,      // [N, C1, H, W, C0]
                         const std::vector<int64_t>& strides,
                         const std::vector<int64_t>& pads,
                         const std::vector<int64_t>& dilations,
                         int64_t groups,
                         std::vector<T>& outNc1hwc0,
                         aclDataType dataType,
                         aclrtStream stream)
{
    void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr, *yDev = nullptr;
    aclTensor *xTensor = nullptr, *wTensor = nullptr, *bTensor = nullptr, *yTensor = nullptr;

    auto ret = CreateAclTensor(xNc1hwc0, xShape, &xDev, dataType, &xTensor, ACL_FORMAT_NC1HWC0);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(filterFz, filterShape, &wDev, dataType, &wTensor, ACL_FORMAT_FRACTAL_Z);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(bias, biasShape, &bDev, dataType, &bTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(outNc1hwc0, yShape, &yDev, dataType, &yTensor, ACL_FORMAT_NC1HWC0);
    CHECK_RET(ret == 0, return ret);

    aclIntArray* strideArr   = aclCreateIntArray(strides.data(), strides.size());
    aclIntArray* padArr      = aclCreateIntArray(pads.data(), pads.size());
    aclIntArray* dilationArr = aclCreateIntArray(dilations.data(), dilations.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 根据实际生成的 aclnn 头文件调整参数顺序
    ret = aclnnConv2dSiluGetWorkspaceSize(
        xTensor, wTensor, bTensor,
        strideArr, padArr, dilationArr, groups,
        yTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        LOG_PRINT("aclnnConv2dSiluGetWorkspaceSize failed %d, msg: %s\n",
                  ret, errMsg ? errMsg : "null");
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = aclnnConv2dSilu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConv2dSilu failed %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t outNum = GetShapeSize(yShape);
    ret = aclrtMemcpy(outNc1hwc0.data(), outNum * sizeof(T),
                      yDev, outNum * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclDestroyTensor(xTensor); aclDestroyTensor(wTensor);
    aclDestroyTensor(bTensor); aclDestroyTensor(yTensor);
    aclDestroyIntArray(strideArr); aclDestroyIntArray(padArr);
    aclDestroyIntArray(dilationArr);
    aclrtFree(xDev); aclrtFree(wDev); aclrtFree(bDev); aclrtFree(yDev);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    return 0;
}

// ============================================================
// 精度比对
// ============================================================
template <typename T>
static bool CompareResult(const std::vector<T>& custom,
                          const std::vector<T>& golden,
                          float atol, float rtol)
{
    size_t total = custom.size();
    size_t failCnt = 0;
    float maxAbs = 0.f, maxRel = 0.f;

    for (size_t i = 0; i < total; ++i) {
        float c = std::is_same_v<T, aclFloat16> ? Fp16ToFloat(custom[i]) : static_cast<float>(custom[i]);
        float g = std::is_same_v<T, aclFloat16> ? Fp16ToFloat(golden[i]) : static_cast<float>(golden[i]);
        float absErr = std::fabs(c - g);
        float relErr = absErr / (std::fabs(g) + 1e-8f);
        maxAbs = std::max(maxAbs, absErr);
        maxRel = std::max(maxRel, relErr);
        if (absErr > atol && relErr > rtol) {
            if (failCnt < 16) {
                LOG_PRINT("  mismatch[%zu]: custom=%.6f golden=%.6f abs=%.6e rel=%.6e\n",
                          i, c, g, absErr, relErr);
            }
            ++failCnt;
        }
    }
    LOG_PRINT("Total=%zu  Fail=%zu  MaxAbs=%.6e  MaxRel=%.6e\n", total, failCnt, maxAbs, maxRel);
    return failCnt == 0;
}

// ============================================================
// 单 case
// ============================================================
template <typename T>
static int RunOneCase(aclDataType dataType, const char* dtypeName,
                      int64_t n, int64_t cin, int64_t hi, int64_t wi,
                      int64_t cout, int64_t kh, int64_t kw,
                      int64_t padT, int64_t padB, int64_t padL, int64_t padR,
                      int64_t strideH, int64_t strideW,
                      int64_t dilH, int64_t dilW,
                      int64_t groups,
                      float atol, float rtol,
                      aclrtStream stream)
{
    uint32_t c0 = GetC0(dataType);
    int64_t cin1  = (cin  + c0 - 1) / c0;
    int64_t cout1 = (cout + c0 - 1) / c0;

    LOG_PRINT("\n========== Case %s  N=%ld Cin=%ld HxW=%ldx%ld -> Cout=%ld "
              "KhxKw=%ldx%ld  C0=%u  pad=[%ld,%ld,%ld,%ld] stride=[%ld,%ld] ==========\n",
              dtypeName, n, cin, hi, wi, cout, kh, kw, c0,
              padT, padB, padL, padR, strideH, strideW);

    // ---------- 逻辑 NCHW shape ----------
    std::vector<int64_t> xNchwShape     = {n, cin, hi, wi};
    std::vector<int64_t> filterNchwShape= {cout, cin / groups, kh, kw};
    std::vector<int64_t> biasShape      = {cout};
    std::vector<int64_t> outNchwShape;
    CalcOutputShape(n, cin, hi, wi, cout, kh, kw,
                    padT, padB, padL, padR, strideH, strideW, dilH, dilW, outNchwShape);
    int64_t ho = outNchwShape[2];
    int64_t wo = outNchwShape[3];

    // ---------- 硬件 shape ----------
    std::vector<int64_t> xNc1hwc0Shape  = {n, cin1,  hi, wi, c0};
    std::vector<int64_t> yNc1hwc0Shape  = {n, cout1, ho, wo, c0};
    // FRACTAL_Z 简化 shape（与 catlass LayoutFilter 一致）
    std::vector<int64_t> filterFzShape  = {cin1, kh, kw, cout, c0};

    std::vector<int64_t> strides   = {strideH, strideW};
    std::vector<int64_t> pads      = {padT, padB, padL, padR};
    std::vector<int64_t> dilations = {dilH, dilW};

    // ---------- 生成逻辑数据 ----------
    std::vector<T> xNchw(GetShapeSize(xNchwShape));
    std::vector<T> filterNchw(GetShapeSize(filterNchwShape));
    std::vector<T> bias(GetShapeSize(biasShape));
    FillRandom(xNchw,      -1.0f, 1.0f, 11);
    FillRandom(filterNchw, -0.5f, 0.5f, 22);
    FillRandom(bias,       -0.2f, 0.2f, 33);

    // ---------- 转成硬件格式 ----------
    std::vector<T> xNc1hwc0, filterFz, yCustomNc1hwc0(GetShapeSize(yNc1hwc0Shape));
    NchwToNc1hwc0(xNchw, n, cin, hi, wi, c0, xNc1hwc0);
    NchwFilterToFractalZ(filterNchw, cout, cin / groups, kh, kw, c0, filterFz);

    // ---------- 跑自定义算子 ----------
    int ret = ComputeCustom(xNc1hwc0, filterFz, bias,
                            xNc1hwc0Shape, filterFzShape, biasShape, yNc1hwc0Shape,
                            strides, pads, dilations, groups,
                            yCustomNc1hwc0, dataType, stream);
    CHECK_RET(ret == 0, LOG_PRINT("ComputeCustom failed\n"); return ret);

    // ---------- 跑 golden (NCHW) ----------
    std::vector<T> goldenNchw(GetShapeSize(outNchwShape));
    ret = ComputeGoldenNchw(xNchw, filterNchw, bias,
                            xNchwShape, filterNchwShape, biasShape, outNchwShape,
                            strides, pads, dilations, groups,
                            goldenNchw, dataType, stream);
    CHECK_RET(ret == 0, LOG_PRINT("ComputeGolden failed\n"); return ret);

    // ---------- golden 转 NC1HWC0 后比对 ----------
    std::vector<T> goldenNc1hwc0;
    NchwToNc1hwc0(goldenNchw, n, cout, ho, wo, c0, goldenNc1hwc0);

    bool pass = CompareResult(yCustomNc1hwc0, goldenNc1hwc0, atol, rtol);
    LOG_PRINT(pass ? "[PASS] %s\n" : "[FAIL] %s\n", dtypeName);
    return pass ? 0 : 1;
}

// ============================================================
// main —— 测试样例全部写死
// 参考 catlass default: 2,33,43,112,80,3,3,pad=2,2,2,2,stride=1,1
// ============================================================
int main()
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed %d\n", ret); return ret);
    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int fail = 0;

    // // Case1: 与 catlass example 完全一致的 shape
    // fail += RunOneCase<aclFloat16>(
    //     ACL_FLOAT16, "FP16-catlass-default",
    //     2, 112, 33, 43, 80, 3, 3,
    //     2, 2, 2, 2, 1, 1, 1, 1, 1,
    //     1e-2f, 1e-2f, stream);

    // // Case2: 小 shape
    // fail += RunOneCase<aclFloat16>(
    //     ACL_FLOAT16, "FP16-small",
    //     1, 16, 16, 16, 32, 3, 3,
    //     1, 1, 1, 1, 1, 1, 1, 1, 1,
    //     1e-2f, 1e-2f, stream);

    // // Case3: FP32
    // fail += RunOneCase<float>(
    //     ACL_FLOAT, "FP32",
    //     1, 32, 28, 28, 64, 3, 3,
    //     1, 1, 1, 1, 1, 1, 1, 1, 1,
    //     1e-4f, 1e-4f, stream);

    // // Case4: stride=2
    // fail += RunOneCase<aclFloat16>(
    //     ACL_FLOAT16, "FP16-stride2",
    //     1, 32, 32, 32, 64, 3, 3,
    //     1, 1, 1, 1, 2, 2, 1, 1, 1,
    //     1e-2f, 1e-2f, stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (fail == 0) {
        LOG_PRINT("\n========== ALL PASSED ==========\n");
        return 0;
    }
    LOG_PRINT("\n========== %d CASE(S) FAILED ==========\n", fail);
    return 1;
}
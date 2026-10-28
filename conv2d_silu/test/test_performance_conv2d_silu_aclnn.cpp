/**
 * test_perf_conv2d_silu_aclnn.cpp
 *
 * 纯性能统计：只对 aclnnConv2dSilu 接口计时
 * - 可在代码内指定 warmup / test 轮次
 * - 统计平均耗时、最小/最大耗时
 *
 * 数据布局与 validation 一致：
 *   x/y   : NC1HWC0  (fp16)
 *   filter: FRACTAL_Z  (fp16)
 *   bias  : ND (CoutRound)  (fp16)
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>

#include "acl/acl.h"
#include "aclnn_conv2d_silu.h"

#include "catlass/conv_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "golden.hpp"

// ============================================================
// 可配置参数（在这里改）
// ============================================================
constexpr int WARMUP_ITERS = 10;   // 预热轮次
constexpr int TEST_ITERS   = 50;   // 正式测试轮次

// ============================================================
// 宏与工具
// ============================================================
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(fmt, ...)         \
    do {                            \
        printf(fmt, ##__VA_ARGS__); \
    } while (0)

#define ACL_CHECK_RET(ret)                                                    \
    do {                                                                      \
        auto _r = (ret);                                                      \
        if (_r != ACL_SUCCESS) {                                              \
            LOG_PRINT("ACL error %d at %s:%d\n", _r, __FILE__, __LINE__);     \
            return -1;                                                        \
        }                                                                     \
    } while (0)

using namespace Catlass;
using fp16_t = aclFloat16;

static inline fp16_t FloatToFp16(float v)
{
    return aclFloatToFloat16(v);
}

static int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
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
// 创建 aclTensor
// ============================================================
template <typename T>
static int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor, aclFormat format)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed. ERROR: %d\n", ret); return ret);

    auto strides = MakeStrides(shape);
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);
    return 0;
}

static void FillRandomFp16(std::vector<fp16_t>& data, float low, float high)
{
    std::vector<float> tmp(data.size());
    golden::FillRandomData<float>(tmp, low, high);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = FloatToFp16(tmp[i]);
    }
}

// ============================================================
// 性能统计：只计时 aclnnConv2dSilu（含 synchronize）
// ============================================================
static int RunPerfAclnnConv2dSilu(
    const Conv2dParams& p,
    const std::vector<fp16_t>& hostFmap,
    const std::vector<fp16_t>& hostFilter,
    const std::vector<fp16_t>& hostBias,
    aclrtStream stream,
    int warmupIters,
    int testIters)
{
    const uint32_t c0 = Conv2dParams::C0;
    const uint32_t cin1 = p.cin1();
    const uint32_t cout1 = p.cout1();
    const uint32_t coutRound = p.coutRound();
    const uint32_t ho = p.ho();
    const uint32_t wo = p.wo();
    const uint32_t kh = p.kh();
    const uint32_t kw = p.kw();

    // shapes
    std::vector<int64_t> xShape = {
        static_cast<int64_t>(p.batch()), static_cast<int64_t>(cin1), static_cast<int64_t>(p.hi()),
        static_cast<int64_t>(p.wi()), static_cast<int64_t>(c0)};
    std::vector<int64_t> filterShape = {
        static_cast<int64_t>(cin1), static_cast<int64_t>(kh), static_cast<int64_t>(kw),
        static_cast<int64_t>(p.cout()), static_cast<int64_t>(c0)};
    std::vector<int64_t> biasShape = {static_cast<int64_t>(coutRound)};
    std::vector<int64_t> yShape = {
        static_cast<int64_t>(p.batch()), static_cast<int64_t>(cout1), static_cast<int64_t>(ho),
        static_cast<int64_t>(wo), static_cast<int64_t>(c0)};

    void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr, *yDev = nullptr;
    aclTensor *xTensor = nullptr, *wTensor = nullptr, *bTensor = nullptr, *yTensor = nullptr;

    // 准备输入输出（只做一次）
    int ret = CreateAclTensor(hostFmap, xShape, &xDev, ACL_FLOAT16, &xTensor, ACL_FORMAT_NC1HWC0);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(hostFilter, filterShape, &wDev, ACL_FLOAT16, &wTensor, ACL_FORMAT_FRACTAL_Z);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(hostBias, biasShape, &bDev, ACL_FLOAT16, &bTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == 0, return ret);

    size_t ySize = static_cast<size_t>(GetShapeSize(yShape)) * sizeof(fp16_t);
    std::vector<fp16_t> hostOutput(static_cast<size_t>(GetShapeSize(yShape)), FloatToFp16(0.0f));
    ret = CreateAclTensor(hostOutput, yShape, &yDev, ACL_FLOAT16, &yTensor, ACL_FORMAT_NC1HWC0);
    CHECK_RET(ret == 0, return ret);

    std::vector<int64_t> strides = {static_cast<int64_t>(p.strideH()), static_cast<int64_t>(p.strideW())};
    std::vector<int64_t> pads = {
        static_cast<int64_t>(p.padTop()), static_cast<int64_t>(p.padBottom()),
        static_cast<int64_t>(p.padLeft()), static_cast<int64_t>(p.padRight())};
    std::vector<int64_t> dilations = {static_cast<int64_t>(p.dilationH()), static_cast<int64_t>(p.dilationW())};
    int64_t groups = 1;

    aclIntArray* strideArr = aclCreateIntArray(strides.data(), strides.size());
    aclIntArray* padArr = aclCreateIntArray(pads.data(), pads.size());
    aclIntArray* dilationArr = aclCreateIntArray(dilations.data(), dilations.size());

    // 第一段：GetWorkspaceSize（只做一次，不计入循环计时）
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnConv2dSiluGetWorkspaceSize(
        xTensor, wTensor, bTensor, strideArr, padArr, dilationArr, groups, yTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        LOG_PRINT("aclnnConv2dSiluGetWorkspaceSize failed. ret=%d msg=%s\n", ret, errMsg ? errMsg : "");
        // cleanup...
        aclDestroyTensor(xTensor);
        aclDestroyTensor(wTensor);
        aclDestroyTensor(bTensor);
        aclDestroyTensor(yTensor);
        aclDestroyIntArray(strideArr);
        aclDestroyIntArray(padArr);
        aclDestroyIntArray(dilationArr);
        aclrtFree(xDev);
        aclrtFree(wDev);
        aclrtFree(bDev);
        aclrtFree(yDev);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // -------------------- 预热 --------------------
    LOG_PRINT("Warmup: %d iters ...\n", warmupIters);
    for (int i = 0; i < warmupIters; ++i) {
        ret = aclnnConv2dSilu(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConv2dSilu (warmup) failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // -------------------- 正式计时 --------------------
    LOG_PRINT("Test:   %d iters ...\n", testIters);
    std::vector<double> timesMs;
    timesMs.reserve(testIters);

    for (int i = 0; i < testIters; ++i) {
        auto t0 = std::chrono::steady_clock::now();

        ret = aclnnConv2dSilu(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConv2dSilu failed. ERROR: %d\n", ret); return ret);
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timesMs.push_back(ms);
    }

    // -------------------- 统计结果 --------------------
    double sum = std::accumulate(timesMs.begin(), timesMs.end(), 0.0);
    double avg = sum / timesMs.size();
    double minT = *std::min_element(timesMs.begin(), timesMs.end());
    double maxT = *std::max_element(timesMs.begin(), timesMs.end());

    // 简单标准差
    double var = 0.0;
    for (double t : timesMs) {
        var += (t - avg) * (t - avg);
    }
    double stddev = std::sqrt(var / timesMs.size());

    LOG_PRINT("\n========== Performance Result ==========\n");
    LOG_PRINT("Case: N=%u Cin=%u HxW=%ux%u -> Cout=%u KhxKw=%ux%u\n",
              p.batch(), p.cin(), p.hi(), p.wi(), p.cout(), p.kh(), p.kw());
    LOG_PRINT("pad=[T%u B%u L%u R%u] stride=[%u,%u] dil=[%u,%u]\n",
              p.padTop(), p.padBottom(), p.padLeft(), p.padRight(),
              p.strideH(), p.strideW(), p.dilationH(), p.dilationW());
    LOG_PRINT("Warmup: %d, Test: %d\n", warmupIters, testIters);
    LOG_PRINT("Avg  : %.4f ms\n", avg);
    LOG_PRINT("Min  : %.4f ms\n", minT);
    LOG_PRINT("Max  : %.4f ms\n", maxT);
    LOG_PRINT("Std  : %.4f ms\n", stddev);
    LOG_PRINT("========================================\n\n");

    // cleanup
    aclDestroyTensor(xTensor);
    aclDestroyTensor(wTensor);
    aclDestroyTensor(bTensor);
    aclDestroyTensor(yTensor);
    aclDestroyIntArray(strideArr);
    aclDestroyIntArray(padArr);
    aclDestroyIntArray(dilationArr);
    aclrtFree(xDev);
    aclrtFree(wDev);
    aclrtFree(bDev);
    aclrtFree(yDev);
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

// ============================================================
// 单 case 入口
// ============================================================
static int RunOneCasePerf(const Conv2dParams& problemParams, aclrtStream stream)
{
    const uint32_t c0 = Conv2dParams::C0;
    const uint32_t batch = problemParams.batch();
    const uint32_t hi = problemParams.hi();
    const uint32_t wi = problemParams.wi();
    const uint32_t cin1 = problemParams.cin1();
    const uint32_t coutRound = problemParams.coutRound();
    const uint32_t kh = problemParams.kh();
    const uint32_t kw = problemParams.kw();
    const uint32_t cout = problemParams.cout();

    size_t lenFmap = static_cast<size_t>(batch) * cin1 * hi * wi * c0;
    size_t lenFilter = static_cast<size_t>(cin1) * kh * kw * cout * c0;
    size_t lenBias = coutRound;

    std::vector<fp16_t> hostFmap(lenFmap);
    std::vector<fp16_t> hostFilter(lenFilter);
    std::vector<fp16_t> hostBias(lenBias);

    FillRandomFp16(hostFmap, -5.0f, 5.0f);
    FillRandomFp16(hostFilter, -5.0f, 5.0f);
    FillRandomFp16(hostBias, -5.0f, 5.0f);

    return RunPerfAclnnConv2dSilu(
        problemParams, hostFmap, hostFilter, hostBias, stream, WARMUP_ITERS, TEST_ITERS);
}

// ============================================================
// main
// ============================================================
int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

    // 与 validation 相同的典型 case，可按需增删
    // 示例：N=2, Cin=112, H=33, W=43, Cout=80, K=3x3, pad=1, stride=1, dil=1
    {
        Conv2dParams p = Conv2dParams::Make(
            /*batch*/ 2,
            /*cin*/ 112,
            /*hi*/ 33,
            /*wi*/ 43,
            /*cout*/ 80,
            /*kh*/ 3,
            /*kw*/ 3,
            /*padTop*/ 1, /*padBottom*/ 1, /*padLeft*/ 1, /*padRight*/ 1,
            /*strideH*/ 1, /*strideW*/ 1,
            /*dilationH*/ 1, /*dilationW*/ 1);
        ret = RunOneCasePerf(p, stream);
        if (ret != 0) {
            LOG_PRINT("RunOneCasePerf failed\n");
        }
    }

    // 可以继续加更多 case ...
    // {
    //     Conv2dParams p = Conv2dParams::Make(...);
    //     RunOneCasePerf(p, stream);
    // }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
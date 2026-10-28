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
 * Golden / 填数 / 比较：复用 catlass/examples/common/golden.hpp
 *   （FillRandomData / ComputeConv2d / ClearInvalidOutput / CompareData / Silu）
 * 算子调用：本工程 aclnn 接口（aclnnConv2dSilu），不走 example 里的 DeviceConv kernel
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_conv2d_silu.h"

// catlass golden + Conv2dParams / layout（与 33_basic_conv2d 一致）
#include "catlass/conv_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "golden.hpp"

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

static inline float Fp16ToFloat(fp16_t v)
{
    return aclFloat16ToFloat(v);
}
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
// 创建 aclTensor（支持 NC1HWC0 / FRACTAL_Z / ND）
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

// ============================================================
// 用 golden.hpp 的 FillRandomData 填 fp16（内部按 float 随机再转换）
// ============================================================
static void FillRandomFp16(std::vector<fp16_t>& data, float low, float high)
{
    std::vector<float> tmp(data.size());
    golden::FillRandomData<float>(tmp, low, high);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = FloatToFp16(tmp[i]);
    }
}

// ============================================================
// 调用自定义算子 aclnnConv2dSilu
// ============================================================
static int RunAclnnConv2dSilu(
    const Conv2dParams& p, const std::vector<fp16_t>& hostFmap, const std::vector<fp16_t>& hostFilter,
    const std::vector<fp16_t>& hostBias, std::vector<fp16_t>& hostOutput, aclrtStream stream)
{
    const uint32_t c0 = Conv2dParams::C0;
    const uint32_t cin1 = p.cin1();
    const uint32_t cout1 = p.cout1();
    const uint32_t coutRound = p.coutRound();
    const uint32_t ho = p.ho();
    const uint32_t wo = p.wo();
    const uint32_t kh = p.kh();
    const uint32_t kw = p.kw();

    // x: NC1HWC0  [N, Cin1, Hi, Wi, C0]
    std::vector<int64_t> xShape = {
        static_cast<int64_t>(p.batch()), static_cast<int64_t>(cin1), static_cast<int64_t>(p.hi()),
        static_cast<int64_t>(p.wi()), static_cast<int64_t>(c0)};
    // filter: FRACTAL_Z  [Cin1, Kh, Kw, Cout, C0]（与 example 一致）
    std::vector<int64_t> filterShape = {
        static_cast<int64_t>(cin1), static_cast<int64_t>(kh), static_cast<int64_t>(kw),
        static_cast<int64_t>(p.cout()), static_cast<int64_t>(c0)};
    // bias: ND [CoutRound]
    std::vector<int64_t> biasShape = {static_cast<int64_t>(coutRound)};
    // y: NC1HWC0 [N, Cout1, Ho, Wo, C0]
    std::vector<int64_t> yShape = {
        static_cast<int64_t>(p.batch()), static_cast<int64_t>(cout1), static_cast<int64_t>(ho),
        static_cast<int64_t>(wo), static_cast<int64_t>(c0)};

    void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr, *yDev = nullptr;
    aclTensor *xTensor = nullptr, *wTensor = nullptr, *bTensor = nullptr, *yTensor = nullptr;

    int ret = CreateAclTensor(hostFmap, xShape, &xDev, ACL_FLOAT16, &xTensor, ACL_FORMAT_NC1HWC0);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(hostFilter, filterShape, &wDev, ACL_FLOAT16, &wTensor, ACL_FORMAT_FRACTAL_Z);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(hostBias, biasShape, &bDev, ACL_FLOAT16, &bTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == 0, return ret);

    size_t ySize = static_cast<size_t>(GetShapeSize(yShape)) * sizeof(fp16_t);
    hostOutput.assign(static_cast<size_t>(GetShapeSize(yShape)), FloatToFp16(0.0f));
    ret = CreateAclTensor(hostOutput, yShape, &yDev, ACL_FLOAT16, &yTensor, ACL_FORMAT_NC1HWC0);
    CHECK_RET(ret == 0, return ret);

    // pads: top, bottom, left, right（若头文件约定不同，按实际 aclnn 签名改）
    std::vector<int64_t> strides = {static_cast<int64_t>(p.strideH()), static_cast<int64_t>(p.strideW())};
    std::vector<int64_t> pads = {
        static_cast<int64_t>(p.padTop()), static_cast<int64_t>(p.padBottom()),
        static_cast<int64_t>(p.padLeft()), static_cast<int64_t>(p.padRight())};
    std::vector<int64_t> dilations = {static_cast<int64_t>(p.dilationH()), static_cast<int64_t>(p.dilationW())};
    int64_t groups = 1;

    aclIntArray* strideArr = aclCreateIntArray(strides.data(), strides.size());
    aclIntArray* padArr = aclCreateIntArray(pads.data(), pads.size());
    aclIntArray* dilationArr = aclCreateIntArray(dilations.data(), dilations.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 本工程 aclnn 接口（勿使用 example 中的 DeviceConv kernel）
    ret = aclnnConv2dSiluGetWorkspaceSize(
        xTensor, wTensor, bTensor, strideArr, padArr, dilationArr, groups, yTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        LOG_PRINT("aclnnConv2dSiluGetWorkspaceSize failed. ret=%d msg=%s\n", ret, errMsg ? errMsg : "");
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

    ret = aclnnConv2dSilu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConv2dSilu failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        const char* errMsg = aclGetRecentErrMsg();
        LOG_PRINT("aclrtSynchronizeStream failed. ret=%d msg=%s\n", ret, errMsg ? errMsg : "");
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
        return ret;
    }

    ret = aclrtMemcpy(hostOutput.data(), ySize, yDev, ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

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
// 单 case：复用 golden.hpp + example 验证流程，算子走 aclnn
// ============================================================
static int RunOneCase(const Conv2dParams& problemParams, aclrtStream stream)
{
    const uint32_t c0 = Conv2dParams::C0;
    const uint32_t batch = problemParams.batch();
    const uint32_t hi = problemParams.hi();
    const uint32_t wi = problemParams.wi();
    const uint32_t cin1 = problemParams.cin1();
    const uint32_t ho = problemParams.ho();
    const uint32_t wo = problemParams.wo();
    const uint32_t cout1 = problemParams.cout1();
    const uint32_t cout = problemParams.cout();
    const uint32_t coutRound = problemParams.coutRound();
    const uint32_t kh = problemParams.kh();
    const uint32_t kw = problemParams.kw();

    LOG_PRINT(
        "\n========== Case FP16  N=%u Cin=%u HxW=%ux%u -> Cout=%u KhxKw=%ux%u "
        "pad=[T%u B%u L%u R%u] stride=[%u,%u] dil=[%u,%u] ==========\n",
        batch, problemParams.cin(), hi, wi, cout, kh, kw, problemParams.padTop(), problemParams.padBottom(),
        problemParams.padLeft(), problemParams.padRight(), problemParams.strideH(), problemParams.strideW(),
        problemParams.dilationH(), problemParams.dilationW());

    size_t lenFmap = static_cast<size_t>(batch) * cin1 * hi * wi * c0;
    size_t lenFilter = static_cast<size_t>(cin1) * kh * kw * cout * c0;
    size_t lenBias = coutRound;
    size_t lenOutput = static_cast<size_t>(batch) * cout1 * ho * wo * c0;

    using LayoutFmap = layout::NC1HWC0;
    using LayoutFilter = layout::CI1KHKWCOCI0;
    using LayoutOutput = layout::NC1HWC0;
    LayoutFmap layoutFmap{batch, cin1, hi, wi, c0};
    LayoutFilter layoutFilter{cin1, kh, kw, cout, c0};
    LayoutOutput layoutOutput{batch, cout1, ho, wo, c0};

    std::vector<fp16_t> hostFmap(lenFmap);
    std::vector<fp16_t> hostFilter(lenFilter);
    std::vector<fp16_t> hostBias(lenBias);
    std::vector<fp16_t> hostOutput(lenOutput);

    // 与 example 相同的随机范围，走 golden::FillRandomData
    FillRandomFp16(hostFmap, -5.0f, 5.0f);
    FillRandomFp16(hostFilter, -5.0f, 5.0f);
    FillRandomFp16(hostBias, -5.0f, 5.0f);

    // ---------- 1. 调用本工程 aclnn 算子 ----------
    int ret = RunAclnnConv2dSilu(problemParams, hostFmap, hostFilter, hostBias, hostOutput, stream);
    CHECK_RET(ret == 0, LOG_PRINT("RunAclnnConv2dSilu failed\n"); return ret);

    // ---------- 2. CPU Golden（照搬 conv2d_bias_silu.cpp，接口来自 golden.hpp） ----------
    // ComputeConv2d 需要 Element 可转 float；先转成 float 再喂 golden，避免 aclFloat16 与模板不匹配
    std::vector<float> hostFmapF(lenFmap);
    std::vector<float> hostFilterF(lenFilter);
    for (size_t i = 0; i < lenFmap; ++i) {
        hostFmapF[i] = Fp16ToFloat(hostFmap[i]);
    }
    for (size_t i = 0; i < lenFilter; ++i) {
        hostFilterF[i] = Fp16ToFloat(hostFilter[i]);
    }

    std::vector<float> goldenConv(lenOutput, 0.0f);
    golden::ComputeConv2d(
        problemParams, hostFmapF, layoutFmap, hostFilterF, layoutFilter, goldenConv, layoutOutput);

    // goldenFinal = SiLU(conv + bias)；尾通道 (cout % C0 != 0) 保持 0
    std::vector<float> goldenFinal(lenOutput, 0.0f);
    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t c1 = 0; c1 < cout1; ++c1) {
            for (uint32_t h = 0; h < ho; ++h) {
                for (uint32_t w = 0; w < wo; ++w) {
                    for (uint32_t c0Idx = 0; c0Idx < c0; ++c0Idx) {
                        uint32_t c = c1 * c0 + c0Idx;
                        if (c >= cout) {
                            continue;
                        }
                        uint64_t offset =
                            (uint64_t(n) * cout1 + c1) * ho * wo * c0 + h * wo * c0 + w * c0 + c0Idx;
                        goldenFinal[offset] =
                            golden::Silu<float>{}(goldenConv[offset] + Fp16ToFloat(hostBias[c]));
                    }
                }
            }
        }
    }

    // 尾通道清零（与 example 中 golden::ClearInvalidOutput 一致）
    std::vector<float> hostOutF(lenOutput);
    for (size_t i = 0; i < lenOutput; ++i) {
        hostOutF[i] = Fp16ToFloat(hostOutput[i]);
    }
    golden::ClearInvalidOutput(hostOutF, problemParams);

    // ---------- 3. 比较（golden::CompareData，computeNum = cin1*kh*kw*c0） ----------
    std::vector<uint64_t> outErrors =
        golden::CompareData(hostOutF, goldenFinal, cin1 * kh * kw * c0);

    auto printErrors = [&](const std::vector<uint64_t>& errors, const char* tag) {
        for (uint64_t i = 0; i < errors.size() && i < 10; ++i) {
            uint64_t off = errors[i];
            uint32_t n = static_cast<uint32_t>(off / (uint64_t(cout1) * ho * wo * c0));
            uint64_t rem = off % (uint64_t(cout1) * ho * wo * c0);
            uint32_t c1 = static_cast<uint32_t>(rem / (uint64_t(ho) * wo * c0));
            rem %= uint64_t(ho) * wo * c0;
            uint32_t h = static_cast<uint32_t>(rem / (uint64_t(wo) * c0));
            rem %= uint64_t(wo) * c0;
            uint32_t w = static_cast<uint32_t>(rem / c0);
            uint32_t c = c1 * c0 + static_cast<uint32_t>(rem % c0);
            float a = hostOutF[off];
            float g = goldenFinal[off];
            LOG_PRINT("  [%s] n=%u c=%u h=%u w=%u  actual=%.6f golden=%.6f\n", tag, n, c, h, w, a, g);
        }
    };

    LOG_PRINT("Output(aclnn) vs golden SiLU(conv+bias) comparison: ");
    if (outErrors.empty()) {
        LOG_PRINT("success.\n");
        return 0;
    }
    LOG_PRINT("failed. Error count: %zu\n", outErrors.size());
    printErrors(outErrors, "out");
    return 1;
}

// ============================================================
// main
// ============================================================
int main(int argc, const char** argv)
{
    // 默认参数与 example 保持一致
    uint32_t dataSizes[5] = {2, 33, 43, 112, 80}; // {batch, hi, wi, cin, cout}
    uint8_t filterSizes[2] = {3, 3};              // {kh, kw}
    uint8_t pads[4] = {2, 2, 2, 2};               // {padLeft, padRight, padTop, padBottom}
    uint8_t strides[2] = {1, 1};                  // {strideH, strideW}
    uint8_t dilations[2] = {1, 1};                // {dilationH, dilationW}
    int32_t deviceId = 0;

    // uint32_t dataSizes[5] = {8, 640, 640, 3, 32}; // {batch, hi, wi, cin, cout}
    // uint8_t filterSizes[2] = {3, 3};              // {kh, kw}
    // uint8_t pads[4] = {1, 1, 1, 1};               // {padLeft, padRight, padTop, padBottom}
    // uint8_t strides[2] = {2, 2};                  // {strideH, strideW}
    // uint8_t dilations[2] = {1, 1};                // {dilationH, dilationW}
    // int32_t deviceId = 0;

    // ./test_conv2d_silu_aclnn [batch hi wi cin cout kh kw padL padR padT padB strideH strideW dilH dilW] [device_id]
    if (argc >= 16) {
        dataSizes[0] = static_cast<uint32_t>(std::atoi(argv[1]));
        dataSizes[1] = static_cast<uint32_t>(std::atoi(argv[2]));
        dataSizes[2] = static_cast<uint32_t>(std::atoi(argv[3]));
        dataSizes[3] = static_cast<uint32_t>(std::atoi(argv[4]));
        dataSizes[4] = static_cast<uint32_t>(std::atoi(argv[5]));
        filterSizes[0] = static_cast<uint8_t>(std::atoi(argv[6]));
        filterSizes[1] = static_cast<uint8_t>(std::atoi(argv[7]));
        pads[0] = static_cast<uint8_t>(std::atoi(argv[8]));
        pads[1] = static_cast<uint8_t>(std::atoi(argv[9]));
        pads[2] = static_cast<uint8_t>(std::atoi(argv[10]));
        pads[3] = static_cast<uint8_t>(std::atoi(argv[11]));
        strides[0] = static_cast<uint8_t>(std::atoi(argv[12]));
        strides[1] = static_cast<uint8_t>(std::atoi(argv[13]));
        dilations[0] = static_cast<uint8_t>(std::atoi(argv[14]));
        dilations[1] = static_cast<uint8_t>(std::atoi(argv[15]));
    }
    if (argc >= 17) {
        deviceId = std::atoi(argv[16]);
    }

    Conv2dParams problemParams =
        Conv2dParams::MakeConv2dParams(dataSizes, filterSizes, pads, strides, dilations);

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int fail = RunOneCase(problemParams, stream);

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
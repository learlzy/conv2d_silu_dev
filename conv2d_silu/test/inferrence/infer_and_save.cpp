/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * infer_and_save.cpp
 *
 * 功能：硬编码raw路径加载输入，执行 aclnnConv2dSilu，dump输出与workspace raw
 * 无golden精度比对，不使用goto
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "acl/acl.h"
#include "aclnn_conv2d_silu.h"
#include "catlass/conv_coord.hpp"
#include "catlass/layout/layout.hpp"

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

#define ACL_CHECK_RET(ret)                                                      \
    do {                                                                        \
        auto _r = (ret);                                                        \
        if (_r != ACL_SUCCESS) {                                                \
            LOG_PRINT("ACL error %d at %s:%d\n", _r, __FILE__, __LINE__);       \
            return -1;                                                          \
        }                                                                       \
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

static bool LoadFp16Raw(const std::string& path, std::vector<fp16_t>& out)
{
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        LOG_PRINT("[ERROR] cannot open %s\n", path.c_str());
        return false;
    }
    fin.read(reinterpret_cast<char*>(out.data()), out.size() * sizeof(fp16_t));
    if (!fin) {
        LOG_PRINT("[ERROR] read %s incomplete\n", path.c_str());
        return false;
    }
    fin.close();
    return true;
}

static bool SaveFp16Raw(const std::string& path, const std::vector<fp16_t>& data)
{
    std::ofstream fout(path, std::ios::binary);
    if (!fout.is_open()) {
        LOG_PRINT("[ERROR] cannot write %s\n", path.c_str());
        return false;
    }
    fout.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(fp16_t));
    fout.close();
    LOG_PRINT("[INFO] saved fp16 raw: %s, elem=%zu\n", path.c_str(), data.size());
    return true;
}

static bool SaveBytesRaw(const std::string& path, const void* src, size_t byteSize)
{
    if (byteSize == 0) {
        LOG_PRINT("[INFO] workspace size=0, skip save %s\n", path.c_str());
        return true;
    }
    std::ofstream fout(path, std::ios::binary);
    if (!fout.is_open()) {
        LOG_PRINT("[ERROR] cannot write %s\n", path.c_str());
        return false;
    }
    fout.write(reinterpret_cast<const char*>(src), byteSize);
    fout.close();
    LOG_PRINT("[INFO] saved workspace raw: %s, bytes=%zu\n", path.c_str(), byteSize);
    return true;
}

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

static int RunAclnnConv2dSilu(
    const Conv2dParams& p,
    std::vector<fp16_t>& hostFmap,
    std::vector<fp16_t>& hostFilter,
    std::vector<fp16_t>& hostBias,
    std::vector<fp16_t>& hostOutput,
    aclrtStream stream,
    const std::string& outRawPath,
    const std::string& wsRawPath)
{
    const uint32_t c0 = Conv2dParams::C0;
    const uint32_t cin1 = p.cin1();
    const uint32_t cout1 = p.cout1();
    const uint32_t coutRound = p.coutRound();
    const uint32_t ho = p.ho();
    const uint32_t wo = p.wo();
    const uint32_t kh = p.kh();
    const uint32_t kw = p.kw();

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
    void* workspaceAddr = nullptr;

    std::vector<int64_t> strides = {static_cast<int64_t>(p.strideH()), static_cast<int64_t>(p.strideW())};
    std::vector<int64_t> pads = {
        static_cast<int64_t>(p.padTop()), static_cast<int64_t>(p.padBottom()),
        static_cast<int64_t>(p.padLeft()), static_cast<int64_t>(p.padRight())};
    std::vector<int64_t> dilations = {static_cast<int64_t>(p.dilationH()), static_cast<int64_t>(p.dilationW())};
    int64_t groups = 1;

    aclIntArray* strideArr = nullptr;
    aclIntArray* padArr = nullptr;
    aclIntArray* dilationArr = nullptr;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = 0;

    do {
        ret = CreateAclTensor(hostFmap, xShape, &xDev, ACL_FLOAT16, &xTensor, ACL_FORMAT_NC1HWC0);
        if (ret != 0) break;

        ret = CreateAclTensor(hostFilter, filterShape, &wDev, ACL_FLOAT16, &wTensor, ACL_FORMAT_FRACTAL_Z);
        if (ret != 0) break;

        ret = CreateAclTensor(hostBias, biasShape, &bDev, ACL_FLOAT16, &bTensor, ACL_FORMAT_ND);
        if (ret != 0) break;

        size_t ySize = static_cast<size_t>(GetShapeSize(yShape)) * sizeof(fp16_t);
        hostOutput.assign(static_cast<size_t>(GetShapeSize(yShape)), FloatToFp16(0.0f));
        ret = CreateAclTensor(hostOutput, yShape, &yDev, ACL_FLOAT16, &yTensor, ACL_FORMAT_NC1HWC0);
        if (ret != 0) break;

        strideArr = aclCreateIntArray(strides.data(), strides.size());
        padArr = aclCreateIntArray(pads.data(), pads.size());
        dilationArr = aclCreateIntArray(dilations.data(), dilations.size());

        ret = aclnnConv2dSiluGetWorkspaceSize(
            xTensor, wTensor, bTensor, strideArr, padArr, dilationArr, groups, yTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            const char* errMsg = aclGetRecentErrMsg();
            LOG_PRINT("aclnnConv2dSiluGetWorkspaceSize failed. ret=%d msg=%s\n", ret, errMsg ? errMsg : "");
            break;
        }

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) break;
        }

        ret = aclnnConv2dSilu(workspaceAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclnnConv2dSilu failed. ERROR: %d\n", ret);
            break;
        }

        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            const char* errMsg = aclGetRecentErrMsg();
            LOG_PRINT("aclrtSynchronizeStream failed. ret=%d msg=%s\n", ret, errMsg ? errMsg : "");
            break;
        }

        ret = aclrtMemcpy(hostOutput.data(), ySize, yDev, ySize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) break;
        SaveFp16Raw(outRawPath, hostOutput);

        if (workspaceSize > 0) {
            std::vector<char> wsHostBuf(workspaceSize);
            ret = aclrtMemcpy(wsHostBuf.data(), workspaceSize, workspaceAddr, workspaceSize, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) break;
            SaveBytesRaw(wsRawPath, wsHostBuf.data(), workspaceSize);
        }

    } while (false);

    // 资源释放，一定会执行
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

static int RunOneCase(const Conv2dParams& problemParams,
                      aclrtStream stream,
                      const std::string& fmapPath,
                      const std::string& filterPath,
                      const std::string& biasPath,
                      const std::string& outPath,
                      const std::string& wsPath)
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

    LOG_PRINT("[INFO] fmap elem: %zu, filter elem: %zu, bias elem: %zu, output elem: %zu\n",
        lenFmap, lenFilter, lenBias, lenOutput);

    std::vector<fp16_t> hostFmap(lenFmap);
    std::vector<fp16_t> hostFilter(lenFilter);
    std::vector<fp16_t> hostBias(lenBias);
    std::vector<fp16_t> hostOutput(lenOutput);

    if (!LoadFp16Raw(fmapPath, hostFmap)) {
        LOG_PRINT("[ERROR] load %s fail\n", fmapPath.c_str());
        return -1;
    }
    if (!LoadFp16Raw(filterPath, hostFilter)) {
        LOG_PRINT("[ERROR] load %s fail\n", filterPath.c_str());
        return -1;
    }
    if (!LoadFp16Raw(biasPath, hostBias)) {
        LOG_PRINT("[ERROR] load %s fail\n", biasPath.c_str());
        return -1;
    }

    int ret = RunAclnnConv2dSilu(problemParams, hostFmap, hostFilter, hostBias, hostOutput,
                                 stream, outPath, wsPath);
    if (ret != 0) {
        LOG_PRINT("[ERROR] RunAclnnConv2dSilu failed\n");
    }
    return ret;
}

int main(int argc, const char** argv)
{
    // ===================== 硬编码raw路径，在这里修改 =====================
    std::string file_dir      = "/mnt/workspace/conv2d_silu_dev/test/build/20261102095418/0/yolov8s_subgraph_base/1/0"; 
    std::string fmapRawPath   = file_dir + "/Conv2D._model_0_conv_Conv.2.45.1793584458686230.input.0.raw";
    std::string filterRawPath = file_dir + "/Conv2D._model_0_conv_Conv.2.45.1793584458686230.input.1.raw";
    std::string biasRawPath   = file_dir + "/Conv2D._model_0_conv_Conv.2.45.1793584458686230.input.2.raw";
    std::string outRawPath    = file_dir + "/out.raw";
    std::string wsRawPath     = file_dir + "/workspace.raw";
    // =====================================================================

    uint32_t dataSizes[5] = {1, 5, 5, 3, 32}; // {batch, hi, wi, cin, cout}
    uint8_t filterSizes[2] = {3, 3};              // {kh, kw}
    uint8_t pads[4] = {1, 1, 1, 1};               // {padLeft, padRight, padTop, padBottom}
    uint8_t strides[2] = {2, 2};                  // {strideH, strideW}
    uint8_t dilations[2] = {1, 1};                // {dilationH, dilationW}
    int32_t deviceId = 0;

    Conv2dParams problemParams =
        Conv2dParams::MakeConv2dParams(dataSizes, filterSizes, pads, strides, dilations);

    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        aclFinalize();
        return ret;
    }
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    int fail = RunOneCase(problemParams, stream,
                          fmapRawPath, filterRawPath, biasRawPath, outRawPath, wsRawPath);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (fail == 0) {
        LOG_PRINT("\n========== ALL FINISHED ==========\n");
        return 0;
    }
    LOG_PRINT("\n========== CASE FAILED ==========\n");
    return 1;
}

#ifndef CONV2D_SILU_PRINT_INFO_H
#define CONV2D_SILU_PRINT_INFO_H

#include "kernel_operator.h"
#include "conv2d_silu_tiling.h"


__aicore__ inline void print_tiling_data(const Conv2dSiluTilingData* tiling) {
    AscendC::printf("Tiling Data:\n");
    AscendC::printf("  batch: %u\n", tiling->batch);
    AscendC::printf("  hi: %u\n", tiling->hi);
    AscendC::printf("  wi: %u\n", tiling->wi);
    AscendC::printf("  cin: %u\n", tiling->cin);
    AscendC::printf("  cout: %u\n", tiling->cout);
    AscendC::printf("  kh: %u\n", tiling->kh);
    AscendC::printf("  kw: %u\n", tiling->kw);
    AscendC::printf("  padLeft: %u\n", tiling->padLeft);
    AscendC::printf("  padRight: %u\n", tiling->padRight);
    AscendC::printf("  padTop: %u\n", tiling->padTop);
    AscendC::printf("  padBottom: %u\n", tiling->padBottom);
    AscendC::printf("  strideH: %u\n", tiling->strideH);
    AscendC::printf("  strideW: %u\n", tiling->strideW);
    AscendC::printf("  dilationH: %u\n", tiling->dilationH);
    AscendC::printf("  dilationW: %u\n", tiling->dilationW);
    AscendC::printf("  cin1: %u\n", tiling->cin1);
    AscendC::printf("  cout1: %u\n", tiling->cout1);
    AscendC::printf("  c0: %u\n", tiling->c0);
    AscendC::printf("  workspaceSize: %llu\n", tiling->workspaceSize);
}


template<typename Param>
__aicore__ inline void print_underlying_param(Param param) {
    AscendC::printf("Param Data:\n");
    AscendC::printf("  batch: %u\n", param.problemShape.batch());
    AscendC::printf("  hi: %u\n", param.problemShape.hi());
    AscendC::printf("  wi: %u\n", param.problemShape.wi());
    AscendC::printf("  cin: %u\n", param.problemShape.cin());
    AscendC::printf("  cout: %u\n", param.problemShape.cout());
    AscendC::printf("  kh: %u\n", param.problemShape.kh());
    AscendC::printf("  kw: %u\n", param.problemShape.kw());
    AscendC::printf("  padLeft: %u\n", param.problemShape.padLeft());
    AscendC::printf("  padRight: %u\n", param.problemShape.padRight());
    AscendC::printf("  padTop: %u\n", param.problemShape.padTop());
    AscendC::printf("  padBottom: %u\n", param.problemShape.padBottom());
    AscendC::printf("  strideH: %u\n", param.problemShape.strideH());
    AscendC::printf("  strideW: %u\n", param.problemShape.strideW());
    AscendC::printf("  dilationH: %u\n", param.problemShape.dilationH());
    AscendC::printf("  dilationW: %u\n", param.problemShape.dilationW());
    AscendC::printf("  cin1: %u\n", param.problemShape.cin1());
    AscendC::printf("  cout1: %u\n", param.problemShape.cout1());
    AscendC::printf("  c0: %u\n", param.problemShape.C0);
}

#endif // CONV2D_SILU_PRINT_INFO_H
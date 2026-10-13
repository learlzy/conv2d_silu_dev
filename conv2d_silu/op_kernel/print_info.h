#include "kernel_operator.h"
#include "conv2d_silu_tiling.h"


__aicore__ inline void print_tiling_data(Conv2dSiluTilingData* tiling) {
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
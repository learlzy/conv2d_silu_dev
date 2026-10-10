#include "es_all_ops.h"      // 内置：Mod, Cast, Const...
#include "es_custom_ops.h"   // 自定义：Conv2dSilu
#include "ge/fusion/pass/pattern_fusion_pass.h"

using namespace ge;
using namespace fusion;

class Conv2dSiluPass : public PatternFusionPass {
 protected:
  // 1) 要匹配的拓扑
  std::vector<PatternUniqPtr> Patterns() override {
    std::cout << "Define pattern for Conv2dSiluPass" << std::endl;

    std::vector<PatternUniqPtr> patterns;

    auto gb = es::EsGraphBuilder("conv2d_silu_pattern");
    auto x = gb.CreateInput(0);
    auto filter = gb.CreateInput(1);
    auto bias = gb.CreateInput(2);
    auto conv2d = es::Conv2D(x, filter, bias, nullptr, {1, 1, 2, 2}, {1, 1, 1, 1}, {1, 1, 1, 1});
    auto sigmoid = es::Sigmoid(conv2d);
    auto mul = es::Mul(conv2d, sigmoid);
    auto graph = gb.BuildAndReset({mul});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    patterns.emplace_back(std::move(pattern));
    return patterns;
  }

  // 2) 可选：额外过滤（dtype、shape、属性等）
  bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
    std::cout << "Define MeetRequirements for Conv2dSiluPass" << std::endl;
    
    return true;
  }

  // 3) 替换成你的融合算子
  GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &match_result) override {
    std::cout << "Define Replacement for Conv2dSiluPass" << std::endl;
    // 找出conv2d节点
    GNode conv2d_node;
    for (const auto &node : match_result->GetMatchedNodes()) {
      AscendString type;
      node.GetType(type);
      string type_str = type.GetString();
      if (type_str == "Conv2D") conv2d_node = node;
    }
    // 获取属性值
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    if (conv2d_node.GetAttr("strides", strides) != GRAPH_SUCCESS ||
        conv2d_node.GetAttr("pads", pads) != GRAPH_SUCCESS ||
        conv2d_node.GetAttr("dilations", dilations) != GRAPH_SUCCESS) {
      return nullptr;
    }
    // 构建新子图
    auto gb = es::EsGraphBuilder("replacement");
    auto x = gb.CreateInput(0);
    auto filter = gb.CreateInput(1);
    auto bias = gb.CreateInput(2);
    auto fused = es::Conv2dSilu(x, filter, bias, strides, pads, dilations);
    return gb.BuildAndReset({fused});
  }
};

// 注册到 GE，指定执行阶段
REG_FUSION_PASS(Conv2dSiluPass)
    .Stage(CustomPassStage::kAfterOriginGraphOptimize);
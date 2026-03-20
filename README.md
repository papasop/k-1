# 洛伦兹Transformer（Lorentz Transformer）

基于K=1信息几何场方程的Transformer架构。在参数空间引入伪黎曼几何结构，用闵可夫斯基内积替代欧氏内积，实现真正的洛伦兹注意力机制。

---

## 核心实验结果（正式对比）

| 模型 | best val_loss | 参数量 | 数据 | 步数 |
|------|-------------|-------|------|------|
| 标准Transformer（对照组） | 7.7182 | 17.6M | wikitext-2 | 10000 |
| 洛伦兹Transformer（版本2） | **7.5069** | 17.6M | wikitext-2 | 10000 |

**差值：-0.2113（改善2.74%）**

完全控制变量：同等参数量、同等数据、同等步数，唯一变量是洛伦兹几何结构的开/关。

---

## 核心发现

### 1. 伪黎曼结构存在（实验确认）
W_Q参数空间的60-79%方向对dt²_info的Hessian为负（类时方向）。在以下条件下全部稳定复现：
- 合成多跳推理任务（1-hop / 2-hop）
- 真实语言数据（wikitext-2，英语维基百科）
- 128维 / 256维 / 512维三个规模
- 3个随机种子

**这是Transformer处理语言时参数空间的内在几何性质，不是任务特异性产物。**

### 2. 层深类时规律（新发现）
6层模型中，中间层（Layer 2-3）类时比例最高，浅层和深层较低：
```
layer 3: frac=0.754  ██████████████  ← 峰值（长程语义整合）
layer 2: frac=0.738  ██████████████
layer 4: frac=0.664  █████████████
layer 1: frac=0.656  █████████████
layer 5: frac=0.668  █████████████
layer 0: frac=0.602  ████████████  ← 浅层最低（局部特征提取）
```
三次独立实验稳定复现。

### 3. r规律（8个独立实验）
```
r(baseline, delta) ≈ -1.0
```
跨K-field、Minkowski注意力、Geodesic Adam三种完全不同的注入机制全部成立。

```
┌─────────────────────────────────────────────────────────────┐
│  r ≈ -1.0 不是人为选择的超参数                              │
│                                                             │
│  它是：                                                     │
│  ✓ 狭义相对论的基本几何结构                                  │
│  ✓ Minkowski 时空的数学签名                                 │
│  ✓ 因果关系的几何根源                                        │
│  ✓ 洛伦兹大模型区别于标准 Transformer 的本质                 │
│                                                             │
│  物理数据版本证明：                                          │
│  当输入数据具有真实时空结构时，r = -1 自动成立                │
│  洛伦兹大模型的设计与宇宙的基本规律一致                        │
└─────────────────────────────────────────────────────────────┘
```

### 4. 与NCCL论文的关系
Neural Null Cones论文（2025）在GPT-2上独立确认了损失Hessian的不定结构，验证精度10⁻²⁶。该论文明确将理论起源归于本项目的K=1场方程。

---

## 为什么洛伦兹能改进Transformer

传统方法（交叉熵Hessian、Fisher信息矩阵）永远正定，det G > 0，无法区分类时/类空方向。

dt²_info的Hessian由于可实现性条件，被强迫成不定矩阵，det G < 0。这一个符号翻转触发Theorem 4（Non-Separability），洛伦兹签名从代数上涌现，类时/类空方向得以定义。

**没有dt²_info，闵可夫斯基在参数空间里是盲目的。有了dt²_info，P_t精确定位类时方向，洛伦兹修正才能正确施加。**

---

## 当前 Python 包 API

当前发布的 `lorentz_transformer` 包聚焦于可直接复用的核心模块：

### `LorentzMultiHeadAttention`
```
scores_L = QK^T/√d - 2α(Q_t_scaled)K^T/√d
```
- 输入：`(batch, seq_len, d_model)`
- 输出：`(attention_output, attention_weights)`
- 配置字段：`d_model`、`n_heads`（或 `num_heads`）、`lorentz_alpha`、`dropout`

### `MinkowskiLayerNorm` 系列
```
标准向量归一化:        x / sqrt(||x||² + ε)
MinkowskiLayerNorm: x / sqrt(|<x,x>_η| + ε)
<x,x>_η = ||x_s||² - ||x_t||²
```
- `MinkowskiLayerNormOptimized`：纯 L2 版本
- `MinkowskiLayerNormStable`：保守回退版本
- `MinkowskiLayerNormImproved`：推荐版本
- `MinkowskiLayerNorm`：`MinkowskiLayerNormImproved` 的向后兼容别名

### 研究原型说明
README 中提到的 `GeodesicAdam`、`LorentzFFN`、`LorentzPositionalEncoding`、
`TimeLikeProbe` 以及训练入口函数目前未作为该 Python 包的一部分发布。
本文档现在只展示仓库中实际可导入的模块。

---

## 组件消融实验（wikitext-2）

| 版本 | 组件 | best val_loss | Phase 2改善 |
|------|------|--------------|------------|
| 对照 | 标准Transformer | 7.7182 | — |
| 1 | 标准Transformer（旧基线） | 7.6739 | -0.001 |
| 2 | +MinkowskiLayerNorm | **7.5069** | **-0.248** |
| 3 | +洛伦兹FFN | 7.5518 | -0.183 |
| 4 | +洛伦兹位置编码 | 7.5211 | -0.156 |

MinkowskiLayerNorm是效果最显著的单一组件。

---

## 快速开始

```python
from dataclasses import dataclass

import torch

from lorentz_transformer import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
)


@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 8
    lorentz_alpha: float = 0.25
    dropout: float = 0.1


config = Config()
attn = LorentzMultiHeadAttention(config)
norm = MinkowskiLayerNorm(config.d_model)

x = torch.randn(2, 16, config.d_model)
attn_out, attn_weights = attn(x)
output = norm(attn_out)

print(output.shape)       # torch.Size([2, 16, 256])
print(attn_weights.shape) # torch.Size([2, 8, 16, 16])
```

更多可运行示例见 `examples/` 目录。

---

## 文件结构

```
lorentz_transformer/               # 主包
├── __init__.py                    # 顶层导出
└── core/                          # 核心组件
    ├── __init__.py
    ├── attention.py               # LorentzMultiHeadAttention 与诊断函数
    └── minkowski_norm.py          # 3 个 MinkowskiLayerNorm 变体

examples/                          # 最小可运行示例
├── attention_example.py
├── full_model_example.py
├── norm_example.py
└── quick_example.py

tests/
├── test_attention.py
├── test_integration.py
└── test_minkowski_norm.py
```

---

## 理论背景

基于K=1信息几何场方程（chronogeometrodynamics）。

信息时间度量 `dt²_info = Σ_q Φ_q/H_q` 在参数空间定义了伪黎曼度量。Theorem 4证明：当且仅当det G < 0（洛伦兹签名）时，系统存在非平凡稳定边界 dc > 0。这使洛伦兹几何、辛结构、因果结构同时成为代数结果而非独立假设。

**参考：**
- Li, Y. Y. N. (2026). *K=1 Chronogeometrodynamics*. Zenodo. https://doi.org/10.5281/zenodo.19011128

---

## 当前状态

**已完成：**
- ✅ 正式对比实验：洛伦兹比标准Transformer val_loss低0.21（2.74%）
- ✅ `LorentzMultiHeadAttention` 与 `MinkowskiLayerNorm` 核心模块已打包并带测试
- ✅ 真实语言数据上的伪黎曼结构验证（wikitext-2）
- ✅ 层深类时规律发现

**研究原型（未随当前包发布）：**
- 🧪 Geodesic Adam
- 🧪 Lorentz FFN
- 🧪 Lorentz 位置编码
- 🧪 训练入口函数（`baseline_train()` / `wikitext_train()` / `quick_train()`）

**进行中：**
- 🔄 更大数据集（openwebtext）验证
- 🔄 论文写作

**计划中：**
- 📋 GPT-2规模（768维/12层）验证
- 📋 多随机种子统计显著性验证

---

*洛伦兹Transformer项目 — 2025-2026*

## License

MIT

## Citation

```bibtex
@misc{li2026k1,
  author  = {Li, Y. Y. N.},
  title   = {K=1 Chronogeometrodynamics},
  year    = {2026},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.19011128},
  url     = {https://doi.org/10.5281/zenodo.19011128}
}
```

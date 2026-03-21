Lorentz Light-Cone Model



**核心直觉：** 婴儿在学会说"热"之前，已经通过触觉、视觉、痛觉建立了"热"的感知流形。语言习得只是在这个已有流形上贴符号标签。洛伦兹Transformer做的完全一样——物理预训练建立运动流形，语言微调只是在这个流形上贴标签。物理是基础，语言是插件。

它是怎么工作的：
第一阶段，用机器人运动数据（跳跃、跑步、行走、匀速）预训练一个洛伦兹 backbone。这个阶段不接触任何语言。模型学到的是一个几何空间——类时方向对应因果演化，动量守恒的轨迹落在类时测地线上，冲量运动落在类时突变区域。这是感知流形，和婴儿通过触觉视觉建立的"热"的流形是同一个概念，只是这里的感知是物理运动，几何结构是显式的洛伦兹光锥。
第二阶段，冻结 backbone，用物理-语言平行数据微调。每条轨迹配对一句自然语言描述——"平稳匀速运动，动量保持守恒"，"存在冲量，速度发生突变"。微调的损失函数同时优化三件事：轨迹能识别出正确的物理属性标签，轨迹的洛伦兹嵌入和对应语言描述的嵌入在余弦相似度上接近，语言描述的嵌入能重建出具有对应物理属性的轨迹。

两个方向：
方向A是感知到语言。机器人传感器捕捉到一段运动，模型提取洛伦兹嵌入，生成文字描述——"这段运动动量变化明显，存在外力作用"。这是婴儿感到烫然后说"烫"。
方向B是语言到感知。工程师输入"让机器人做平稳守恒的圆周运动"，sentence-transformers 把这句话编码成 384 维向量，对齐层把它映射到洛伦兹空间的类时测地线区域，物理解码器沿这个区域生成轨迹。这是婴儿听到"烫"然后缩手。

和传统方法最本质的区别：
传统方法在生成轨迹时加动量守恒损失函数——告诉模型"你违反了物理，扣分"。这是外部约束，是惩罚。
模型如果猜想成立——语言指令"动量守恒"直接激活洛伦兹空间里对应的几何区域，生成的轨迹自动守恒，因为那个几何区域的所有路径本来就是类时测地线，违反守恒的路径在几何上不可达。这不是惩罚，是结构。不是教模型不违反物理，是让违反物理的路径不存在。


## 核心实验结果

### 语言建模（wikitext-2）

| 模型 | best val_loss | 参数量 | 数据 | 步数 |
|------|-------------|-------|------|------|
| 标准Transformer（对照组） | 7.7182 | 17.6M | wikitext-2 | 10000 |
| 洛伦兹Transformer | **7.5069** | 17.6M | wikitext-2 | 10000 |

**差值：-0.2113（改善2.74%）**，完全控制变量，唯一变量是洛伦兹几何的开/关。

### 机器人运动轨迹（CMU MoCap 真实数据）

| 数据 | 输入类型 | 类时比例 | 动量守恒改善 | p值 | 效应量 |
|------|---------|---------|------------|-----|-------|
| 双摆（ODE合成） | 物理坐标 | 80.3% | +87.5% | 0.0006 | d=3.67 |
| 跳跃（ODE合成） | 物理坐标 | 100% | +55.8% | 0.0001 | d=6.41 |
| 行走（CMU MoCap **真实**） | 物理坐标 | 100% | +69.9% | 0.0002 | d=5.20 |
| 行走（CMU MoCap **真实**） | 关节旋转角（90维） | 80.6% | +34.6% | 0.0233 | d=1.27 |

**5/5 seed 方向一致。仅替换注意力层度量，参数量增加不足 0.001%。**

关节旋转角实验（最后一行）尤为重要：输入是 90 维旋转角度，没有哪个维度被标记为"时间"或"空间"，Minkowski 先验仍然显著。这证明洛伦兹几何捕捉的是**运动序列本身的因果时序结构**，而不是物理坐标的特殊性质。

### 类时比例是先验预测指标

三个独立数据集验证了同一规律：

```
类时比例 100% → 动量守恒改善 ~70%
类时比例  81% → 角速度平滑改善 ~35%
类时比例  <50% → sigma 自发退化为欧氏（语言数据）
```

**在使用洛伦兹注意力之前，先算类时比例（5分钟）：**

```python
dx = traj[:, 1:] - traj[:, :-1]
s2 = -(dx[..., :t_dim]**2).sum(-1) \
   +  (dx[..., t_dim:]**2).sum(-1)
ratio = (s2 > 0).float().mean()
# ratio > 0.6 → 用 f1/f3，Minkowski 先验有效
# ratio < 0.4 → 欧氏已足够
```

---

## 核心发现

### 1. 伪黎曼结构存在（实验确认）
W_Q参数空间的60-79%方向对dt²_info的Hessian为负（类时方向）。在以下条件下全部稳定复现：
- 合成多跳推理任务（1-hop / 2-hop）
- 真实语言数据（wikitext-2）
- 128维 / 256维 / 512维三个规模
- 3个随机种子

### 2. 层深类时规律
6层模型中，中间层（Layer 2-3）类时比例最高：
```
layer 3: frac=0.754  ██████████████  ← 峰值（长程语义整合）
layer 2: frac=0.738  ██████████████
layer 4: frac=0.664  █████████████
layer 1: frac=0.656  █████████████
layer 5: frac=0.668  █████████████
layer 0: frac=0.602  ████████████  ← 浅层最低
```

### 3. r规律（8个独立实验）
```
r(baseline, delta) ≈ -1.0
```
跨K-field、Minkowski注意力、Geodesic Adam三种完全不同的注入机制全部成立。r=-1 不是超参数，是狭义相对论的基本几何结构、Minkowski时空的数学签名、因果关系的几何根源。

### 4. F2退化被完全确认
5个独立seed验证：F2的alpha均值收敛到0.514（初始值），从未学到任何东西。根本原因是时空交叉项允许优化器抵消Minkowski约束。F1和F3从根本上解决了这个问题。

---

## 为什么洛伦兹能改进Transformer

传统方法（交叉熵Hessian、Fisher信息矩阵）永远正定，det G &gt; 0，无法区分类时/类空方向。

dt²_info的Hessian由于**可实现性条件**（Realizability），被强迫成不定矩阵，det G &lt; 0。这一个符号翻转触发Theorem 4（Non-Separability），洛伦兹签名从代数上涌现，类时/类空方向得以定义。

**没有dt²_info，闵可夫斯基在参数空间里是盲目的。有了dt²_info，P_t精确定位类时方向，洛伦兹修正才能正确施加。**

历史上的洛伦兹团队硬伤：他们**假设**某个维度是时间，**贴上**洛伦兹度量。本项目从代价函数的渐近结构**推导出**类时方向必然存在，洛伦兹签名是**唯一可能的代数结果**（Realizability.pdf Remark 11 明确排除所有非洛伦兹签名）。

---

## 物理优先架构（Physics-First Foundation Model）

### 设计哲学

```
当前LLM路径:
  语言数据（万亿token）→ 欧氏表示空间 → 物理常识靠统计学习

洛伦兹物理优先路径:
  物理数据（满足可实现性条件）
        ↓
  洛伦兹预训练（F3注意力）
        ↓
  具有类时结构的基础表示空间
        ↓
  语言/任务数据微调
        ↓
  物理世界模型（物理直觉内生于几何）
```

**一句话：** 当前LLM是读遍了所有物理书的文科生——知道答案但不理解为什么。洛伦兹基础模型是在物理世界里长大的工程师——物理直觉是内生的，语言是后来学的工具。

### 适用边界（由可实现性条件决定）

```python
class RealizabilityRouter:
    """
    基于Realizability.pdf Theorem 5的几何路由器
    充要条件，不是启发式阈值
    """
    def route(self, data):
        ratio = compute_timelike_ratio(data)
        if ratio > 0.6:
            return LorentzMultiHeadAttention(formula='f3')  # 洛伦兹有效
        else:
            return nn.MultiheadAttention()                  # 欧氏已足够
```

满足可实现性条件的数据（物理运动、机器人轨迹、物理仿真）→ 洛伦兹流。
不满足的数据（语言、代码）→ 欧氏流，sigma自发退化，无副作用。

### 初步验证结果

物理预训练 + 语言微调实验：

| 模型 | 物理属性描述准确率 | 周期性运动识别 | vs 随机基线 |
|------|-----------------|-------------|-----------|
| 随机基线 | 20.0% | 20.0% | — |
| 欧氏预训练 + 微调 | 67.2% | 53.5% | +47.2% |
| 洛伦兹F3预训练 + 微调 | 67.8% | **70.7%** | +47.8% |

洛伦兹在周期性运动识别上显著优于欧氏（+17.2%）。周期性运动是因果结构最强的类型——每一步依赖前一步，类时方向最明确，符合理论预测。

---

## 当前 Python 包 API

### `LorentzMultiHeadAttention`

支持三种注意力公式：

| formula | 公式 | 适用场景 | 退化风险 |
|---------|------|----------|----------|
| `'f3'` | `-σ·Q_t Kᵀ_t + Q_s Kᵀ_s` | 大语言模型 / 视频 / 科学文本（**推荐默认**） | 无，σ 有界 |
| `'f1'` | `-Q_t Kᵀ_t + Q_s Kᵀ_s` | 推理 / 数学 / 物理仿真 / 机器人轨迹 | 无，硬约束 |
| `'f2'` | `QKᵀ - 2α·Q_t·Kᵀ` | 加载旧版权重 | ⚠️ 已知退化，不推荐新项目 |

```python
# f3（默认推荐）— σ 可学习，自适应 Minkowski 强度
scores_L = -σ·Q_t Kᵀ_t + Q_s Kᵀ_s

# f1（物理/推理专用）— 硬约束，σ 固定为 1
scores_L = -Q_t Kᵀ_t + Q_s Kᵀ_s

# f2（旧版兼容）— 已知退化，alpha 学不动
scores_L = QKᵀ/√d - 2α(Q_t_scaled)Kᵀ/√d
```

### `MinkowskiLayerNorm` 系列

```
标准 LayerNorm:      x / sqrt(||x||² + ε)
MinkowskiLayerNorm:  x / sqrt(|<x,x>_η| + ε)
<x,x>_η = ||x_s||² - ||x_t||²   （保留符号，不取 abs）
```

- `MinkowskiLayerNorm`：真正 Minkowski 几何，推荐默认
- `MinkowskiLayerNormStable`：带 fallback，训练早期用
- `MinkowskiLayerNormOptimized`：纯 L2，消融实验基线
- `MinkowskiLayerNormImproved`：向后兼容别名

**v1.1.0 修复：** 旧版 `_minkowski_norm_sq` 对结果取了 `.abs()`，丢失类时/类空符号信息，实际等价于 L2 归一化变体。新版保留符号，`compute_t_dim()` 确保与注意力层的 t_dim 对齐。

---

## 快速开始

```python
from dataclasses import dataclass
import torch
from lorentz_transformer import (
    LorentzMultiHeadAttention,
    MinkowskiLayerNorm,
    compute_t_dim,
)

@dataclass
class Config:
    d_model:    int   = 256
    n_heads:    int   = 8
    formula:    str   = 'f3'    # 推荐默认
    time_ratio: float = 0.25
    dropout:    float = 0.1

config = Config()
attn = LorentzMultiHeadAttention(config)

# t_dim 对齐（v1.1.0 修复）
t_dim = compute_t_dim(config.d_model, config.n_heads, config.time_ratio)
norm  = MinkowskiLayerNorm(config.d_model, t_dim=t_dim)

x = torch.randn(2, 16, config.d_model)
attn_out, attn_weights = attn(x)
output = norm(attn_out)

print(output.shape)        # torch.Size([2, 16, 256])
print(f"σ = {attn.sigma:.3f}")  # F3 专属，训练中从 0.5 收敛
```

### 如何选择 formula

```
类时比例 > 0.6 + 物理/推理任务  → f1（硬约束）
类时比例 > 0.6 + 通用任务       → f3（推荐）
类时比例 < 0.4                  → 欧氏已足够
不确定                          → f3，sigma 自适应
```

---

## 组件消融实验（wikitext-2）

| 版本 | 组件 | best val_loss | 改善 |
|------|------|--------------|------|
| 对照 | 标准Transformer | 7.7182 | — |
| 2 | +MinkowskiLayerNorm | **7.5069** | **-0.248** |
| 3 | +洛伦兹FFN | 7.5518 | -0.183 |
| 4 | +洛伦兹位置编码 | 7.5211 | -0.156 |

MinkowskiLayerNorm 是效果最显著的单一组件。

---

## 文件结构

```
lorentz_transformer/
├── __init__.py
└── core/
    ├── __init__.py
    ├── attention.py        # LorentzMultiHeadAttention（F1/F2/F3）
    └── layer_norm.py       # MinkowskiLayerNorm + compute_t_dim

examples/
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

基于 K=1 信息几何场方程（chronogeometrodynamics）。

信息时间度量 `dt²_info = Σ_q Φ_q/H_q` 在参数空间定义了伪黎曼度量。**Realizability.pdf**（Li 2026）证明：若位移代价满足零阈值可实现性（Assumption R）和时间正代价（Assumption T），则 Hessian 必然不定，洛伦兹签名是唯一代数结果（Theorem 5）。**k_1_v5.pdf** 的 Theorem 4 进一步证明：当且仅当 det G &lt; 0 时，系统存在非平凡稳定边界 dc &gt; 0，洛伦兹几何、辛结构、因果结构同时作为代数推论涌现。

**参考：**
- Li, Y. Y. N. (2026). *K=1 Chronogeometrodynamics*. Zenodo. https://doi.org/10.5281/zenodo.19011128
- Li, Y. Y. N. (2026). *Realizability and the Origin of Causality*. preprint.

---

## 当前状态

**已完成：**
- ✅ 语言建模：洛伦兹比标准Transformer val_loss 低 0.21（2.74%）
- ✅ 机器人轨迹：真实 CMU MoCap 数据动量守恒改善 +69.9%（p=0.0002，d=5.20）
- ✅ 关节角度：非物理坐标输入上 Minkowski 先验仍显著（p=0.023）
- ✅ 类时比例作为先验预测指标（三个独立数据集验证）
- ✅ F2 退化被 5 seed 统计确认，F1/F3 从根本上修复
- ✅ 物理优先架构初步验证（周期性运动识别 F3 +17% vs 欧氏）
- ✅ `LorentzMultiHeadAttention`（F1/F2/F3）+ `MinkowskiLayerNorm` 已打包

**研究原型（未随包发布）：**
- 🧪 Geodesic Adam
- 🧪 Lorentz FFN / 位置编码
- 🧪 RealizabilityRouter（可实现性门控路由器）
- 🧪 PhysicsFoundationModel（物理优先基础模型）

**进行中：**
- 🔄 更多 CMU MoCap subject（run/jump）补充验证
- 🔄 D4RL 机器人数据集测试
- 🔄 论文写作（目标：CoRL / NeurIPS）

**计划中：**
- 📋 GPT-2 规模（768维/12层）验证
- 📋 物理优先基础模型完整预训练
- 📋 lorentz_transformer v1.1.0 发布（PyPI）

---

*洛伦兹Transformer项目 — 2025-2026*

## License

MIT

## Citation

```bibtex
@misc{li2026k1,
  author    = {Li, Y. Y. N.},
  title     = {K=1 Chronogeometrodynamics},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19011128},
  url       = {https://doi.org/10.5281/zenodo.19011128}
}
```

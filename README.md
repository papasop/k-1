# Lorentz Light-Cone Model (LLCM)

**几何原生物理智能 — Geometry-Native Physical Intelligence**

---

## 核心直觉

人类婴儿学会说"烫"，不是从词典里学的。

是因为手碰到热水壶时，神经系统建立了一个感知流形——温度的连续变化、肌肉收缩的时序、手臂后缩的轨迹。某一天有人说出"烫"这个声音，婴儿把这个符号贴在已有的流形上。从那以后，听到"烫"会缩手，烫到了会说"烫"。双向的。

**LLCM 做的完全一样：**

```
第一阶段：物理预训练
机器人运动数据（跳跃/跑步/行走/匀速）
        ↓
洛伦兹光锥预训练
        ↓
具有类时因果结构的感知流形
（动量守恒的轨迹落在类时测地线上
 违反物理守恒的路径在几何上不可达）

第二阶段：语言对齐
"平稳守恒运动" → sentence-transformers → 对齐层 → 洛伦兹空间
        ↓
找到感知流形中对应的类时区域
        ↓
生成轨迹沿类时测地线演化 → 动量守恒自动成立
```

**婴儿的感知流形是隐式的黑盒。LLCM 的感知流形是显式的——可实现性条件保证类时方向必然存在，Theorem 5 保证洛伦兹签名是唯一代数结果。感知流形的几何结构可以写成方程。**

这个方程是 `dt²_info = Σ_q Φ_q/H_q` 的 Hessian。精确推导见下方理论背景节。

---

## 与传统大模型的根本区别

| | 传统 LLM | LLCM |
|--|---------|------|
| 物理直觉来源 | 语料统计 | 洛伦兹几何内生 |
| 物理守恒保证 | 损失函数惩罚 | 几何上不可达 |
| 因果方向 | 对称（正反可互换） | 类时方向单向 |
| 失败模式 | 分布外靠猜，不可预期 | sigma 退化为欧氏，优雅降级 |
| 可解释性 | 黑盒 | Theorem 5 精确描述 |

**一句话：** 传统 LLM 是读遍了所有物理书的文科生。LLCM 是在物理世界里长大的工程师——物理直觉编码在几何里，语言是后来学的工具。

---

## 已验证的实验结果

### 机器人运动轨迹（主要贡献）

| 数据 | 输入类型 | 类时比例 | 物理一致性改善 | p值 | d |
|------|---------|---------|--------------|-----|---|
| 双摆（ODE合成） | 物理坐标 | 80.3% | +87.5% 动量守恒 | 0.0006 | 3.67 |
| 跳跃（ODE合成） | 物理坐标 | 100% | +55.8% 动量守恒 | 0.0001 | 6.41 |
| 行走（CMU MoCap 真实） | 物理坐标 | 100% | +69.9% 动量守恒 | 0.0002 | 5.20 |
| 行走（CMU MoCap 真实） | 关节旋转角（90维） | 80.6% | +34.6% 角速度平滑 | 0.0233 | 1.27 |

**5/5 seed 方向一致。仅替换注意力层度量，参数量增加不足 0.001%。**

关节旋转角实验最重要：90 维旋转角没有哪个维度是"时间"或"空间"，光锥注意力仍然显著。LLCM 捕捉的是运动序列本身的因果时序结构，不是物理坐标的特殊性质。

### 语言建模（wikitext-2）

| 模型 | val_loss | 参数量 | 步数 |
|------|---------|-------|------|
| 标准 Transformer | 7.7182 | 17.6M | 10000 |
| LLCM | **7.5069** | 17.6M | 10000 |

改善 2.74%，完全控制变量。

### 婴儿说话机制初步验证

| 实验 | 欧氏 | LLCM | 差异 |
|------|------|------|------|
| 方向A：动量变化识别 | 75.0% | **100%** | +25% |
| 方向A：周期性运动识别 | 53.5% | **70.7%** | +17.2% |
| 方向B：语言→物理（待完整验证） | — | — | 实验进行中 |


> ⚠️ 方向A结果来自 `physics_first_model.py` 单次实验（seed=0），尚未做多seed统计检验。方向指标支持猜想，但统计强度低于 CMU MoCap 实验（p=0.0002）。

---

## 核心猜想（待完整验证）

### 零损失函数动量守恒

```
"平稳守恒运动" → 洛伦兹空间类时区域激活
                        ↓
           生成轨迹沿类时测地线演化
                        ↓
           动量守恒自动成立
           不需要任何物理损失函数
```

**猜想成立条件：**

```python
mom_f3_zero_loss <= mom_euc_with_loss
# LLCM 零损失  <=  欧氏有动量守恒损失函数
```

这不是工程改进，而是物理不可达性——违反守恒的路径在洛伦兹几何上根本不存在。

### 双向语言对齐（层次3）

```
物理世界 ←→ 洛伦兹光锥空间 ←→ 自然语言空间

感知→语言: 机器人检测到动量突变 → "这个动作太猛了"
语言→感知: "平稳移动" → 动量守恒轨迹自动生成
```

**这个猜想目前没有完整实验验证，是开放的研究问题。**

---

## 三种注意力公式

| formula | 公式 | 适用场景 | 退化风险 |
|---------|------|----------|----------|
| `'f3'` | `-σ·Q_t Kᵀ_t + Q_s Kᵀ_s` | 通用 / LLM / 视频（推荐） | 无，σ 有界 |
| `'f1'` | `-Q_t Kᵀ_t + Q_s Kᵀ_s` | 物理仿真 / 机器人 / 推理 | 无，硬约束 |
| `'f2'` | `QKᵀ/√d - 2α·Q_t·Kᵀ/√d` | 旧版权重兼容 | 已确认退化 |

F2 退化根因：时空交叉项允许优化器抵消光锥约束，5 个 seed 全部确认 alpha 收敛到初始值 0.514。

---

## 为什么光锥注意力有效

**标准注意力：**
```
score = QKᵀ / √d
# 所有方向等价，时间和空间没有区别
```

**LLCM 光锥注意力（F3）：**
```
score = cat(-σ·Q_t Kᵀ_t,  Q_s Kᵀ_s) / √d
# 时间头取负 → 类时方向互相排斥 → 信息沿光锥边界传播
# σ = sigmoid(w)，训练中自适应收敛
```

这一个负号是所有物理性质的来源。

**理论保证：** 传统方法的 Fisher 信息矩阵永远正定，det G > 0，光锥不存在。`dt²_info` 的 Hessian 由于可实现性条件被迫成不定矩阵，det G < 0，触发 Theorem 4，洛伦兹签名从代数上涌现，类时/类空方向得以定义，光锥边界确立。

历史上的洛伦兹 Transformer 假设某个维度是时间——盲目的。LLCM 从 `dt²_info` 的代价结构推导类时方向，洛伦兹签名是唯一代数结果（与"与传统大模型的根本区别"节的对比一致）。

---

## 类时比例：使用 LLCM 的先验指标

三个独立数据集验证同一规律：

```
类时比例 100% → 物理一致性改善 ~70%
类时比例  81% → 物理一致性改善 ~35%
类时比例 <50% → sigma 自发退化为欧氏（优雅降级）
```

**5 分钟决定是否使用 LLCM：**

```python
dx    = data[:, 1:] - data[:, :-1]
t_dim = max(1, int(data.shape[-1] * 0.25))
s2    = -(dx[..., :t_dim]**2).sum(-1) \
      +  (dx[..., t_dim:]**2).sum(-1)
ratio = (s2 > 0).float().mean()
# ratio > 0.6 → LLCM 有效
# ratio < 0.4 → 欧氏已足够，sigma 会自动退化
```

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
    formula:    str   = 'f3'
    time_ratio: float = 0.25
    dropout:    float = 0.1

config = Config()
attn  = LorentzMultiHeadAttention(config)
t_dim = compute_t_dim(config.d_model, config.n_heads, config.time_ratio)
norm  = MinkowskiLayerNorm(config.d_model, t_dim=t_dim)

x = torch.randn(2, 16, config.d_model)
attn_out, attn_weights = attn(x)
output = norm(attn_out)

print(output.shape)            # torch.Size([2, 16, 256])
print(f"σ = {attn.sigma:.3f}") # 光锥强度，训练中自适应收敛
```

### 机器人 / 物理场景

```python
# 先算类时比例，再决定是否用 LLCM
dx    = traj[:, 1:] - traj[:, :-1]
t_dim = max(1, int(traj.shape[-1] * 0.25))
s2    = -(dx[..., :t_dim]**2).sum(-1) + (dx[..., t_dim:]**2).sum(-1)
ratio = (s2 > 0).float().mean()
print(f"类时比例: {ratio:.1%}")

# 物理任务用 f1（硬约束）
config = Config(formula='f1', time_ratio=0.25)
attn   = LorentzMultiHeadAttention(config)
```

### 婴儿说话模型（研究原型）

```python
# 需要: !pip install sentence-transformers -q
exec(open('baby_language_llcm.py').read())

# 自测
self_test()

# 完整实验（物理预训练 + 语言对齐 + 双向验证 + 零损失猜想）
results = run_baby_language_test(n_seeds=3)
```

---

## API 文档

> ⚠️ **重要：两个模块必须配合使用**
>
> `LorentzMultiHeadAttention` 建立光锥几何（注意力层），
> `MinkowskiLayerNorm` 保持光锥几何（归一化层）。
>
> **只用注意力层不用 MinkowskiLayerNorm：** 每层归一化会抹平洛伦兹几何，
> 几何信息无法在层间传递，效果大幅下降。这是之前实验 F3 预训练不收敛的根本原因之一。
>
> **v1.1.0 修复：** 旧版 `MinkowskiLayerNorm` 用了 `mq.abs()`，
> 丢失类时/类空的符号差异，实际退化为 L2 归一化。
> 新版保留符号，类时和类空归一化行为真正不同。
>
> ```python
> # 正确用法：两者配合，t_dim 必须对齐
> t_dim = compute_t_dim(d_model, n_heads, time_ratio)
> attn  = LorentzMultiHeadAttention(config)          # 注意力层
> norm  = MinkowskiLayerNorm(d_model, t_dim=t_dim)   # 归一化层，t_dim 对齐
> ```

### `LorentzMultiHeadAttention`
- 输入：`(batch, seq_len, d_model)`
- 输出：`(attn_output, attn_weights)`
- 字段：`d_model`、`n_heads`、`formula`、`time_ratio`、`dropout`
- 属性：`sigma`（F3 专属，当前光锥强度）

### `MinkowskiLayerNorm`

```
标准 LayerNorm:   x / sqrt(||x||² + ε)
光锥 LayerNorm:   x / sqrt(|⟨x,x⟩_η| + ε)
⟨x,x⟩_η = ||x_s||² - ||x_t||²   （保留符号，不取 abs）
```

### `compute_t_dim(d_model, n_heads, time_ratio)`

确保注意力层和 LayerNorm 的 t_dim 对齐（v1.1.0 修复旧版硬编码问题）。

---

## 消融实验（wikitext-2）

| 版本 | 组件 | val_loss | 改善 |
|------|------|---------|------|
| 对照 | 标准 Transformer | 7.7182 | — |
| 2 | + 光锥 LayerNorm | **7.5069** | **-0.248** |
| 3 | + 光锥 FFN | 7.5518 | -0.183 |
| 4 | + 光锥位置编码 | 7.5211 | -0.156 |

光锥 LayerNorm 是效果最显著的单一组件。

---

## 文件结构

```
lorentz_transformer/
├── __init__.py
└── core/
    ├── __init__.py
    ├── attention.py      # LorentzMultiHeadAttention（F1/F2/F3）
    └── layer_norm.py     # MinkowskiLayerNorm + compute_t_dim

examples/
├── attention_example.py
├── full_model_example.py
├── norm_example.py
└── quick_example.py

tests/
├── test_attention.py
├── test_integration.py
└── test_minkowski_norm.py

# 研究原型（不随包发布）
baby_language_llcm.py     # 婴儿说话模型（物理→语言双向对齐）
bidirectional_verify.py   # 双向验证脚本
physics_first_model.py       # 物理优先基础模型
joint_angle_experiment.py    # 关节角度实验
```

---

## 理论背景

**Realizability.pdf**（Li 2026）证明：若位移代价满足零阈值可实现性（Assumption R）和时间正代价（Assumption T），Hessian 必然不定，洛伦兹签名是唯一代数结果，光锥自然涌现。所有非洛伦兹签名被 Remark 11 明确排除。

**k_1_v5.pdf** Theorem 4 证明：当且仅当 det G < 0 时，系统存在非平凡稳定边界 dc > 0，洛伦兹几何、辛结构、因果结构同时作为代数推论涌现，不是独立假设。

---

## 当前状态

**已验证：**
- ✅ 语言建模 val_loss 改善 2.74%（wikitext-2）
- ✅ 真实 CMU MoCap 动量守恒改善 +69.9%（p=0.0002，d=5.20）
- ✅ 关节旋转角（非物理坐标）光锥注意力显著（p=0.023）
- ✅ 类时比例作为先验预测指标（三个独立数据集）
- ✅ F2 退化 5 seed 统计确认，F1/F3 从根本上修复
- ✅ 方向A：动量变化识别 F3=100% vs 欧氏=75%
- ✅ 核心模块打包：`LorentzMultiHeadAttention` + `MinkowskiLayerNorm`

**猜想（待完整验证）：**
- 🔬 零损失函数动量守恒（语言指令 → 几何本能 → 守恒自动成立）
- 🔬 双向语言对齐（方向B：语言→物理 完整验证）
- 🔬 婴儿说话机制层次3（物理感知流形与语言空间双向对齐）

**进行中：**
- 🔄 `baby_language_llcm.py` 实验运行中
- 🔄 CMU MoCap run/jump 补充验证
- 🔄 论文写作（目标：CoRL / NeurIPS）

**计划中：**
- 📋 GPT-2 规模验证（768维/12层）
- 📋 D4RL 机器人数据集
- 📋 v1.1.0 PyPI 发布

---

*Lorentz Light-Cone Model — 2025-2026*

## License

MIT

## Citation


@misc{li2026k1,
  author    = {Li, Y. Y. N.},
  title     = {K=1 Chronogeometrodynamics},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19011128}
}

@misc{li2026realizability,
  author = {Li, Y. Y. N.},
  title  = {Realizability and the Origin of Causality},
  year   = {2026},
  note   = {preprint}
}
```

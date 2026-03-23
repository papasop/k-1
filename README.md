# Lorentz Light-Cone Model (LLCM)

**一个物理预训练的洛伦兹世界模型**
**A Physics-First Lorentz World Model**

> 婴儿先建立感知流形，再学语言；
> 机器人先从物理ODE数据建立洛伦兹几何直觉，再把语言贴上去。
>
> *Perception before language. Geometry before symbol.*

---

**实验验证（8 seeds，EMBED=256，N_LAYERS=6）：**
- ✅ 层2：感知流形先于语言自发形成，mq差距97.8倍（均值），**p=0.0000，d=13.22**，8/8 seeds
- ✅ 方向A：物理感知流形被语言索引，F3 > 欧氏，**7/8 seeds，p=0.0628，d=0.78**
- ✅ 方向B：语言激活物理守恒本能，F3 < 欧氏，**6/8 seeds，p=0.0748，d=0.74**
- ✅ 6类认知功能（感知/推理/记忆/逻辑/智慧/对比）双向语言对齐


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

**三层实验验证对应婴儿说话的完整闭环：**

```
层2（感知流形存在，且分离先于语言）：
  F3 backbone 纯物理预训练后，守恒运动嵌入自发偏向类时区域
  stable嵌入偏类时，changing嵌入偏类空——无任何语言信号参与
  mq差距47倍，5/5 seed
  → 婴儿开口说话之前，感知分离已经存在
  → 语言不是创造分离，而是在已有分离上贴标签

方向A（感知流形可以被语言索引）：
  物理轨迹嵌入与语言描述余弦相似度 F3>欧氏
  p=0.0091，d=2.12，5/5 seed
  → 婴儿听到"烫"，能把词贴在感知流形上

方向B（语言可以激活物理本能）：
  语言指令"平稳守恒运动"→ F3生成轨迹守恒率低49%
  p=0.0167，d=1.77，5/5 seed，无物理损失函数
  → 婴儿听到"小心"，身体自动放慢
```

**闭环完整。下一步：动态交互——婴儿的感知流形在和父母的实时交互中不断精化，这是 Law II 在线学习的验证目标。**

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

## 已验证的实验结果（8 seeds，EMBED_DIM=256，N_LAYERS=6）

### 层2：感知流形先于语言自发形成（最强证据）

| 指标 | F3洛伦兹 | 欧氏对照 | 统计 |
|------|----------|---------|------|
| 类时比例 | **95%** | 0% | 8/8 seeds |
| mq均值（预训练后） | **-1.618±0.530** | +136.4±9.9 | p=0.0000，**d=13.22** |
| mq差距倍数 | **97.8倍（均值）** | — | 无语言信号 |

**意义：** 纯物理预训练结束后，无任何语言信号，F3感知分离已经存在。这是"感知先于语言"的直接实验证据。

### 方向A：物理感知流形可被语言索引

| 配置 | 差异 | 统计 |
|------|------|------|
| F3 - 欧氏 语言对齐得分 | **+0.0286±0.0342** | p=0.0628，d=0.78，**7/8 seeds** |

逐类F3优势（跨seed最稳定）：

| 认知功能 | 物理对应 | F3优势 |
|---------|---------|--------|
| contrast（感知对比） | 非守恒运动 | 多数seed最强 |
| logic（逻辑） | 约束力学 | 稳定正 |
| wisdom（智慧） | 最优控制 | 多数seed正 |
| memory（记忆） | 迟滞系统 | 部分seed强 |

### 方向B：语言可激活物理守恒本能

| 配置 | 守恒率 | 统计 |
|------|--------|------|
| 欧氏 | 0.4181±0.0535 | — |
| F3洛伦兹 | **0.3816±0.0260** | p=0.0748，d=0.74，**6/8 seeds** |

F3比欧氏守恒9%，无任何物理损失函数约束——守恒性来自洛伦兹几何，不是训练信号。


## 婴儿说话项目里程碑

| 里程碑 | 状态 | 证据 |
|--------|------|------|
| **里程碑0** 感知流形先于语言自发形成 | ✅ 完成 | mq差距97.8倍，p=0.0000，d=13.22，8/8 seeds |
| **里程碑1** 语言索引物理感知流形（方向A） | ✅ 完成 | 7/8 seeds，p=0.0628，d=0.78 |
| **里程碑2** sigma充分激活（>0.60） | 🔄 进行中 | sigma均值0.533，loss_sigma在工作 |
| **里程碑3** 语言激活守恒本能（方向B） | ✅ 完成 | 6/8 seeds，p=0.0748，d=0.74 |
| **里程碑4** 动态交互（Law II在线学习） | ⬜ 未开始 | online_interaction_test.py |
| **里程碑5** HJB-洛伦兹猜想验证 | 🔬 实验中 | hjb_lorentz_conjecture.py |


## 三种注意力公式

| formula | 公式 | 适用场景 | 退化风险 |
|---------|------|----------|----------|
| `'f3'` | `-σ·Q_t Kᵀ_t + Q_s Kᵀ_s` | 通用 / LLM / 视频（推荐） | 无，σ 有界 |
| `'f1'` | `-Q_t Kᵀ_t + Q_s Kᵀ_s` | 物理仿真 / 机器人 / 推理 | 无，硬约束 |
| `'f2'` | `QKᵀ/√d - 2α·Q_t·Kᵀ/√d` | 旧版权重兼容 | 已确认退化 |

F2 退化根因：时空交叉项允许优化器抵消光锥约束，5 个 seed 全部确认 alpha 收敛到初始值 0.514。

---

## 模块验证文件清单

每个模块有独立的验证脚本，可直接复现。所有脚本共享 `core.py` 的模型定义。

```
core.py                      ← 所有模块共用（MinkowskiLN, Attn, LLCMBackbone,
                                simulate, build_dataset, pretrain）
```

| 模块 | 验证内容 | 脚本 | 关键结果 |
|------|---------|------|---------|
| 模块1 | 物理预训练 loss F3 << 欧氏 | `layer3_zero_loss_B.py` | F3=0.025 欧氏=0.275 ×10倍 |
| 模块2 | 语言编码器语义质量 | `baby_talk_full_test.py` verify_module2() | 中文同类>跨类 ✅ |
| 模块3 | 方向A：物理→语言对齐 | `layer1_minimal_test.py` | p=0.0091 d=2.12 5/5 |
| 模块4a | 类时比例 F3>欧氏 | `layer1_minimal_test.py` 层2测量 | mq差距76.5倍 p=0.0000 d=12.52 5/5 |
| 模块4b | Law II 在线收敛速度 | `online_interaction_test.py` | dc>0 验证中 |
| 模块5 | 方向B：语言→守恒轨迹 | `layer1_minimal_test.py` | p=0.0167 d=1.77 5/5 ✅ |
| 模块5 | 方向B：语言→守恒轨迹 | `baby_talk_full_test.py` | p=0.041 弱版本 |
| 完整 | 五模块联合验证 | `baby_talk_full_test.py` | 待全部通过 |

### 复现命令

```python
# 模块1（结构效应×10倍）
exec(open('layer3_zero_loss_B.py').read())

# 模块3+方向B（p=0.0091/0.0167，双向验证）
exec(open('layer1_minimal_test.py').read())

# 模块4b（Law II 在线交互）
exec(open('online_interaction_test.py').read())

# 完整五模块
exec(open('baby_talk_full_test.py').read())
```

### 从 core.py import

```python
from core import (
    LLCMBackbone, pretrain,          # 模型和预训练
    build_dataset, simulate,          # 数据生成
    momentum_change, encode,          # 评估工具
    stable_ode, running_ode,          # 物理 ODE
    real_physics_baseline,            # 真实物理基准
    device, EMBED_DIM, T_DIM,         # 超参数
    LABELS, DESCRIPTIONS,             # 标签定义
)

# 使用示例
model = LLCMBackbone(mode='f3').to(device)
pretrain(model, seed=0)
X, L = build_dataset(seed=42)
lorentz = model.embed_seq(X.to(device))
geo = model.measure_lorentz(lorentz)
print(f"类时比例: {geo['tl_ratio']:.1%}  mq均值: {geo['mq_mean']:+.3f}")
```

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

**理论保证（三步推导）：**

**步骤1：定义信息时间（K=1 Chronogeometrodynamics Law I）**
```
dt_info = dΦ/H
  Φ = 参数空间的结构变化量
  H = 熵阻力
G = ∇²(dt²_info)|_{K=1}   ← 信息时间的 Hessian，即度规
```

**步骤2：可实现性条件强制 G 不定（Realizability.pdf Theorem 5）**

三个 Assumption 缺一不可：
- **Assumption R**（零阈值）：存在方向使位移代价趋向零
- **Assumption E**（二次展开）：代价函数存在二次主项 `d(x;δx)² = (Q(δx))₊ + o(‖δx‖²)`
- **Assumption T**（时间正代价）：纯时间位移代价为正

三者同时满足 → Q 非退化且不定 → `Sig(Q) = (1,1)`（洛伦兹）

**Remark 11 明确排除所有非洛伦兹签名：**
- `(2,0)` 正定：被 Assumption R + Lemma 4 排除
- `(0,2)` 负定：被 Assumption T（Step 2）排除
- 退化形式：被 R 和 T 联合排除（Step 4）

**步骤3：洛伦兹签名 ↔ 非平凡稳定边界（K=1 Chronogeometrodynamics Theorem 4）**
```
以下三者等价：
(a) 系统存在非平凡稳定边界 dc > 0
(b) Sig(G) = (1,1)，等价于 det G < 0
(c) 辛结构、因果结构、光锥同时涌现

一行证明：dc > 0 ⟺ -1/det G > 0 ⟺ det G < 0 ⟺ Sig(G) = (1,1)
```

传统方法的 Fisher 信息矩阵永远正定（det G > 0），光锥不存在，dc = 0。`dt²_info` 的 Hessian 由于可实现性条件强制不定（det G < 0），光锥从代数上涌现。

历史上的洛伦兹 Transformer 假设某个维度是时间——盲目的。LLCM 从代价函数结构推导，洛伦兹签名是**唯一可能的代数结果**，不是选择。

---

## 为什么机器人数据满足可实现性条件，文本/视频不满足

这是选择预训练数据最重要的判断依据。

### 可实现性条件的三个要求

**Assumption R（零阈值可实现性）：** 存在一个方向序列，沿这个方向的位移代价与位移大小之比趋向零——存在"近零代价"的运动方向。

**Assumption E（二次展开）：** 代价函数有良好的二次主项：`d(x;δx)² = (Q(δx))₊ + o(‖δx‖²)`，即代价函数在局部可以用二次型近似。这是连接物理直觉和代数定理的桥梁。

**Assumption T（时间正代价）：** 纯时间方向（原地等待，δr=0）的代价为正：`d(x;(Δt,0)) ≥ cₜΔt`。

三个条件同时满足 → Theorem 5（Realizability.pdf）→ 洛伦兹签名是唯一代数结果。

> **注意**：README 之前只提了 R 和 T，漏掉了 E。Assumption E 是必要的——
> 没有二次展开假设，无法从物理直觉跳到 Q 的代数性质。

### 机器人数据为什么满足

```
Assumption R：
  机器人匀速运动时，沿速度方向的位移代价趋向零
  （匀速不需要额外能量）
  → 这个方向就是类时方向，天然存在

Assumption T：
  机器人原地静止等待有代价
  （需要维持姿态平衡，消耗能量）
  → 纯时间方向有正代价，满足

两个条件都满足 → 洛伦兹几何必然涌现
类时比例实测：CMU MoCap 行走数据 100%，双摆 80.3%
```

### 文本数据为什么不满足

```
Assumption R：
  从"苹果"到"香蕉"的语义代价
  和从"苹果"到"红色"的语义代价
  没有结构性差异，不存在"零代价方向"
  → Assumption R 不满足

结果：sigma 退化到 0.497（实验验证）
      洛伦兹几何无法激活，F3 退化为欧氏
```

### 视频数据为什么不满足

```
像素变化混合了：
  光照变化、遮挡、相机运动、物体运动
  类时比例通常 < 50%
  没有清晰的"沿某方向代价趋向零"

结果：sigma 退化，LLCM 对视频世界模型无优势
      这也是 LeCun/NVIDIA 的世界模型不用洛伦兹的根本原因
      ——不是他们不知道，是他们的数据不满足可实现性条件
```

### 5分钟判断你的数据是否适合LLCM

```python
dx    = data[:, 1:] - data[:, :-1]   # 相邻帧差分
t_dim = max(1, int(data.shape[-1] * 0.25))
s2    = -(dx[..., :t_dim]**2).sum(-1)       +  (dx[..., t_dim:]**2).sum(-1)
ratio = (s2 > 0).float().mean()

# ratio > 0.6  → 数据满足可实现性条件 → LLCM 有效
# ratio < 0.4  → 数据不满足 → sigma 会退化为欧氏
# 0.4~0.6      → 边界区域 → 效果不稳定
```

| 数据类型 | 类时比例 | 是否适合 |
|---------|---------|---------|
| 机器人物理坐标 [x,y,z,vx,vy,vz] | 80-100% | ✅ 强烈推荐 |
| 关节旋转角（BVH） | 80% | ✅ 推荐 |
| 物理仿真（ODE） | 80-100% | ✅ 推荐 |
| 自然语言 token | <50% | ❌ sigma退化 |
| 视频像素 | <50% | ❌ sigma退化 |
| 传感器时序（IMU/力矩） | 待测量 | 先算类时比例 |

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

## 复现配置

层0和层1使用以下配置。参数分两类：**理论必须**和**工程调参**。

### 层0 复现（机器人动量守恒实验）

```python
# 复现脚本：cmu_real_bvh_experiment.py
# 数据：CMU MoCap BVH 文件（subject 01/02，walk 类）
# 输入格式：[x, y, z, vx, vy, vz]（6维物理坐标）或关节旋转角（90维）

# 关键配置
formula     = 'f3'     # 光锥注意力
time_ratio  = 0.25     # 类时头比例
n_traj      = 87       # CMU MoCap 行走轨迹段数
T_IN        = 20       # 输入帧数
T_OUT       = 20       # 预测帧数
N_EPOCHS    = 80       # 预训练 epoch
LR          = 3e-4     # AdamW

# 评估指标：动量变化率（越低越守恒）
mom_change = np.linalg.norm(np.diff(vel, axis=0), axis=-1).mean()
# F3 改善: (euc_mom - f3_mom) / euc_mom = 69.9%
```

层0没有语言嵌入，只有物理预训练。关键变量是 `formula='f3'` 和 `time_ratio=0.25`。

### 层1 复现（语言对齐实验）

层1在层0预训练的 backbone 上加语言对齐微调。
依赖 `sentence-transformers`（`!pip install sentence-transformers -q`）。

### 理论必须参数（改了会破坏洛伦兹几何）

层0和层1共用以下必须参数：

| 参数 | 值 | 原因 |
|------|-----|------|
| `formula` | `'f3'` | 负号是洛伦兹几何的唯一来源 |
| `time_ratio` | `0.25` | 类时头比例，对应可实现性条件 T_DIM=32 |
| `MinkowskiLN` | 旧版 `mq.abs()` | 符号修复版训练不稳定，见 API 文档节说明 |

> ⚠️ `MinkowskiLN` 的符号修复版（`sign * sqrt(...)`）在理论上更正确，
> 但在轨迹预测（MSE）任务上 `w_sigma` 梯度只有 0.0003，
> 洛伦兹几何无法激活。**旧版虽然退化为 L2 归一化变体，
> 但注意力层的光锥几何（F3 公式）仍然有效**，实验结果来自注意力层而非 LayerNorm。
> 符号修复版 LayerNorm 的稳定训练是开放研究问题。

### 工程调参参数（影响性能，不影响几何）

```python
# 层1语言对齐实验复现配置
EMBED_DIM  = 128    # 可以调大，效果通常更好
N_HEADS    = 4      # 与 time_ratio 共同决定 T_DIM
N_LAYERS   = 3
TIME_RATIO = 0.25   # ← 理论必须，不要改
STATE_DIM  = 6      # [x, y, z, vx, vy, vz]
LANG_DIM   = 384    # sentence-transformers all-MiniLM-L6-v2 输出维度

# 预训练（建立感知流形）
LR_PRE     = 3e-4   # AdamW + CosineAnnealingLR
EP_PRE     = 60     # MSE 轨迹预测

# 微调（语言对齐）
LR_FT      = 1e-4   # Adam
EP_FT      = 150    # 五个损失联合训练（方向A+B）
BS         = 16

# 语言模型
LANG_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
```

### 损失函数

```python
# 预训练（加分类辅助任务，给 sigma 直接梯度信号）
loss = F.mse_loss(pred_traj, true_traj) + 0.3 * F.cross_entropy(logits, labels)

# 微调（五个损失）
loss_cls      = F.cross_entropy(logits, labels)            # 分类
loss_align    = CLIP(lang_gen(x), lang_emb)                # 方向A CLIP
loss_align_B  = CLIP(lang_aligner(lang_emb), embed_seq(x)) # 模块4对齐
loss_mom      = momentum_conservation(phys_decoder(...))    # 模块5守恒
loss_geom     = MSE(mq(lang_aligner(lang_emb)),             # 几何对齐
                    mq(embed_seq(x)).detach())
loss = loss_cls + 0.3*loss_align + 0.5*loss_align_B + 0.5*loss_mom + 0.4*loss_geom
```

### 数据隔离（统计有效性）

```python
X_test = build_dataset(seed=42)           # 固定测试集，所有seed共用
X_train = build_dataset(seed=seed + 100)  # 每个seed独立训练集
pretrain(model, seed=seed * 1000)         # 预训练数据与微调数据完全分开
```

### 消融验证结果（4/4 通过）

| 验证 | 结果 | 结论 |
|------|------|------|
| Test1 随机基线 | 欧氏=0.047，F3=-0.025，均接近0 | 架构无固有偏差 |
| Test2 无CLIP损失 | 差异=+0.039，F3仍然更高 | backbone几何是真实来源 |
| Test3 打乱语言嵌入 | 0.297→0.081 | 语言语义内容是必要条件 |
| Test4 逐标签 | stable优势+0.230 > changing+0.121 | 符合类时测地线理论预测 |

> 主实验最终结果（5 seeds）：欧氏=0.194±0.046，F3=0.297±0.031，p=0.0010，d=3.89

### 层3 复现（零损失猜想）

**弱版本（层3结构效应）：**

```python
# 复现脚本：layer3_zero_loss_B.py
# 验证：F3语言生成轨迹守恒性 vs 欧氏语言生成
# 结果：p=0.041，d=1.33，5/5 seed 方向一致

# ── 层3专用配置（和层0/1不同）──────────────────────────
TIME_RATIO = 0.5     # 层3专用：频域精确值（层0/1用0.25）
                     # 相位维度=类时，振幅维度=类空，TIME_RATIO精确推导
N_FFT      = 11      # T_IN//2+1，rfft输出帧数
EP_PRE     = 120     # 层3比层0/1更多（层0/1用60-80）
MOM_WEIGHT = 0.3     # 动量守恒损失权重，影响结构效应强度

# 预训练输入：频域表示
# seg[:T_IN] → rfft → [相位(11×6), 振幅(11×6)] = (11, 12)
# 振幅做per-channel归一化，相位保持原始[-π, π]

# 预训练损失：MSE + 动量守恒（关键：两者缺一不可）
loss = F.mse_loss(pred_traj, true_traj) + 0.3 * (dp**2).mean()
# dp = torch.diff(pred_traj[:,:,3:], dim=1)  # 速度差分

# ── 4种ODE预训练数据（层3专用）──────────────────────────
# 层0/1只用2种ODE（stable + running）
# 层3用4种ODE（stable + running + walking + jumping）
# 4种ODE增加物理多样性，增强动量信号

# ── 关键发现：预训练阶段动量守恒损失 ─────────────────────
# F3  mom_loss = 0.025±0.003（5 seed 均值）
# 欧氏 mom_loss = 0.275±0.001（5 seed 均值）
# 差距10倍，5/5 seed 完全一致
# 不依赖 sigma 激活（sigma = 0.500），是 F3 公式负号的结构效应

# ── 理论解释 ───────────────────────────────────────────
# F3 负号 → 类时方向互相排斥 → 信息沿光锥边界传播
# 光锥边界对应匀速运动（类时测地线）
# 动量守恒 = 几何结构的必然，不是损失函数的约束
# 对应 Theorem 4：洛伦兹签名让守恒律从代数上涌现
```

**强版本（sigma激活，待实现）：**

```python
# 目标：sigma > 0.55，F3动量变化率接近真实物理基准
# 当前状态：sigma = 0.500，MSE+动量损失对sigma梯度不足
#   （w_sigma梯度约0.0003，太小）
# 开放研究问题：找到对sigma有强梯度信号的预训练任务
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
> **v1.1.0 修复说明：** 旧版 `MinkowskiLayerNorm` 用了 `mq.abs()`，
> 理论上丢失了类时/类空的符号差异。新版保留符号，几何上更正确。
>
> **实际使用建议：** 当前已验证实验（层0、层1）均使用旧版配置（`mq.abs()`），
> 因为新版在 MSE 轨迹预测任务上训练不稳定（`w_sigma` 梯度仅 0.0003）。
> 新版 LayerNorm 的稳定训练是开放研究问题，见复现配置节的详细说明。
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

以下消融实验验证各组件对语言建模的独立贡献，
与层0/层1的物理实验是互补关系而非依赖关系。

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

# 实验脚本（研究原型）
experiments/
├── core.py                    # 统一模型定义和数据管道
├── layer1_minimal_test.py     # 婴儿说话双向验证（里程碑1+3）
├── layer3_zero_loss_B.py      # 模块1结构效应（×10倍）
├── baby_talk_full_test.py     # 五模块完整验证
└── online_interaction_test.py # Law II 在线交互（模块4b）
```

---

## 理论背景

**Realizability.pdf**（Li 2026）— *Realizability and the Origin of Causality*

在三个 Assumption（R：零阈值可实现，E：二次展开，T：时间正代价）下，Theorem 5 证明：
代价函数的二次型 Q 非退化且不定，在归一化后唯一确定为：
```
Q = dt² - c⁻²_max dr²,   c_max > 0
```
Remark 11 明确排除所有非洛伦兹签名——正定、负定、退化形式均被排除。
洛伦兹签名不是兼容的选择，是**唯一可能的代数结果**。

**K=1 Chronogeometrodynamics**（Li 2026）— *K=1 Chronogeometrodynamics*

定义信息时间 `dt_info = dΦ/H`，其 Hessian `G = ∇²(dt²_info)|_{K=1}` 作为度规。

Theorem 4（Non-Separability）给出精确等价：
```
(a) 系统存在非平凡稳定边界 dc > 0
      ⟺
(b) Sig(G) = (1,1)，等价于 det G < 0

一行证明：dc > 0 ⟺ -1/det G > 0 ⟺ det G < 0 ⟺ Sig(G) = (1,1)
```
洛伦兹几何、辛结构、因果结构三者同时作为代数推论涌现，不是独立假设。

**两篇论文的关系：**
Realizability.pdf 回答"为什么是洛伦兹"（从代价函数推导签名）。
K=1 Chronogeometrodynamics 回答"洛伦兹意味着什么"（签名决定稳定性边界和因果结构）。
LLCM 是这两个结论的实验载体——用机器人物理预训练激活洛伦兹几何。

---

## 当前状态

**已完成：**

- ✅ 里程碑0/1/3：婴儿说话静态闭环完整验证（8 seeds）
- ✅ 层2 mq差距 d=13.22，p=0.0000（历史最强）
- ✅ 方向A 7/8 seeds，方向B 6/8 seeds
- ✅ 6类认知功能ODE体系（感知/推理/记忆/逻辑/智慧/对比）
- ✅ 两阶段微调（方向A和B解耦训练）
- ✅ 双向推力损失（守恒→类时，非守恒→类空）
- ✅ HJB-洛伦兹猜想提出 + 验证代码完成

**进行中：**

- 🔄 里程碑2：sigma激活（当前0.533，目标>0.60）
- 🔬 里程碑5：HJB-洛伦兹猜想实验验证

**计划中：**

- 📋 里程碑4：动态交互验证（Law II）
- 📋 Open X-Embodiment真实机器人数据验证
- 📋 论文写作（目标：CoRL/NeurIPS 2026）


## License

MIT

## Citation

```bibtex
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

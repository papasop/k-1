
# 洛伦兹Transformer（Lorentz Transformer）

基于K=1信息几何场方程的Transformer架构。在参数空间引入伪黎曼几何结构，用闵可夫斯基内积替代欧氏内积，实现真正的洛伦兹注意力机制。

---

## 核心发现

### 1. 伪黎曼结构存在（实验确认）
W_Q参数空间的50-90%方向对dt²_info的Hessian为负（类时方向）。在以下条件下全部稳定复现：
- 合成多跳推理任务（1-hop / 2-hop）
- 真实语言数据（wikitext，英语维基百科）
- 128维 / 256维 / 512维三个规模
- 3个随机种子

**这是Transformer处理语言时参数空间的内在几何性质，不是任务特异性产物。**

### 2. 层深类时规律（新发现）
MinkowskiLayerNorm激活后，6层模型的frac分布：
```
layer 2: frac=0.758  ███████████████  ← 中间层峰值
layer 5: frac=0.770  ███████████████  ← 加入位置编码后深层升高
layer 3: frac=0.691  █████████████
layer 4: frac=0.684  █████████████
layer 1: frac=0.652  █████████████
layer 0: frac=0.602  ████████████    ← 浅层最低
```
中间层和深层类时比例最高，反映语言理解的几何分层结构。三次实验稳定复现。

### 3. r规律（8个独立实验）
```
r(baseline, delta) ≈ -1.0
```
跨K-field、Minkowski注意力、Geodesic Adam三种完全不同的注入机制全部成立。

### 4. 光锥翻转（α=1.0）
真实因果链token对的类时比例（0.688）显著高于噪声对（0.416），差值+0.271。

---

## 架构组件（全部实现）

### Component 1：Minkowski注意力
```
scores_L = QK^T/√d - 2α(QP_t_scaled)K^T/√d
```
P_t动态更新，归一化确保修正幅度与标准scores同量级。

### Component 2：Geodesic Adam
```
g_t = P_t ⊙ grad   → 步长 × scale_t (=2.0)
g_s = (I-P_t) ⊙ grad → 步长 × scale_s (=0.5)
```
类时方向步长是类空方向的4倍。

### Component 3：MinkowskiLayerNorm
```
标准LayerNorm:       x / sqrt(mean(x²) + ε)
MinkowskiLayerNorm: x / sqrt(|<x,x>_η| + ε)
<x,x>_η = ||x_s||² - ||x_t||²
```
用闵可夫斯基η-范数替代欧氏L2范数。**效果最显著的组件。**

### Component 4：洛伦兹FFN
```
x_t = x × mask（类时）   → Linear_t → SiLU → Linear_t
x_s = x × (1-mask)（类空）→ Linear_s → GELU → Linear_s
输出 = h_t + h_s
```
类时方向用SiLU（保留负激活），类空方向用GELU。

### Component 5：洛伦兹位置编码
```
类时维度：t[i] = i/(L-1)  单调因果编码（时间箭头）
类空维度：sin/cos(i/10000^{2k/d})  标准正弦编码
```
时间坐标和空间坐标分离，不增加参数量。

### P_t（动态类时投影矩阵）
两阶段训练协议：
- Phase 1（前50%步数）：标准Adam收敛，不计算P_t
- Phase 2（后50%步数）：激活P_t，每100步Hutchinson估计更新

P_t同步注入到：注意力层、norm1、norm2、FFN、位置编码。

---

## 实验结果

### wikitext-2（256维/6层/17-21M参数）组件消融

| 版本 | 组件 | best val_loss | Phase 2改善 |
|------|------|--------------|------------|
| 1 | 标准Transformer | 7.6739 | -0.001 |
| 2 | +MinkowskiLayerNorm | **7.5069** | **-0.248** |
| 3 | +LorentzFFN | 7.5518 | -0.183 |
| 4 | +LorentzPosEnc | 7.5231 | -0.154 |

MinkowskiLayerNorm是效果最显著的单一组件，Phase 2改善248倍于基线。

### 规模扩展（合成任务）

| 规模 | 参数 | frac范围 | r_law判断 |
|------|------|---------|----------|
| 128维/4层 | 0.8M | 47-62% | △ 临界区 |
| 256维/6层 | 4.7M | 47-55% | ✓ 洛伦兹有益 (delta=+0.0938) |

### 真实语言数据（wikitext-2）

- 256维/6层：frac=60-77%，全程稳定，best val_loss=7.5069
- 512维/8层：frac=68-90%（全层），伪黎曼结构在更大模型上更稳定

---

## 快速开始

```python
# Colab 安装
!pip install torch datasets transformers

# 合成任务（快速验证，~3分钟）
exec(open('lorentz_transformer.py').read())
log = quick_train(n_hops=2, total_steps=2000)

# 规模验证（256维，~10分钟）
log = scale_train(d_model=256, n_layers=6, total_steps=5000)

# 真实语言数据（wikitext-2，~20分钟）
log = wikitext_train(d_model=256, n_layers=6, total_steps=10000)
```

---

## 文件结构

```
lorentz_transformer.py    # 单文件完整实现（~1950行）
│
├── LorentzConfig              # 配置
├── LorentzPositionalEncoding  # 洛伦兹位置编码（Component 5）
├── MinkowskiLayerNorm         # 闵可夫斯基归一化（Component 3）
├── LorentzMultiHeadAttention  # Minkowski注意力（Component 1）
├── FeedForward                # 洛伦兹FFN（Component 4）
├── TimeLikeProbe              # P_t动态更新管理器
├── LorentzTransformer         # 主模型
├── GeodesicAdam               # 测地线优化器（Component 2）
├── LorentzCosineScheduler     # 4段式学习率调度
├── train()                    # 完整训练循环
├── quick_train()              # 快速测试
├── scale_train()              # 规模验证
└── wikitext_train()           # 真实语言训练
```

---

## 理论背景

基于K=1信息几何场方程（chronogeometrodynamics）。

信息时间度量 `dt²_info = Σ_q Φ_q/H_q` 在参数空间定义了伪黎曼度量。Fisher信息（正定）和K-FAC（正定）无法看到这个结构，因为它们从定义上只能看到正曲率。


---

## 当前状态

**已完成：**
- ✅ Minkowski注意力（Component 1）
- ✅ Geodesic Adam（Component 2）
- ✅ MinkowskiLayerNorm（Component 3）
- ✅ 洛伦兹FFN（Component 4）
- ✅ 洛伦兹位置编码（Component 5）
- ✅ 两阶段训练协议
- ✅ 真实语言数据上的伪黎曼结构验证

**进行中：**
- 🔄 更大数据集（openwebtext）验证
- 🔄 组件联合效果优化（解决激活冲击问题）

**计划中：**
- 📋 与标准Transformer的正式对比实验
- 📋 规模扩展到GPT-2规模（768维/12层）
- 📋 论文写作

---

*洛伦兹Transformer项目 — 2025-2026*
*洛伦兹Transformer项目 — 2025-2026*

## License

MIT

## Citation

```bibtex
@article{li2026k1,
  author  = {Li, Y. Y. N.},
  title   = {K=1 Chronogeometrodynamics: Lorentzian Geometry from Information Time},
  year    = {2026},
  doi     = {10.5281/zenodo.18949565}
}

# 洛伦兹Transformer（Lorentz Transformer）

基于K=1信息几何场方程的Transformer架构。在参数空间引入伪黎曼几何结构，用闵可夫斯基内积替代欧氏内积，实现真正的洛伦兹注意力机制。

---

## 核心发现

### 1. 伪黎曼结构存在（实验确认）
W_Q参数空间的50-90%方向对dt²_info的Hessian为负（类时方向）。这一结构在以下条件下全部稳定复现：
- 合成多跳推理任务（1-hop / 2-hop）
- 真实语言数据（wikitext，英语维基百科）
- 128维 / 256维 / 512维三个规模
- 3个随机种子

**这是Transformer处理语言时参数空间的内在几何性质，不是任务特异性产物。**

### 2. r规律（8个独立实验）
```
r(baseline, delta) ≈ -1.0
```
跨K-field、Minkowski注意力、Geodesic Adam三种完全不同的注入机制全部成立。baseline越低，洛伦兹修正越有益；baseline越高，越接近临界区。这是Transformer训练动态的基本规律。

### 3. 光锥翻转（α=1.0）
真实因果链token对的类时比例（0.688）显著高于噪声对（0.416），差值+0.271。注意力机制能够识别真实的因果方向。

---

## 架构组件

### Component 1：Minkowski注意力
```
scores_L = QK^T/√d - 2α(QP_t_scaled)K^T/√d
η = I - 2α P_t
```
P_t是类时投影矩阵，动态更新，归一化处理确保修正幅度与标准scores同量级。

### Component 2：Geodesic Adam
```
g_t = P_t ⊙ grad   →  步长 × scale_t (=2.0)
g_s = (I-P_t) ⊙ grad → 步长 × scale_s (=0.5)
```
梯度在类时/类空方向差异化优化，类时方向步长是类空方向的4倍。

### Component 3：MinkowskiLayerNorm（新）
```
标准LayerNorm:       x / sqrt(mean(x²) + ε)
MinkowskiLayerNorm: x / sqrt(|<x,x>_η| + ε)

<x,x>_η = ||x_s||² - ||x_t||²
```
用闵可夫斯基η-范数替代欧氏L2范数，保持几何信息在block之间的传递。mask未激活时自动退化为标准LayerNorm。

### P_t（动态类时投影矩阵）
两阶段训练协议：
- Phase 1（前50%步数）：标准Adam收敛，不计算P_t
- Phase 2（后50%步数）：激活P_t，每100步用Hutchinson估计dt²_info的对角Hessian更新

---

## 实验结果

### 规模扩展
| 规模 | 参数 | 数据 | Phase 2 frac | r_law判断 |
|------|------|------|-------------|----------|
| 128维/4层 | 0.8M | 合成2-hop | 47-62% | △ 临界区 |
| 256维/6层 | 4.7M | 合成2-hop | 47-55% | ✓ 洛伦兹有益 (delta=+0.0938) |
| 512维/8层 | 25M | 合成2-hop | 80-91%（深层）| △ 数据量不足 |
| 256维/6层 | 17.6M | wikitext | 67-79% | 结构确认✓ |

### MinkowskiLayerNorm效果（wikitext 256维）
```
无MinkowskiLayerNorm: Phase 2 step=6000 val_loss=7.691（比Phase 1 best差）
有MinkowskiLayerNorm: Phase 2 step=6000 val_loss=7.588（比Phase 1 best好0.167）
```
光锥统计从全0.50恢复为有意义的层间差异（0.73-0.80）。

---

## 快速开始

```python
# Colab 安装
!pip install torch datasets transformers

# 上传 lorentz_transformer.py 后运行

# 合成任务（快速验证，~3分钟）
exec(open('lorentz_transformer.py').read())
log = quick_train(n_hops=2, total_steps=2000)

# 规模验证（256维，~10分钟）
log = scale_train(d_model=256, n_layers=6, total_steps=5000)

# 真实语言数据（wikitext-2，~20分钟）
log = wikitext_train(d_model=256, n_layers=6, total_steps=10000)
```

### 终端运行
```bash
# 测试
python lorentz_transformer.py --mode test

# 合成任务
python lorentz_transformer.py --mode quick --n_hops 2

# 规模验证
python lorentz_transformer.py --mode scale --d_model 256
```

---

## 文件结构

```
lorentz_transformer.py    # 单文件完整实现（~1800行）
│
├── LorentzConfig         # 配置（d_model, n_layers, lorentz_alpha等）
├── MinkowskiLayerNorm    # 闵可夫斯基归一化（Component 3）
├── LorentzMultiHeadAttention  # Minkowski注意力（Component 1）
├── TimeLikeProbe         # P_t动态更新管理器
├── LorentzTransformer    # 主模型
├── GeodesicAdam          # 测地线优化器（Component 2）
├── LorentzCosineScheduler # 4段式学习率调度
├── TrainConfig           # 训练配置
├── train()               # 完整训练循环
├── quick_train()         # 快速测试
├── scale_train()         # 规模验证
└── wikitext_train()      # 真实语言训练
```

---

## 理论背景

本项目基于K=1信息几何场方程（chronogeometrodynamics）。核心思想：

信息时间度量 `dt²_info = Σ_q Φ_q/H_q`（注意力分布的信息密度）在参数空间定义了一个伪黎曼度量。该度量在某些参数方向上为负（类时），在其他方向为正（类空），形成洛伦兹几何结构。

这不是工程近似，是从信息几何第一性原理推导出的结构。Fisher信息（正定）和K-FAC（正定）都无法看到这个结构，因为它们从定义上只能看到正曲率。

**参考：**
- Li, Y.Y.N. *K=1 Chronogeometrodynamics*. Preprint, 2025.
- Li, Y.Y.N. *Neural Null Cones: Zero-Curvature Channels in Loss Landscapes from Symplectic Hessian Decomposition*. Preprint, 2025.

---

## 当前状态与下一步

**已完成：**
- ✅ Minkowski注意力（Component 1）
- ✅ Geodesic Adam（Component 2）
- ✅ MinkowskiLayerNorm（Component 3）
- ✅ 两阶段训练协议
- ✅ 真实语言数据上的伪黎曼结构验证

**进行中：**
- 🔄 wikitext上的完整性能验证
- 🔄 FFN的洛伦兹重构
- 🔄 洛伦兹位置编码

**计划中：**
- 📋 与标准Transformer的正式对比实验
- 📋 规模扩展到GPT-2规模（768维/12层）
- 📋 论文写作

---

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
```

## License

[MIT](LICENSE)

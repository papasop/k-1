"""
core/ 模块 - Minkowski 几何基础

这是洛伦兹Transformer的第一个可上传GitHub的完整模块。
"""

# lorentz_transformer/core/

Component 1: Minkowski 注意力机制

## 模块内容

```
core/
├── __init__.py                  # 模块导出接口
├── attention.py                 # Minkowski多头注意力核心实现
│   ├── compute_dt2_info()       # K=1信息时间度量
│   ├── hutchinson_diag_hessian() # Hessian对角线估计
│   └── class LorentzMultiHeadAttention
└── README.md                    # 本文件
```

## 关键类和函数

### `LorentzMultiHeadAttention`

闵可夫斯基多头注意力机制。

**核心公式：**
```
scores_L = Q·η·K^T / √d_h
其中 η = I - 2α·P_t (Minkowski符号矩阵)

展开：
scores_L = (QK^T)/√d_h - 2α·(Q_t K^T)/√d_h
         = scores_std - 2α × 时间内积
```

**特点：**
- ✅ 完全兼容标准MultiHeadAttention（α=0时等价）
- ✅ 无mask时自动回退到标准注意力
- ✅ 支持加性attention mask（causal、padding等）
- ✅ 保存诊断信息（last_intervals用于光锥分析）

**使用示例：**
```python
from lorentz_transformer.core import LorentzMultiHeadAttention
from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 8
    lorentz_alpha: float = 0.25
    dropout: float = 0.1

config = Config()
attn = LorentzMultiHeadAttention(config)

# 前向传播
x = torch.randn(2, 128, 256)  # (batch, seq_len, d_model)
output, weights = attn(x)

# 注入类时掩码（由外部TimeLikeProbe提供）
mask = torch.randint(0, 2, (256,)).bool()
attn.set_timelike_mask(mask)

# 应用causal mask
L = 128
causal_mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1)
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
output_causal, weights_causal = attn(x, causal_mask)
```

### `compute_dt2_info(attn_w)`

计算K=1信息时间度量。

**用途：** 诊断注意力权重的信息几何性质

**公式：**
```
dt²_info = Σ_q K_q = Σ_q (Φ_q / H_q)

其中：
  H_q = -Σ_j a_qj·log(a_qj)    [Shannon熵]
  Φ_q = Σ_j a_qj²               [集中度]
  K_q = Φ_q / H_q               [信息时间密度]
```

**用法：**
```python
from lorentz_transformer.core import compute_dt2_info

attn_weights = torch.rand(2, 8, 128, 128)  # (batch, heads, seq, seq)
dt2_info = compute_dt2_info(attn_weights)
print(f"Information time metric: {dt2_info.item():.6f}")
```

### `hutchinson_diag_hessian(loss_fn, param, n_samples=20)`

用Hutchinson方法估计Hessian的对角元素。

**用途：** 识别参数空间中的类时方向（dt²_info的凹方向）

**数学原理：**
```
G_ii ≈ (1/K) Σ_k v_k[i] · (H·v_k)[i]

其中：
  v_k ~ Rademacher{±1}        [随机符号向量]
  H·v_k = ∂²loss/∂param²·v_k  [Hessian-向量乘积]

解释：
  G_ii < 0  ⟹  参数i是类时维度（凹方向）
  G_ii > 0  ⟹  参数i是类空维度（凸方向）
```

**用法：**
```python
from lorentz_transformer.core import hutchinson_diag_hessian

# 定义损失函数
def loss_fn():
    x = torch.randn(2, 128, 256)
    output, weights = attn(x)
    return compute_dt2_info(weights)

# 计算对角Hessian
W_Q = attn.q_proj.weight
G = hutchinson_diag_hessian(loss_fn, W_Q, n_samples=20)

# 识别类时维度
is_timelike = (G < 0)
timelike_frac = is_timelike.float().mean()
print(f"Timelike fraction: {timelike_frac:.1%}")
```

## 文件说明

### `attention.py` (~600行)

**Part 1: 辅助函数**
- `compute_dt2_info()` - K=1信息度量
- `hutchinson_diag_hessian()` - Hessian估计

**Part 2: LorentzMultiHeadAttention**
- 核心实现：scores_L = QK^T/√d - 2α(Q_t K^T/√d)
- 方法：
  - `__init__()` - 初始化
  - `set_timelike_mask()` - 注入类时掩码
  - `forward()` - 前向传播
  - `extra_repr()` - 调试信息

**Part 3: 单元测试**
- 基础功能测试
- Minkowski修正测试
- 注意力掩码测试
- 数值稳定性测试

### `__init__.py`

导出公共API：
```python
from lorentz_transformer.core import (
    LorentzMultiHeadAttention,
    compute_dt2_info,
    hutchinson_diag_hessian,
)
```

## 快速开始

### 安装

```bash
# 如果已pip install lorentz-transformer
from lorentz_transformer.core import LorentzMultiHeadAttention

# 或直接导入（开发模式）
import sys
sys.path.insert(0, 'path/to/lorentz-transformer')
from lorentz_transformer.core import LorentzMultiHeadAttention
```

### 5分钟示例

```python
import torch
from lorentz_transformer.core import LorentzMultiHeadAttention

# 创建模块
from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 8
    lorentz_alpha: float = 0.25
    dropout: float = 0.1

attn = LorentzMultiHeadAttention(Config())

# 前向传播
x = torch.randn(2, 128, 256)
output, weights = attn(x)
print(f"Output shape: {output.shape}")      # [2, 128, 256]
print(f"Weights shape: {weights.shape}")    # [2, 8, 128, 128]

# 注入类时掩码
mask = torch.randint(0, 2, (256,)).bool()
attn.set_timelike_mask(mask)

# Minkowski修正激活
output_lorentz, weights_lorentz = attn(x)
print(f"Lorentz attention applied!")
```

## 运行测试

```bash
# 使用pytest
pytest tests/test_attention.py -v

# 或直接运行
python lorentz_transformer/core/attention.py

# 输出示例：
# ======================================================================
# Testing LorentzMultiHeadAttention
# ======================================================================
# [Test 1] 基础前向传播（α=0, 无mask）
# ✓ Input shape: torch.Size([2, 128, 256])
# ✓ Output shape: torch.Size([2, 128, 256])
# ✓ Weights shape: torch.Size([2, 8, 128, 128])
# ✓ Test 1 passed
# ...
# ✅ All tests passed!
```

## 依赖

- torch >= 1.12
- numpy

## API参考

| 函数/类 | 参数 | 返回值 | 说明 |
|---------|------|--------|------|
| `compute_dt2_info()` | attn_w: (B,H,L,L) | scalar | K=1信息度量 |
| `hutchinson_diag_hessian()` | loss_fn, param, n_samples | tensor | 对角Hessian |
| `LorentzMultiHeadAttention()` | config | module | Minkowski注意力 |
| `.set_timelike_mask()` | mask: (d_model,) | None | 注入类时掩码 |
| `.forward()` | x, mask | (output, weights) | 前向传播 |

## 关键特性

✅ **完全兼容标准Transformer**
- α=0时等价于标准MultiHeadAttention
- 可直接替换任何Transformer的注意力层

✅ **数值稳定**
- 幅度归一化避免数值不稳定
- 安全的mask处理

✅ **诊断友好**
- 保存last_intervals用于光锥分析
- extra_repr()显示关键参数

✅ **研究友好**
- 详细的代码注释和数学公式
- 完整的单元测试
- 可独立使用，无需其他组件

## 扩展建议

这个core模块可以独立用于：
1. **研究** - 理解Minkowski几何在注意力中的作用
2. **集成** - 集成到现有Transformer实现中
3. **消融实验** - 对比α=0和α>0的效果
4. **可视化** - 分析注意力间隔和类时比例

## 引用

如果使用这个模块，请引用：
```bibtex
@software{lorentz_transformer_2024,
  title={Lorentz Transformer: Component 1 - Minkowski Attention},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/lorentz-transformer}
}
```

## 许可

MIT License

## 联系

有问题？提交Issue或Pull Request！

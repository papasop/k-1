# core/ 模块 - Minkowski 几何基础

这是洛伦兹 Transformer 的第一个可独立发布模块，提供 Minkowski 多头注意力及其诊断工具。

## 模块内容

```text
core/
├── __init__.py
├── attention.py
└── README.md
```

## 导出 API

```python
from lorentz_transformer.core import (
    LorentzMultiHeadAttention,
    compute_dt2_info,
    hutchinson_diag_hessian,
)
```

## `LorentzMultiHeadAttention`

闵可夫斯基多头注意力机制。

核心公式：

```text
scores_L = Q·η·K^T / √d_h
η = I - 2α·P_t

展开后：
scores_L = (QK^T)/√d_h - 2α·(Q_t K^T)/√d_h
         = scores_std - 2α × 时间内积
```

特点：

- α=0 时与标准 MultiHeadAttention 等价
- 未注入 timelike mask 时自动回退到标准注意力
- 支持加性 attention mask（causal、padding）
- 保存 `last_intervals` 与 `last_intervals_raw` 供诊断分析

### 使用示例

```python
import torch
from dataclasses import dataclass
from lorentz_transformer.core import LorentzMultiHeadAttention


@dataclass
class Config:
    d_model: int = 256
    n_heads: int = 8
    lorentz_alpha: float = 0.25
    dropout: float = 0.1


attn = LorentzMultiHeadAttention(Config())
x = torch.randn(2, 128, 256)
output, weights = attn(x)

mask = torch.randint(0, 2, (256,)).bool()
attn.set_timelike_mask(mask)

causal_mask = torch.triu(torch.full((128, 128), float("-inf")), diagonal=1)
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
output_causal, weights_causal = attn(x, causal_mask)
```

## `compute_dt2_info(attn_w)`

计算 K=1 信息时间度量：

```text
dt²_info = Σ_q K_q = Σ_q (Φ_q / H_q)
H_q = -Σ_j a_qj·log(a_qj)
Φ_q = Σ_j a_qj²
K_q = Φ_q / H_q
```

用途：诊断注意力权重的信息几何性质。

## `hutchinson_diag_hessian(loss_fn, param, n_samples=20)`

使用 Hutchinson 方法估计 Hessian 对角线：

```text
G_ii ≈ (1/K) Σ_k v_k[i] · (H·v_k)[i]
v_k ~ Rademacher{±1}
```

用途：识别参数空间中的类时方向与类空方向。

## 运行测试

```bash
python -m unittest tests.test_attention -v
python lorentz_transformer/core/attention.py
```

## 依赖

- torch >= 1.12
- numpy >= 1.20

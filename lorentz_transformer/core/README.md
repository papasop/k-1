# core/ 模块 - Minkowski 几何基础

这是洛伦兹 Transformer 的第一个可独立上传 GitHub 的完整模块。

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

闵可夫斯基多头注意力机制：

```text
scores_L = Q·η·K^T / √d_h
η = I - 2α·P_t

展开后：
scores_L = (QK^T)/√d_h - 2α·(Q_t K^T)/√d_h
```

### 特点

- α=0 时与标准多头注意力等价
- 未注入类时掩码时自动回退到标准注意力
- 支持加性 attention mask
- 保留 `last_intervals` 和 `last_intervals_raw` 诊断信息

### 使用示例

```python
from dataclasses import dataclass

import torch

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

mask = torch.randint(0, 2, (256,), dtype=torch.bool)
attn.set_timelike_mask(mask)

causal_mask = torch.triu(
    torch.full((128, 128), float("-inf")),
    diagonal=1,
).unsqueeze(0).unsqueeze(0)
output_causal, weights_causal = attn(x, causal_mask)
```

## `compute_dt2_info(attn_w)`

计算 K=1 信息时间度量：

```text
dt²_info = mean_q (Φ_q / H_q)

H_q = -Σ_j a_qj·log(a_qj)
Φ_q = Σ_j a_qj²
K_q = Φ_q / H_q
```

输入张量形状为 `(B, H, L, L)`，输出为一个标量张量。

## `hutchinson_diag_hessian(loss_fn, param, n_samples=20)`

使用 Hutchinson 方法估计 Hessian 对角线：

```text
G_ii ≈ (1/K) Σ_k v_k[i] · (H·v_k)[i]
```

适用于识别参数空间中的类时方向：

- `G_ii < 0`：类时维度
- `G_ii > 0`：类空维度

## 运行测试

```bash
python -m unittest tests.test_attention -v
```

如果环境中安装了 pytest，也可以运行：

```bash
pytest tests/test_attention.py -v
```

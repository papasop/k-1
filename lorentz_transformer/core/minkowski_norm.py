"""
lorentz_transformer/core/minkowski_norm.py

Component 3: Minkowski LayerNorm
"""

from typing import Union

import torch
import torch.nn as nn


class MinkowskiLayerNorm(nn.Module):
    """
    闵可夫斯基层归一化（Minkowski LayerNorm）。

    在伪黎曼流形中对特征进行归一化。使用闵可夫斯基内积而非欧氏内积。

    数学定义：
        output = γ * (x / ||x||_η) + β
        其中 ||x||_η = sqrt(||x_s||² - ||x_t||² + ε)

        x_s = x ⊙ (1 - timelike_mask)  # 类空部分
        x_t = x ⊙ timelike_mask        # 类时部分

    参数：
        d_model (int): 特征维度
        eps (float): 数值稳定性的小常数
        elementwise_affine (bool): 是否使用可学习缩放和偏移
        use_mean_shift (bool): 是否先进行均值平移
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_mean_shift: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.use_mean_shift = use_mean_shift
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer("weight", torch.ones(d_model))
            self.register_buffer("bias", torch.zeros(d_model))

        self.register_buffer(
            "timelike_mask",
            torch.zeros(d_model, dtype=torch.bool),
            persistent=False,
        )
        self._has_mask = False

    def set_timelike_mask(
        self, mask: Union[torch.Tensor, list, tuple]
    ) -> None:
        """
        设置类时掩码。

        Args:
            mask: 形状为 (d_model,) 的 bool 张量、list 或 tuple。
        """
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.to(
                device=self.timelike_mask.device,
                dtype=torch.bool,
            )
        elif isinstance(mask, (list, tuple)):
            mask_tensor = torch.tensor(
                mask, dtype=torch.bool, device=self.timelike_mask.device
            )
        else:
            raise TypeError(f"mask 必须是 Tensor、list 或 tuple，得到 {type(mask)}")

        if mask_tensor.numel() != self.d_model:
            raise ValueError(
                f"mask 维度必须为 {self.d_model}，得到 {mask_tensor.numel()}"
            )

        self.timelike_mask.copy_(mask_tensor.view(self.d_model))
        self._has_mask = bool(self.timelike_mask.any().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 任意末维为 d_model 的张量

        Returns:
            与输入形状相同的归一化张量
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"输入末维必须为 {self.d_model}，得到 {x.shape[-1]}"
            )

        original_shape = x.shape
        compute_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        x_reshaped = x.reshape(-1, self.d_model).to(compute_dtype)

        if self.use_mean_shift:
            x_reshaped = x_reshaped - x_reshaped.mean(dim=-1, keepdim=True)

        timelike_mask = self.timelike_mask.to(
            device=x_reshaped.device, dtype=compute_dtype
        )
        spacelike_mask = 1.0 - timelike_mask

        x_spacelike = x_reshaped * spacelike_mask.unsqueeze(0)
        x_timelike = x_reshaped * timelike_mask.unsqueeze(0)

        norm_spacelike_sq = (x_spacelike ** 2).sum(dim=-1, keepdim=True)
        norm_timelike_sq = (x_timelike ** 2).sum(dim=-1, keepdim=True)

        norm_sq = norm_spacelike_sq - norm_timelike_sq
        norm_sq = torch.nan_to_num(norm_sq, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.sqrt(norm_sq.abs() + self.eps)

        normalized = x_reshaped / norm

        weight = self.weight.to(device=x_reshaped.device, dtype=compute_dtype)
        bias = self.bias.to(device=x_reshaped.device, dtype=compute_dtype)
        output = normalized * weight.unsqueeze(0) + bias.unsqueeze(0)

        return output.to(dtype=x.dtype).reshape(original_shape)

    def extra_repr(self) -> str:
        """模块的额外信息表示。"""
        return (
            f"d_model={self.d_model}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, "
            f"use_mean_shift={self.use_mean_shift}, "
            f"has_mask={self._has_mask}"
        )

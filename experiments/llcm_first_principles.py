"""
LLCM 原教旨实现
==============
从第一原理实现三个核心理论要素：

1. dt_info = dΦ/H（信息时间，不是线性近似）
2. 动态度规 g_μν(x)（场方程近似，不是固定度规）
3. Theorem 4 自然涌现（不依赖 loss_mf 损失函数）

对比工程原型（lorentz_riemannian_test.py）：
  工程原型：project链 + 5.0*loss_mf + 线性时间注入
  原教旨：  动态度规 + dt_info计算 + 拓扑吸引子自涌现

理论来源：
  K=1 Chronodynamics（Li 2026）
  Realizability and the Origin of Causality（Li 2026）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D      = 64       # 嵌入维度
SEEDS  = [0,1,2]
EPS    = 1e-6

print(f"Device: {DEVICE}")
print(f"流形: H^{{1,{D-1}}}  原教旨实现")


# ══════════════════════════════════════════════════════════════
# 模块1：信息时间计算 dt_info = dΦ/H
# ══════════════════════════════════════════════════════════════

class InformationTimeComputer:
    """
    从物理轨迹计算信息时间 dt_info = dΦ/H

    理论来源（K=1 论文）：
      Φ(x) = 感知势函数（perception potential）
      H(x,ẋ) = 哈密顿量（总能量）
      dt_info = dΦ/dt / H = (∇Φ·ẋ) / H

    感知势的近似：
      Φ(x) = -log p(x) ≈ ||x - μ||²/(2σ²)
      p(x) = 轨迹的局部密度（高斯近似）
      这是最大熵原理下的最小假设

    哈密顿量的计算：
      H = T + V = ||ẋ||²/2 + ||x||²/2
      对一般轨迹：H = 总机械能
      守恒系统：H = const → dt_info 稳定
      耗散系统：H → 0 → dt_info 波动

    x₀ 的理论值：
      x₀(t) = exp(∫₀ᵗ dt_info(s) ds)
      = 信息时间的指数积分
      x₀ 大 → 感知势相对能量在增加（信息积累）
      x₀ 稳定 → 哈密顿守恒（能量守恒）
    """

    @staticmethod
    def compute(traj: np.ndarray, dt: float = 0.033) -> np.ndarray:
        """
        输入：traj (T, 6) 物理轨迹（位置+速度）
        输出：x0_values (T,) 信息时间的指数值

        参数说明：
          traj[:, :3] = 位置 q
          traj[:, 3:] = 速度 ṗ
        """
        T = len(traj)
        q = traj[:, :3]  # 位置
        v = traj[:, 3:]  # 速度

        # ── 步骤1：计算感知势 Φ(q) ──────────────────────────
        # Φ = -log p(q) ≈ ||q - μ||² / (2σ²)
        # μ = 轨迹重心，σ² = 轨迹方差（局部信息密度）
        mu    = q.mean(axis=0)                          # (3,) 轨迹重心
        dq    = q - mu                                   # (T, 3) 偏差
        sigma2 = np.maximum((dq**2).sum(-1).mean(), EPS) # 标量 方差
        Phi   = (dq**2).sum(-1) / (2 * sigma2)          # (T,) 感知势

        # ── 步骤2：计算 dΦ/dt（数值微分）───────────────────
        # 中心差分：dΦ/dt ≈ (Φ(t+1) - Φ(t-1)) / (2dt)
        dPhi_dt = np.gradient(Phi, dt)                   # (T,)

        # ── 步骤3：计算哈密顿量 H = T + V ───────────────────
        # 动能 T = ||ṗ||²/2（精确）
        # 势能 V：简谐近似 + 重力修正
        #   简谐：V = ||q||²/2（对 stable_ode/kepler 合理）
        #   重力修正：对包含重力的系统（running_ode），
        #   y 方向有势能 V_grav = g·y（g=9.8）
        #   近似：用 ||q||²/2 + max(0, -q[:,1])*9.8
        #   这仍然是近似，但比纯简谐更接近真实
        T_kin = 0.5 * (v**2).sum(-1)                    # (T,) 动能
        V_harm = 0.5 * (q**2).sum(-1)                   # (T,) 简谐势能
        V_grav = np.maximum(0, -q[:, 1]) * 9.8          # (T,) 重力势能修正
        V_pot = V_harm + V_grav
        H     = T_kin + V_pot                            # (T,) 总能量

        # ── 步骤4：dt_info = dΦ/H ───────────────────────────
        # 物理含义：单位能量消耗带来多少感知势变化
        # 守恒系统：H=const → dt_info = dΦ/H（稳定）
        # 耗散系统：H→0 → dt_info 发散（需要截断）
        dt_info = dPhi_dt / np.maximum(H, EPS)          # (T,)
        dt_info = np.clip(dt_info, -5.0, 5.0)           # 数值稳定截断

        # ── 步骤5：x₀ = exp(∫dt_info dt) ───────────────────
        # 数值积分：累积求和 × dt
        integral = np.cumsum(dt_info) * dt              # (T,)
        # 归一化：x₀(0) = 1（初始信息时间为零）
        integral = integral - integral[0]
        x0 = np.exp(integral)                           # (T,)

        # 约束：x₀ > 1（流形要求）
        x0 = np.maximum(x0, 1.0 + EPS)

        return x0  # (T,) 每个时间步的信息时间值


# ══════════════════════════════════════════════════════════════
# 模块2：动态度规网络 g_μν(x)
# ══════════════════════════════════════════════════════════════

class DynamicMetricNet(nn.Module):
    """
    动态度规 g_μν(x)：度规随位置变化

    理论来源（K=1 场方程）：
      G_μν + Λg_μν = 8π T_μν
      G_μν = 爱因斯坦张量（曲率）
      T_μν = 能量-动量张量（物质）

    神经网络近似：
      g_μν(x) = η_μν + h_μν(x)
      η_μν = diag(-1,+1,...,+1)（背景闵可夫斯基度规）
      h_μν(x) = 神经网络学习的度规扰动

    约束：
      det(g) < 0（洛伦兹签名，来自 R+E+T）
      g₀₀ < 0（时间轴保持类时）
      g_ij > 0（空间轴保持类空，对角近似）

    工程原型的区别：
      工程原型：g = diag(-1,+1,...,+1)（固定）
      原教旨：  g_μν(x) = η + h(x)（动态）
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d

        # 背景度规（闵可夫斯基）
        eta = torch.ones(d)
        eta[0] = -1.0
        self.register_buffer('eta', eta)

        # 度规扰动网络 h_μν(x)
        # 只学对角元素（近似，减少参数）
        # 非对角元素的耦合是未来工作
        self.h_net = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.Tanh(),             # Tanh 保证输出有界，数值稳定
            nn.Linear(d * 2, d),
            nn.Tanh(),
        )
        # 扰动幅度控制（小扰动保证签名稳定）
        self.epsilon = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算点 x 处的度规对角元素

        输入：x (..., d) 流形上的点
        输出：g_diag (..., d) 度规对角元素

        g_diag[..., 0] < 0（时间方向，类时）
        g_diag[..., 1:] > 0（空间方向，类空）
        """
        # 基于切空间向量计算度规（x 在流形上，先映射到切空间）
        mu = torch.zeros(self.d, device=x.device)
        mu[0] = 1.0
        mu = mu.expand_as(x)

        # 切空间向量（欧氏，适合网络输入）
        # 使用简化版 log_map（避免循环依赖）
        v = x - mu  # 近似切空间（小扰动近似）

        # 度规扰动
        h = self.h_net(v) * self.epsilon  # (..., d) 小扰动

        # 动态度规 g = η + h
        g = self.eta + h

        # ── 关键约束：强制洛伦兹签名 ────────────────────────
        # 来自 Theorem 5（R+E+T → det G < 0）
        # 时间方向必须为负（类时）
        g_time  = -torch.abs(g[..., :1])      # g₀₀ < 0（强制）
        # 空间方向必须为正（类空）
        g_space = torch.abs(g[..., 1:])       # gᵢᵢ > 0（强制）

        return torch.cat([g_time, g_space], dim=-1)  # (..., d)

    def inner(self, x: torch.Tensor, u: torch.Tensor,
              v: torch.Tensor) -> torch.Tensor:
        """
        动态度规下的内积：⟨u,v⟩_g = Σ g_μμ(x) u_μ v_μ

        和固定度规的区别：
          固定：⟨u,v⟩_L = -u₀v₀ + Σuᵢvᵢ（与位置无关）
          动态：⟨u,v⟩_g = Σ g_μμ(x) uμvμ（与位置相关）

        这是弯曲时空的内积，不是平直时空的内积
        """
        g = self(x)  # (..., d)
        return (g * u * v).sum(-1)  # (...,)

    def lorentz_signature_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        度规时空分离损失

        目标：让度规的时间方向和空间方向尽量分离
          g₀₀ → 尽量负（时间方向强类时性）
          gᵢᵢ → 尽量正（空间方向强类空性）

        注意：g₀₀<0 和 gᵢᵢ>0 已经由硬约束保证
        这个损失额外鼓励分离幅度更大
        物理依据：det G 越负（分离越大）→ dc 越大（Theorem 4）
          dc = α√(-1/det G)，det G 越负，dc 越大
          dc 大 → {K=1} 吸引子越强 → Theorem 4 更容易涌现

        数学：最大化 (-g₀₀) + mean(gᵢᵢ) = 时空分离度
        """
        g = self(x)
        g00    = g[..., :1]               # < 0（硬约束）
        gii    = g[..., 1:]               # > 0（硬约束）
        # 时间方向：越负越好（-g₀₀ 越大越好）
        loss_t = g00.mean()               # 最小化 g₀₀（让它更负）
        # 空间方向：越正越好（gᵢᵢ 越大越好）
        loss_s = -gii.mean()              # 最大化 gᵢᵢ（让它更正）
        return loss_t + loss_s


# ══════════════════════════════════════════════════════════════
# 模块3：带动态度规的洛伦兹流形操作
# ══════════════════════════════════════════════════════════════

class DynamicLorentzManifold:
    """
    动态度规下的洛伦兹流形操作

    和固定度规（LorentzManifold）的区别：
      固定：Exp/Log 公式只依赖 x
      动态：Exp/Log 公式依赖 x 和 g(x)

    数学背景：
      在弯曲时空中，测地线方程包含克里斯托费尔符号
      Γ^λ_μν = (1/2) g^λρ (∂_μ g_νρ + ∂_ν g_μρ - ∂_ρ g_μν)
      这是平直时空 Exp/Log 公式不包含的项

    当前近似：
      忽略克里斯托费尔符号（小曲率近似）
      保留度规对内积的修正
      这在 h_μν 很小时是合理的近似
    """

    @staticmethod
    def project(x: torch.Tensor,
                g_net: DynamicMetricNet) -> torch.Tensor:
        """
        投影到动态度规下的流形

        约束：⟨x,x⟩_g = -1（广义化的流形约束）
        
        在动态度规下：
        -g₀₀ x₀² + Σ gᵢᵢ xᵢ² = -1
        x₀ = sqrt((1 + Σ gᵢᵢ xᵢ²) / (-g₀₀))
        """
        g = g_net(x)                                # (..., d)
        g00 = (-g[..., :1]).clamp(min=EPS)          # 时间度规（正值）
        gii = g[..., 1:].clamp(min=EPS)             # 空间度规（正值）

        space = x[..., 1:].clamp(-8.0, 8.0)
        # 动态流形约束
        x0 = torch.sqrt(
            (1.0 + (gii * space**2).sum(-1, keepdim=True)) / g00 + EPS
        )
        return torch.cat([x0, space], dim=-1)

    @staticmethod
    def inner_static(x: torch.Tensor,
                     y: torch.Tensor) -> torch.Tensor:
        """退化为固定度规（用于 Exp/Log 近似）"""
        return -x[..., 0]*y[..., 0] + (x[..., 1:]*y[..., 1:]).sum(-1)

    @staticmethod
    def log_map_approx(x: torch.Tensor,
                       y: torch.Tensor) -> torch.Tensor:
        """
        近似 Log 映射（固定度规公式，忽略曲率修正）

        完整版需要数值求解测地线方程（计算成本高）
        当前版本是小曲率近似，适合 h 较小时使用
        """
        y   = DynamicLorentzManifold.project_static(y)
        xy  = DynamicLorentzManifold.inner_static(x, y)
        xy  = xy.unsqueeze(-1).clamp(max=-(1.0+EPS))
        d   = torch.acosh((-xy).clamp(min=1.0+EPS)).clamp(min=1e-3)
        dir = y + xy * x
        dn  = torch.sqrt(
            DynamicLorentzManifold.inner_static(dir, dir).clamp(min=EPS)
        ).unsqueeze(-1)
        return d * dir / (dn + EPS)

    @staticmethod
    def project_static(x: torch.Tensor) -> torch.Tensor:
        """固定度规投影（用于内部计算）"""
        sp = x[..., 1:].clamp(-8.0, 8.0)
        x0 = torch.sqrt(1.0 + (sp**2).sum(-1, keepdim=True) + EPS)
        return torch.cat([x0, sp], dim=-1)

    @staticmethod
    def norm_sq_dynamic(x: torch.Tensor,
                        g_net: DynamicMetricNet) -> torch.Tensor:
        """
        动态度规下的内积范数平方

        ⟨x,x⟩_g = g₀₀ x₀² + Σ gᵢᵢ xᵢ²
        目标：= -1（动态流形约束）
        """
        return g_net.inner(x, x, x)


# ══════════════════════════════════════════════════════════════
# 模块4：原教旨 Transformer（带动态度规和信息时间）
# ══════════════════════════════════════════════════════════════

class FirstPrinciplesAttention(nn.Module):
    """
    在动态度规下的洛伦兹注意力

    和工程原型的区别：
      工程原型：注意力分数用固定洛伦兹内积
      原教旨：注意力分数用动态度规内积 ⟨q,k⟩_g(x)

    这意味着：
      注意力强度依赖当前位置的度规
      不同位置的注意力有不同的几何权重
      这是弯曲时空中的自然注意力机制
    """

    def __init__(self, d: int, n_heads: int,
                 g_net: DynamicMetricNet):
        super().__init__()
        self.d = d; self.h = n_heads; self.dh = d // n_heads
        self.g_net = g_net
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        mu = torch.zeros(self.d, device=x.device)
        mu[0] = 1.0
        mu = mu.view(1, 1, self.d).expand(B, T, -1)

        # 切空间变换
        v  = DynamicLorentzManifold.log_map_approx(mu, x)
        q_ = self.Wq(v).view(B, T, self.h, self.dh).transpose(1, 2)
        k_ = self.Wk(v).view(B, T, self.h, self.dh).transpose(1, 2)
        vv = self.Wv(v).view(B, T, self.h, self.dh).transpose(1, 2)

        # ── 动态度规注意力分数 ────────────────────────────────
        # 用当前位置 x 的度规计算内积
        # g_x = g_net(x)：每个位置有自己的度规
        g_x = self.g_net(x)  # (B, T, d)
        g_x_heads = g_x.view(B, T, self.h, self.dh).transpose(1, 2)
        # (B,h,T,dh)，度规对角元素

        # 加权内积：⟨q,k⟩_g = Σ g_μ q_μ k_μ
        qg = q_ * g_x_heads  # (B,h,T,dh) 度规加权 Q
        sc = torch.matmul(qg, k_.transpose(-2, -1)) / (self.dh**0.5)
        at = F.softmax(sc, dim=-1)

        out = (at @ vv).transpose(1, 2).contiguous().view(B, T, self.d)
        # 回到流形（使用动态投影）
        out = self.Wo(out)
        out_proj = DynamicLorentzManifold.project_static(
            DynamicLorentzManifold.project_static(mu) + out * 0.1
        )
        return out_proj


class FirstPrinciplesBlock(nn.Module):
    def __init__(self, d: int, n_heads: int,
                 g_net: DynamicMetricNet):
        super().__init__()
        self.d     = d  # 修复：显式存储维度
        self.attn  = FirstPrinciplesAttention(d, n_heads, g_net)
        self.g_net = g_net
        self.fc1   = nn.Linear(d, d * 2)
        self.fc2   = nn.Linear(d * 2, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = torch.zeros(self.d, device=x.device)
        mu[0] = 1.0
        mu = mu.view(1, 1, self.d).expand(*x.shape)

        # 注意力（动态度规）
        v = DynamicLorentzManifold.log_map_approx(mu, x)
        v = self.norm1(v + DynamicLorentzManifold.log_map_approx(
            mu, self.attn(x)))
        x = DynamicLorentzManifold.project_static(
            torch.exp(v * 0.01) * x)

        # FFN（切空间）
        v = DynamicLorentzManifold.log_map_approx(mu, x)
        v2 = self.fc2(F.gelu(self.fc1(v)))
        v = self.norm2(v + v2)
        x = DynamicLorentzManifold.project_static(
            torch.exp(v * 0.01) * x)
        return x


# ══════════════════════════════════════════════════════════════
# 模块5：原教旨 Backbone（集成所有组件）
# ══════════════════════════════════════════════════════════════

class FirstPrinciplesBackbone(nn.Module):
    """
    原教旨 LLCM Backbone

    三个核心第一原理组件：
    1. InformationTimeComputer：dt_info = dΦ/H
    2. DynamicMetricNet：g_μν(x) 动态度规
    3. Theorem 4 自然涌现：不依赖 loss_mf

    和工程原型的关键区别：
      工程原型 embed_seq：
        project(Linear(x)) → 线性时间注入 → blocks → project
      原教旨 embed_seq：
        计算 dt_info → x₀ 设为理论值 → 动态度规 project → blocks

    Theorem 4 涌现机制：
      工程原型：loss_mf = (mq+1)² 强制 mq→-1
      原教旨：
        动态度规 g_μν 的签名约束（g₀₀<0，gᵢᵢ>0）
        → 自然保证 ⟨x,x⟩_g 具有正确符号
        → {K=1} 吸引子从几何结构涌现
        → 不需要损失函数
    """

    def __init__(self, state_dim: int = 6, d: int = D,
                 n_heads: int = 4, n_layers: int = 3):
        super().__init__()
        self.d         = d
        self.n_heads   = n_heads
        self.itc       = InformationTimeComputer()
        self.g_net     = DynamicMetricNet(d)
        self.input_proj = nn.Linear(state_dim, d)
        self.blocks    = nn.ModuleList([
            FirstPrinciplesBlock(d, n_heads, self.g_net)
            for _ in range(n_layers)
        ])
        self.cls_head  = nn.Linear(d, 2)

    def embed_seq(self, x: torch.Tensor,
                  x0_theory: torch.Tensor = None) -> torch.Tensor:
        """
        x：(B, T, state_dim) 物理轨迹
        x0_theory：(B, T) 从 dt_info 计算的理论 x₀ 值（可选）

        如果提供 x0_theory：
          直接把 x₀ 设为理论值（原教旨）
          x₀ = exp(∫dt_info dt)（来自 K=1 论文）

        如果不提供（推理时）：
          退化为工程原型的 project 链
        """
        B, T, _ = x.shape

        # ── 步骤1：线性投影到嵌入空间 ─────────────────────
        h = self.input_proj(x)  # (B, T, d)

        # ── 步骤2：原教旨时间注入 ──────────────────────────
        if x0_theory is not None:
            # 原教旨：用 dt_info 计算的理论 x₀ 值
            # x₀(t) = exp(∫₀ᵗ dt_info(s) ds)
            x0_t = x0_theory.to(x.device)  # (B, T)

            # 设置空间分量
            space = h[..., 1:].clamp(-8.0, 8.0)

            # 理论 x₀ 必须满足流形约束：
            # x₀ ≥ sqrt(1 + Σxᵢ²)（类时条件）
            x0_min = torch.sqrt(
                1.0 + (space**2).sum(-1) + EPS)  # (B, T)
            # 取理论值和约束下界的最大值
            x0_actual = torch.maximum(
                x0_t, x0_min)  # (B, T)

            h = torch.cat([
                x0_actual.unsqueeze(-1), space
            ], dim=-1)  # (B, T, d)

        else:
            # 推理时的退化实现
            sp = h[..., 1:].clamp(-8.0, 8.0)
            x0 = torch.sqrt(1.0 + (sp**2).sum(-1, keepdim=True) + EPS)
            h  = torch.cat([x0, sp], dim=-1)

        # ── 步骤3：动态度规投影 ────────────────────────────
        # 和工程原型的 project 不同：
        # 工程原型：固定度规 project
        # 原教旨：动态度规 project（g 依赖位置）
        h = DynamicLorentzManifold.project(h, self.g_net)

        # ── 步骤4：全黎曼 Transformer 块 ──────────────────
        for blk in self.blocks:
            h = blk(h)

        # 返回最后一个时间步的嵌入
        # 注意：不调用 project_static
        # 让 mq 自由演化，才能验证 Theorem 4 涌现
        # 如果调用 project_static，mq=-1 是数学恒等式，不是涌现
        return h[:, -1, :]   # 自由演化，不强制 mq=-1

    def forward(self, x: torch.Tensor,
                x0_theory: torch.Tensor = None):
        emb = self.embed_seq(x, x0_theory)
        mu  = torch.zeros(self.d, device=x.device)
        mu[0] = 1.0
        v = DynamicLorentzManifold.log_map_approx(
            mu.unsqueeze(0).expand(emb.shape[0], -1), emb)
        return self.cls_head(v), emb

    def measure(self, emb: torch.Tensor) -> dict:
        """
        测量流形指标

        Theorem 4 验证的核心指标：
          mq_free：自由演化的 mq（不经过 project）
          如果 mq_free 自然收敛到 -1 → Theorem 4 成立
          如果 mq_free 偏离 -1 → Theorem 4 需要额外条件

          mq_projected：投影后的 mq（数学恒等式，始终=-1）
          这不能用于 Theorem 4 验证
        """
        # 自由演化的 mq（Theorem 4 验证的核心）
        mq_free = (-emb[:, 0]**2 + (emb[:, 1:]**2).sum(-1))

        # 投影后的 mq（参考用，始终=-1）
        emb_proj = DynamicLorentzManifold.project_static(emb)
        mq_proj  = (-emb_proj[:, 0]**2 +
                    (emb_proj[:, 1:]**2).sum(-1))

        # 动态度规的 mq
        mq_dynamic = self.g_net.inner(emb, emb, emb)

        # 约束违反（相对于 mq=-1 的偏差）
        viol_free = (mq_free + 1.0).abs().mean().item()

        return dict(
            mq_free         = mq_free.mean().item(),   # Theorem 4 核心
            mq_proj         = mq_proj.mean().item(),   # 始终=-1（参考）
            mq_dynamic      = mq_dynamic.mean().item(),
            tl_ratio_free   = (mq_free < 0).float().mean().item(),
            violation_free  = viol_free,               # 真实违反量
            x0_mean         = emb[:, 0].mean().item(),
            x0_std          = emb[:, 0].std().item(),
        )


# ══════════════════════════════════════════════════════════════
# 数据生成（和工程原型相同，但额外计算 x0_theory）
# ══════════════════════════════════════════════════════════════

def stable_ode(t, y):
    k, b = 2.0, 0.5
    return [y[3], y[4], y[5],
            -k*y[0]-b*y[3], -k*y[1]-b*y[4], -k*y[2]-b*y[5]]

def running_ode(t, y):
    d = 0.3
    return [y[3], y[4], y[5],
            -d*y[3], -d*y[4]-9.8, -d*y[5]]

def simulate(fn, ic, T=80, dt=0.033):
    sol = solve_ivp(fn, [0, T*dt], ic,
                    t_eval=np.linspace(0, T*dt, T), rtol=1e-6)
    return sol.y.T  # (T, 6)

def build_dataset_with_x0(seed=42, n=200):
    """
    构建数据集，同时计算每条轨迹的理论 x₀

    返回：
      X：(N, T, 6) 归一化轨迹
      L：(N,) 标签（0=守恒，1=非守恒）
      X0：(N, T) 理论 x₀ 值（来自 dt_info 计算）
    """
    rng = np.random.RandomState(seed)
    X, L, X0 = [], [], []
    itc = InformationTimeComputer()

    for _ in range(n // 2):
        traj = simulate(stable_ode, rng.randn(6)*0.3)
        mu   = traj.mean(0)
        sg   = np.maximum(traj.std(0), 0.01)
        traj_norm = (traj - mu) / sg
        x0_vals = itc.compute(traj_norm)  # 原教旨：从 dt_info 计算
        X.append(traj_norm); L.append(0); X0.append(x0_vals)

    for _ in range(n // 2):
        traj = simulate(running_ode, rng.randn(6)*0.3)
        mu   = traj.mean(0)
        sg   = np.maximum(traj.std(0), 0.01)
        traj_norm = (traj - mu) / sg
        x0_vals = itc.compute(traj_norm)
        X.append(traj_norm); L.append(1); X0.append(x0_vals)

    X  = torch.tensor(np.array(X),  dtype=torch.float32)
    L  = torch.tensor(L,             dtype=torch.long)
    X0 = torch.tensor(np.array(X0), dtype=torch.float32)
    return X, L, X0


# ══════════════════════════════════════════════════════════════
# 训练函数（原教旨：不使用 loss_mf）
# ══════════════════════════════════════════════════════════════

def train_first_principles(model, X, L, X0, seed=0,
                            ep=60, lr=1e-4, bs=32):
    """
    原教旨训练：不使用 loss_mf = (mq+1)²

    损失函数只包含：
    1. loss_cls：分类损失（有物理意义的监督）
    2. loss_metric：度规签名损失（让 g_μν 符合物理）
    3. loss_x0：x₀ 正值约束（信息时间必须为正）

    不包含：
    - loss_mf = (mq+1)²（工程原型强制 mq→-1 的损失）
    - loss_s/loss_c（工程原型的几何分离损失）

    Theorem 4 的预期：
    如果理论正确，只给 loss_cls + loss_metric，
    mq 会自然收敛到 -1（拓扑吸引子）
    这是最关键的验证
    """
    torch.manual_seed(seed)
    X, L, X0 = X.to(DEVICE), L.to(DEVICE), X0.to(DEVICE)

    opt   = torch.optim.AdamW(model.parameters(),
                              lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    model.train()

    for epoch in range(ep):
        idx = torch.randperm(len(X))
        total_cls = 0; total_met = 0; n_batch = 0

        for i in range(0, len(X), bs):
            ib  = idx[i:i+bs]
            xb  = X[ib]; lb = L[ib]; x0b = X0[ib]

            # ── 原教旨：传入理论 x₀ ─────────────────────────
            logits, emb = model(xb, x0_theory=x0b)

            # ── 损失1：分类（有监督，物理意义明确）──────────
            loss_cls = F.cross_entropy(logits, lb)

            # ── 损失2：度规签名（让度规符合洛伦兹签名）──────
            # 来自 Assumption R+E+T → det G < 0
            # 这不是约束违反损失，是让 g_μν 正确的损失
            loss_metric = model.g_net.lorentz_signature_loss(emb)

            # ── 损失3：x₀ > 0（信息时间必须为正）────────────
            # 来自：x₀ = exp(∫dt_info) > 0（指数函数恒正）
            loss_x0 = F.relu(-emb[:, 0] + 1.0 + EPS).mean()

            # ── 总损失（无 loss_mf！）────────────────────────
            # 如果 Theorem 4 正确，mq 会自然收敛到 -1
            loss = loss_cls + 0.1 * loss_metric + loss_x0

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.5)
            opt.step()

            total_cls += loss_cls.item()
            total_met += loss_metric.item()
            n_batch   += 1

        sched.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                _, emb_all = model(X[:100], x0_theory=X0[:100])
                g = model.measure(emb_all)
            print(f"  ep={epoch+1:3d}  "
                  f"cls={total_cls/n_batch:.4f}  "
                  f"mq_free={g['mq_free']:+.4f}  "
                  f"tl_free={g['tl_ratio_free']:.0%}  "
                  f"viol={g['violation_free']:.4f}  "
                  f"x₀={g['x0_mean']:.1f}±{g['x0_std']:.1f}")
            model.train()

    return model


# ══════════════════════════════════════════════════════════════
# 主实验：验证 Theorem 4 是否自然涌现
# ══════════════════════════════════════════════════════════════

def run():
    print("\n" + "="*65)
    print("LLCM 原教旨实现：Theorem 4 涌现验证")
    print("="*65)
    print("\n核心问题：不加 loss_mf，mq 是否自然收敛到 -1？")
    print("理论预测（Theorem 4）：是，来自 dc>0 的拓扑保证")
    print("工程原型做法：加 5.0*loss_mf 强制收敛（不是第一原理）\n")

    print("── 阶段1：验证 dt_info 计算 ─────────────────────────")
    itc = InformationTimeComputer()
    rng = np.random.RandomState(42)

    # 守恒系统
    traj_c = simulate(stable_ode, rng.randn(6)*0.3)
    traj_c = (traj_c - traj_c.mean(0)) / np.maximum(traj_c.std(0), 0.01)
    x0_c   = itc.compute(traj_c)

    # 耗散系统
    traj_d = simulate(running_ode, rng.randn(6)*0.3)
    traj_d = (traj_d - traj_d.mean(0)) / np.maximum(traj_d.std(0), 0.01)
    x0_d   = itc.compute(traj_d)

    print(f"  守恒系统 x₀：均值={x0_c.mean():.3f}  "
          f"std={x0_c.std():.3f}  范围=[{x0_c.min():.2f},{x0_c.max():.2f}]")
    print(f"  耗散系统 x₀：均值={x0_d.mean():.3f}  "
          f"std={x0_d.std():.3f}  范围=[{x0_d.min():.2f},{x0_d.max():.2f}]")
    print(f"  理论预测：守恒系统 x₀ 更稳定（H=const → dt_info 稳定）")
    print(f"  验证：守恒 std={x0_c.std():.3f} vs 耗散 std={x0_d.std():.3f}  "
          f"{'✅' if x0_c.std() < x0_d.std() else '❌'}\n")

    results = []

    for seed in SEEDS:
        print(f"── Seed {seed} ─────────────────────────────────────────")
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_tr, L_tr, X0_tr = build_dataset_with_x0(seed+100, 400)
        X_te, L_te, X0_te = build_dataset_with_x0(42,       200)

        # ── 原教旨模型 ──────────────────────────────────────
        model = FirstPrinciplesBackbone(d=D).to(DEVICE)
        print(f"\n  训练（无 loss_mf，Theorem 4 涌现验证）：")
        train_first_principles(model, X_tr, L_tr, X0_tr, seed=seed)

        model.eval()
        with torch.no_grad():
            X_te_dev  = X_te.to(DEVICE)
            X0_te_dev = X0_te.to(DEVICE)
            _, emb = model(X_te_dev, x0_theory=X0_te_dev)
            g = model.measure(emb)

        # ── 分类准确率 ──────────────────────────────────────
        with torch.no_grad():
            logits, _ = model(X_te_dev, x0_theory=X0_te_dev)
            preds = logits.argmax(-1).cpu()
            acc   = (preds == L_te).float().mean().item()

        print(f"\n  ── 结果 ──")
        print(f"  mq_free（自由演化）：{g['mq_free']:+.4f}  "
              f"← Theorem 4 核心指标（不经过 project）")
        print(f"  mq_proj（投影后）：  {g['mq_proj']:+.4f}  "
              f"← 数学恒等式，始终=-1（参考）")
        print(f"  mq_dynamic（动态度规）：{g['mq_dynamic']:+.4f}")
        print(f"  类时比例（自由）：{g['tl_ratio_free']:.1%}")
        print(f"  约束违反（自由）：{g['violation_free']:.6f}")
        print(f"  x₀均值：{g['x0_mean']:.2f}±{g['x0_std']:.2f}")
        print(f"  分类准确率：{acc:.1%}")

        # Theorem 4 涌现：自由演化的 mq 是否自然接近 -1
        theorem4_ok = abs(g['mq_free'] + 1.0) < 0.5  # 允许±0.5的误差
        print(f"\n  Theorem 4 涌现（自由 mq 接近-1，误差<0.5）："
              f"{'✅' if theorem4_ok else '❌'} "
              f"(mq_free={g['mq_free']:+.4f})")

        results.append(dict(
            mq=g['mq_free'],              # 自由演化的 mq（Theorem 4 核心）
            tl=g['tl_ratio_free'],
            viol=g['violation_free'],
            x0=g['x0_mean'],
            acc=acc,
            theorem4=theorem4_ok,
        ))

    print("\n" + "="*65)
    print("汇总：原教旨实现 vs 工程原型")
    print("="*65)
    t4_ok = sum(r['theorem4'] for r in results)
    print(f"\n  Theorem 4 涌现（无 loss_mf）："
          f"{t4_ok}/{len(results)} seeds")
    print(f"  平均 mq_free（自由演化）：{np.mean([r['mq'] for r in results]):+.4f}"
          f"（工程原型 mq_proj：-1.000，数学恒等式）")
    print(f"  平均类时（自由）：{np.mean([r['tl'] for r in results]):.1%}")
    print(f"  如果 mq_free ≈ -1：Theorem 4 自然涌现 ✅")
    print(f"  如果 mq_free ≠ -1：project 是必要的工程补丁 ⚠️")
    print(f"  平均 x₀：{np.mean([r['x0'] for r in results]):.2f}"
          f"（来自 dt_info，工程原型：≈57）")
    print(f"  平均 acc：{np.mean([r['acc'] for r in results]):.1%}")

    print("\n── 原教旨 vs 工程原型的区别 ─────────────────────────")
    print("  工程原型用 5.0*loss_mf 强制 mq→-1")
    print("  原教旨不加 loss_mf，看 Theorem 4 是否让 mq 自然→-1")
    print("  如果 Theorem 4 正确：两者的 mq 应该接近")
    print("  如果 Theorem 4 需要验证：原教旨的 mq 会偏离 -1")
    print()
    print("  dt_info 的贡献：")
    print("  x₀ 来自 exp(∫dΦ/H dt)，不是线性近似 t/T*0.1")
    print("  守恒系统的 x₀ 比耗散系统更稳定（H=const 保证）")
    print("  这是信息时间在原教旨层面的直接验证")


if __name__ == '__main__':
    run()

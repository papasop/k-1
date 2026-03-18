# ============================================================
# optimizer.py
# 洛伦兹Transformer第二组件：测地线Adam优化器
#
# GeodesicAdam：在洛伦兹流形上沿类时测地线更新参数
#
# 核心思想：
#   标准Adam：所有梯度方向平等（欧氏空间假设）
#   GeodesicAdam：梯度分解为类时（G_ii<0）和类空（G_ii>0）分量
#                 类时方向步长更大（信息几何上更有效的方向）
#                 类空方向步长更小（保护已有知识）
#
# 数学：
#   g_t = P_t ⊙ g          (类时梯度，mask=1的位置)
#   g_s = (I-P_t) ⊙ g      (类空梯度，mask=0的位置)
#   g_geo = scale_t×g_t + scale_s×g_s
#   θ_{t+1} = Adam_update(g_geo)
#
# 实验结果（geodesic_adam.py）：
#   最优配置：scale_t=2.0, scale_s=0.5（ratio=4）
#   r规律在GeodesicAdam下同样成立（r=-0.90到-0.997）
#   seed 1（弱baseline）：geo_200_050 delta=+0.025
#
# 用法：
#   from optimizer import GeodesicAdam
#   optimizer = GeodesicAdam(model.parameters(),
#                             lr=3e-4, scale_t=2.0, scale_s=0.5)
#   # 每步更新P_t后：
#   optimizer.update_masks(model.probe.get_param_mask_pairs(device))
#   optimizer.step()
# ============================================================

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable


class GeodesicAdam(torch.optim.Adam):
    """
    测地线Adam优化器

    继承标准Adam，在调用父类step之前对梯度做类时/类空分解：

      g_t    = mask ⊙ g           (类时分量，沿负Hessian方向)
      g_s    = (1-mask) ⊙ g       (类空分量)
      g_geo  = scale_t×g_t + scale_s×g_s
      → Adam(g_geo)

    其中mask来自TimeLikeProbe（G_ii<0的位置为1）。

    参数：
      params:    模型参数（同torch.optim.Adam）
      lr:        基础学习率
      betas:     Adam的β1, β2
      eps:       Adam的ε
      weight_decay: 权重衰减
      scale_t:   类时方向学习率乘数（推荐2.0，实验验证）
      scale_s:   类空方向学习率乘数（推荐0.5，实验验证）
      adaptive_scale: 是否根据类时比例自适应调整缩放
                      True时：比例越高，类时放大越保守
                      False时：固定使用scale_t和scale_s

    自适应缩放（adaptive_scale=True）：
      如果类时比例frac很高（如0.8），说明大部分方向都是类时，
      放大所有类时方向会导致有效学习率显著上升。
      自适应模式下：effective_scale_t = 1 + (scale_t-1) × (1-frac)
      确保类时比例高时自动降低放大幅度。

    注意：
      只对通过update_masks()注册了mask的参数做分解。
      其他参数（位置编码、LayerNorm等）使用标准Adam。
      这保证了只有W_Q受到洛伦兹几何约束。
    """

    def __init__(self,
                 params,
                 lr: float = 3e-4,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 scale_t: float = 2.0,
                 scale_s: float = 0.5,
                 adaptive_scale: bool = False):

        super().__init__(params, lr=lr, betas=betas,
                         eps=eps, weight_decay=weight_decay)

        self.scale_t        = scale_t
        self.scale_s        = scale_s
        self.adaptive_scale = adaptive_scale

        # param_id → timelike_mask (与param同形状的0/1 float张量)
        self._masks: Dict[int, torch.Tensor] = {}

        # 诊断统计
        self.n_masked_params   = 0     # 有mask的参数数量
        self.last_timelike_frac = 0.0  # 最近一次的类时比例均值

    def update_masks(self,
                     param_mask_pairs: List[Tuple[nn.Parameter,
                                                   torch.Tensor]]):
        """
        注入类时掩码

        param_mask_pairs: list of (param, mask)
          param: nn.Parameter（通常是W_Q）
          mask:  与param同形状的bool或float张量
                 True/1.0 = 类时，False/0.0 = 类空

        调用时机：在每次probe.step()之后调用，确保mask是最新的。

        train.py里的调用方式：
          probe.step(x, global_step)
          optimizer.update_masks(probe.get_param_mask_pairs(device))
        """
        self._masks.clear()
        fracs = []
        for param, mask in param_mask_pairs:
            m = mask.float().to(param.device)
            self._masks[id(param)] = m
            fracs.append(m.mean().item())

        self.n_masked_params   = len(param_mask_pairs)
        self.last_timelike_frac = sum(fracs) / len(fracs) if fracs else 0.0

    def _get_effective_scales(self, frac: float) -> Tuple[float, float]:
        """
        计算有效的类时/类空缩放系数

        固定模式：直接返回scale_t和scale_s
        自适应模式：根据类时比例调整，防止有效学习率过大
        """
        if not self.adaptive_scale:
            return self.scale_t, self.scale_s

        # 自适应：类时比例越高，放大越保守
        # effective_scale_t ∈ [1.0, scale_t]
        # frac=0 → scale_t（最大放大）
        # frac=1 → 1.0（不放大，所有方向都是类时时无法区分）
        eff_t = 1.0 + (self.scale_t - 1.0) * (1.0 - frac)
        eff_s = self.scale_s
        return eff_t, eff_s

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        执行一步参数更新

        流程：
          1. 对有mask的参数，把grad分解为类时/类空分量并重新缩放
          2. 调用标准Adam的step（使用修改后的grad）
          3. 恢复grad（避免影响后续调用）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 存储原始grad（用于恢复）
        original_grads: Dict[int, Optional[torch.Tensor]] = {}

        # ── 梯度分解（仅对有mask的参数）──
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                mask = self._masks.get(id(p))
                if mask is None:
                    continue   # 无mask：使用标准Adam，不修改

                # 保存原始grad
                original_grads[id(p)] = p.grad.data.clone()

                g    = p.grad.data                          # (d_out, d_in)
                mask = mask.to(g.device)

                # 类时/类空分解（逐元素乘，对角P_t作用于每行）
                g_t  = mask * g                             # 类时分量
                g_s  = (1.0 - mask) * g                    # 类空分量

                # 有效缩放系数
                eff_t, eff_s = self._get_effective_scales(
                    self.last_timelike_frac)

                # 合并：类时放大，类空缩小
                g_geo = eff_t * g_t + eff_s * g_s
                p.grad.data = g_geo

        # ── 标准Adam更新（使用修改后的grad）──
        super().step(closure=None)

        # ── 恢复原始grad（保持调用者的grad状态）──
        for group in self.param_groups:
            for p in group['params']:
                if id(p) in original_grads:
                    p.grad.data = original_grads[id(p)]

        return loss

    def state_dict_lorentz(self) -> dict:
        """
        保存包含洛伦兹状态的完整state_dict

        标准optimizer.state_dict()不包含mask信息。
        这个方法额外保存mask，用于从checkpoint完整恢复。
        """
        base = super().state_dict()
        base['lorentz'] = {
            'scale_t':         self.scale_t,
            'scale_s':         self.scale_s,
            'adaptive_scale':  self.adaptive_scale,
            'n_masked_params': self.n_masked_params,
            # mask不保存（会从probe重新计算）
        }
        return base

    @classmethod
    def load_lorentz(cls, state_dict: dict,
                     params, **kwargs) -> 'GeodesicAdam':
        """
        从state_dict恢复优化器（包含洛伦兹配置）
        """
        lorentz_cfg = state_dict.pop('lorentz', {})
        kwargs.setdefault('scale_t', lorentz_cfg.get('scale_t', 2.0))
        kwargs.setdefault('scale_s', lorentz_cfg.get('scale_s', 0.5))
        kwargs.setdefault('adaptive_scale',
                          lorentz_cfg.get('adaptive_scale', False))
        opt = cls(params, **kwargs)
        opt.load_state_dict(state_dict)
        return opt

    def report(self) -> str:
        """打印优化器状态摘要"""
        lines = [
            f'GeodesicAdam:',
            f'  scale_t={self.scale_t}  scale_s={self.scale_s}  '
            f'ratio={self.scale_t/self.scale_s:.1f}',
            f'  adaptive_scale={self.adaptive_scale}',
            f'  masked_params={self.n_masked_params}',
            f'  timelike_frac={self.last_timelike_frac:.3f}',
        ]
        if self.adaptive_scale and self.last_timelike_frac > 0:
            eff_t, eff_s = self._get_effective_scales(
                self.last_timelike_frac)
            lines.append(
                f'  effective: scale_t={eff_t:.3f}  scale_s={eff_s:.3f}')
        return '\n'.join(lines)


# ─────────────────────────────────────────────
# 学习率调度器（配合GeodesicAdam使用）
# ─────────────────────────────────────────────
class LorentzCosineScheduler:
    """
    配合洛伦兹四阶段训练的余弦学习率调度器

    四阶段（对应train.py的训练协议）：
      Phase 0 [0, warmup):          线性warmup，α=0（标准注意力）
      Phase 1 [warmup, 10%):        余弦衰减，仅正则化，α=0
      Phase 2 [10%, 90%):           余弦衰减，全组件，α=lorentz_alpha
      Phase 3 [90%, 100%]:          余弦衰减+α退火到0，逐步关闭洛伦兹

    用法：
      scheduler = LorentzCosineScheduler(optimizer, total_steps,
                                          warmup_steps, base_lr)
      # 每步调用：
      scheduler.step(global_step, model)
    """

    def __init__(self,
                 optimizer: GeodesicAdam,
                 total_steps: int,
                 warmup_steps: int,
                 base_lr: float,
                 min_lr: float = 1e-5,
                 lorentz_alpha: float = 0.25):

        self.optimizer     = optimizer
        self.total_steps   = total_steps
        self.warmup_steps  = warmup_steps
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self.lorentz_alpha = lorentz_alpha

        # 阶段边界
        self.phase1_end = int(total_steps * 0.10)
        self.phase2_end = int(total_steps * 0.90)

    def get_lr(self, step: int) -> float:
        """计算当前步的学习率"""
        if step < self.warmup_steps:
            # 线性warmup
            return self.base_lr * step / max(1, self.warmup_steps)

        # warmup结束后：余弦衰减
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def get_alpha(self, step: int) -> float:
        """计算当前步的洛伦兹强度α"""
        if step < self.phase1_end:
            return 0.0   # Phase 0/1：纯标准注意力

        if step < self.phase2_end:
            return self.lorentz_alpha   # Phase 2：完整洛伦兹

        # Phase 3：α余弦退火到0
        progress = (step - self.phase2_end) / max(
            1, self.total_steps - self.phase2_end)
        return self.lorentz_alpha * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, global_step: int, model=None):
        """
        更新学习率和α

        global_step: 当前全局训练步
        model:       LorentzTransformer实例（用于更新α）
        """
        lr    = self.get_lr(global_step)
        alpha = self.get_alpha(global_step)

        # 更新学习率
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

        # 更新洛伦兹强度
        if model is not None:
            model.set_lorentz_alpha(alpha)

        return lr, alpha

    def report(self, step: int) -> str:
        lr    = self.get_lr(step)
        alpha = self.get_alpha(step)
        phase = (0 if step < self.warmup_steps else
                 1 if step < self.phase1_end else
                 2 if step < self.phase2_end else 3)
        return (f'step={step}  phase={phase}  '
                f'lr={lr:.2e}  α={alpha:.4f}')


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    # Colab: model.py가 /content/ 에 있는 경우를 포함한 경로 탐색
    for _p in ['.', '/content', os.path.dirname(os.path.abspath(__file__))
               if hasattr(sys.modules[__name__], '__file__') else '.']:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    # model.py 로드 시도 — 없으면 최소 stub으로 대체
    try:
        from model import LorentzTransformer, LorentzConfig
    except ModuleNotFoundError:
        # Colab: /content/ 에서 재시도
        sys.path.insert(0, '/content')
        try:
            from model import LorentzTransformer, LorentzConfig
        except ModuleNotFoundError:
            print("경고: model.py 없음 — 최소 stub으로 검증 실행")
            # ── 최소 stub (model.py 없이 optimizer 자체만 검증) ──
            import dataclasses
            @dataclasses.dataclass
            class LorentzConfig:
                vocab_size:int=1000; d_model:int=128; n_heads:int=4
                n_layers:int=4; max_seq_len:int=64; dropout:float=0.1
                lorentz_alpha:float=0.25; hess_update_freq:int=10
                hess_warmup_steps:int=5; hutchinson_k:int=5
                ema_decay:float=0.7; lambda_spacelike:float=1e-4
                lambda_timelike:float=0.0; d_ff:int=0
                def __post_init__(self):
                    if self.d_ff==0: self.d_ff=4*self.d_model
                @property
                def head_dim(self): return self.d_model//self.n_heads
            class _FakeProbe:
                def __init__(self,m,c):
                    self.timelike_fracs=[0.0]*c.n_layers
                def step(self,x,s): pass
                def get_param_mask_pairs(self,dev): return []
            class _FakeLinear(torch.nn.Linear):
                def __init__(self,d):
                    super().__init__(d,d,bias=False)
                    self.timelike_mask=torch.zeros(d,dtype=torch.bool)
                def set_timelike_mask(self,m): self.timelike_mask=m
            class _FakeAttn(torch.nn.Module):
                def __init__(self,d):
                    super().__init__()
                    self.q_proj=_FakeLinear(d)
                    self.register_buffer('timelike_mask',
                                         torch.zeros(d,dtype=torch.bool))
                    self._has_mask=False
                    self.alpha=0.25
            class _FakeBlock(torch.nn.Module):
                def __init__(self,cfg):
                    super().__init__()
                    self.attn=_FakeAttn(cfg.d_model)
                    self.norm1=torch.nn.LayerNorm(cfg.d_model)
            class LorentzTransformer(torch.nn.Module):
                def __init__(self,cfg):
                    super().__init__()
                    self.config=cfg
                    self.embed=torch.nn.Embedding(cfg.vocab_size,cfg.d_model)
                    self.pos_emb=torch.nn.Embedding(cfg.max_seq_len,cfg.d_model)
                    self.blocks=torch.nn.ModuleList([
                        _FakeBlock(cfg) for _ in range(cfg.n_layers)])
                    self.norm_f=torch.nn.LayerNorm(cfg.d_model)
                    self.lm_head=torch.nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
                    self.probe=_FakeProbe(self,cfg)
                def get_num_params(self,**kw):
                    return sum(p.numel() for p in self.parameters())
                def set_lorentz_alpha(self,a): pass
                def regularization_loss(self):
                    return torch.tensor(0.0,requires_grad=True)
                def forward(self,x,mask=None):
                    B,L=x.shape
                    h=self.embed(x)+self.pos_emb(
                        torch.arange(L,device=x.device).unsqueeze(0))
                    return self.lm_head(self.norm_f(h))

    print('='*55)
    print('GeodesicAdam 快速验证')
    print('='*55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 小模型
    cfg = LorentzConfig(
        vocab_size=1000, d_model=128, n_heads=4,
        n_layers=4, max_seq_len=64,
        lorentz_alpha=0.25,
        hess_warmup_steps=5, hess_update_freq=10,
        hutchinson_k=5,
    )
    model = LorentzTransformer(cfg).to(device)

    # GeodesicAdam
    optimizer = GeodesicAdam(
        model.parameters(), lr=3e-4,
        scale_t=2.0, scale_s=0.5)
    print(f'\n{optimizer.report()}')

    # 调度器
    total_steps = 1000
    scheduler   = LorentzCosineScheduler(
        optimizer, total_steps=total_steps,
        warmup_steps=50, base_lr=3e-4,
        lorentz_alpha=0.25)

    # 阶段检查
    print('\n调度器阶段检查:')
    for step in [0, 25, 50, 100, 500, 900, 999]:
        print(f'  {scheduler.report(step)}')

    # 训练步测试（10步）
    x    = torch.randint(0, 1000, (4, 32), device=device)
    mask = torch.ones(4, 32, dtype=torch.bool, device=device)

    print('\n训练步测试（10步）:')
    for step in range(40):
        # P_t更新
        model.probe.step(x, step)

        # 同步mask到optimizer
        pairs = model.probe.get_param_mask_pairs(device)
        optimizer.update_masks(pairs)

        # 调度器更新
        lr, alpha = scheduler.step(step, model)

        # 前向 + 反向
        logits = model(x, mask)
        loss   = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 1000),
            x[:, 1:].reshape(-1))
        loss  += model.regularization_loss()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 9:
            frac = model.probe.timelike_fracs
            print(f'  step={step+1:3d}  loss={loss.item():.4f}  '
                  f'lr={lr:.2e}  α={alpha:.3f}  '
                  f'timelike={[round(f,2) for f in frac]}  '
                  f'masked={optimizer.n_masked_params}')

    print(f'\n{optimizer.report()}')

    # 梯度分해 검증：mask있는 파라미터에 대해 grad가 수정되는지 확인
    print('\n梯度分解验证:')
    x2 = torch.randint(0, 1000, (2, 16), device=device)
    logits2 = model(x2)
    loss2   = logits2.mean()
    loss2.backward()

    for li, block in enumerate(model.blocks):
        W_Q  = block.attn.q_proj
        if W_Q.weight.grad is not None:
            g    = W_Q.weight.grad
            mask_li = block.attn.timelike_mask.float()
            g_t  = (mask_li * g).abs().mean().item()
            g_s  = ((1-mask_li) * g).abs().mean().item()
            print(f'  layer {li}: |g_t|={g_t:.6f}  |g_s|={g_s:.6f}  '
                  f'frac={mask_li.mean().item():.3f}')

    print('\n所有测试通过 ✓')
    print('下一步: train.py (完整训练循环)')

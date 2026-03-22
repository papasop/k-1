"""
llcm/core.py
============
LLCM（Lorentz Light-Cone Model）核心模块

理论基础：
  Realizability.pdf（Li 2026）— Theorem 5：洛伦兹签名从代价函数唯一涌现
  K=1 Chronogeometrodynamics（Li 2026）— Theorem 4：det G < 0 ⟺ dc > 0

所有实验从这里 import，避免重复定义：
  from core import (
      MinkowskiLN, Attn, LLCMBackbone,
      stable_ode, running_ode, simulate, build_dataset,
      momentum_change, encode, pretrain
  )

模块对应关系：
  模块1（物理预训练）：LLCMBackbone + pretrain()
  模块2（语言编码器）：encode()，基于 sentence-transformers
  模块3（方向A）：    LLCMBackbone.forward_A_gen() + lang_gen 头
  模块4（lang_aligner）：LLCMBackbone.lang_aligner
  模块5（phys_decoder）：LLCMBackbone.phys_decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, TensorDataset
import warnings; warnings.filterwarnings('ignore')

# ── 默认超参数（实验文件可覆盖） ───────────────────────────────
EMBED_DIM  = 128
N_HEADS    = 4
N_LAYERS   = 3
TIME_RATIO = 0.25
T_IN       = 20
T_OUT      = 20
STATE_DIM  = 6
LANG_DIM   = 384
N_LABELS   = 2
N_PER      = 50
EP_PRE     = 80
LR_PRE     = 3e-4
BS         = 16
T_DIM      = max(1, int(N_HEADS*TIME_RATIO)) * (EMBED_DIM//N_HEADS)

LABELS = {0:'momentum_stable', 1:'momentum_changing'}
DESCRIPTIONS = {
    0: ["平稳匀速运动，动量保持守恒",
        "smooth constant velocity motion with conserved momentum",
        "steady movement without any acceleration"],
    1: ["动量持续变化，存在外力作用",
        "changing momentum with continuous force application",
        "accelerating or decelerating movement"],
}
STABLE_INSTRUCTIONS = [
    "平稳匀速运动，动量保持守恒",
    "机器人以恒定速度移动，没有加速或减速",
    "smooth constant velocity motion with conserved momentum",
    "steady movement without acceleration changes",
]

# ── 设备 ──────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 语言编码器（模块2） ────────────────────────────────────────
_lang_enc = None

def get_lang_enc():
    global _lang_enc
    if _lang_enc is None:
        from sentence_transformers import SentenceTransformer
        _lang_enc = SentenceTransformer('all-MiniLM-L6-v2')
    return _lang_enc

def encode(texts, dev=None):
    """语言嵌入，懒加载"""
    if dev is None: dev=device
    return get_lang_enc().encode(
        texts, convert_to_tensor=True,
        show_progress_bar=False).to(dev)

# ── 物理数据 ───────────────────────────────────────────────────
def stable_ode(t, y):
    """
    稳定匀速运动 ODE（动量守恒）
    Assumption R 满足：沿速度方向的位移代价趋向零
    """
    x,yp,z,vx,vy,vz=y
    return [vx, 0, vz, -0.001*vx, 0, -0.001*vz]

def running_ode(t, y):
    """
    奔跑运动 ODE（动量不守恒）
    外力驱动，速度持续变化
    """
    x,yp,z,vx,vy,vz=y; g=9.81; m=70.0
    phase=(3.0*t)%1.0; L=0.95+0.12*abs(np.sin(3.0*np.pi*t))
    pen=max(0,L-yp)
    Fv=(2000*pen-30*vy) if (phase<0.4 and yp<L) else 0
    return [vx,vy,vz,
            m*0.002*2000*(3.5-vx)/m,
            (Fv-m*g)/m,
            (-80*z-15*vz)/m]

def simulate(ic, ode_fn, seed=0, total_frames=80):
    """
    ODE 仿真，返回 (total_frames, STATE_DIM) 轨迹
    失败返回 None
    """
    rng=np.random.RandomState(seed)
    sol=solve_ivp(ode_fn,(0,total_frames*0.033),ic,
                  t_eval=np.linspace(0,total_frames*0.033,total_frames),
                  method='RK45',rtol=1e-6,atol=1e-8)
    if not sol.success: return None
    traj=sol.y.T.astype(np.float32)
    if np.any(np.isnan(traj)) or np.any(np.abs(traj)>200): return None
    pos=traj[:,:3]+rng.randn(total_frames,3)*0.002
    vel=np.zeros_like(pos)
    vel[1:-1]=(pos[2:]-pos[:-2])/(2*0.033)
    vel[0]=(pos[1]-pos[0])/0.033
    vel[-1]=(pos[-1]-pos[-2])/0.033
    return np.concatenate([pos,vel],axis=1)

def build_dataset(seed=0, n_per=None, t_in=None, verbose=True):
    """
    构建物理数据集
    X: (N, T_IN, STATE_DIM) 归一化轨迹
    L: (N,) 标签 0=stable 1=changing
    """
    if n_per is None: n_per=N_PER
    if t_in  is None: t_in=T_IN
    rng=np.random.RandomState(seed)
    X_list,L_list=[],[]
    odes=[(stable_ode, [0,1.0,0,1.0,0,0],[0.3,0,0,0.1,0,0]),
          (running_ode,[0,1.0,0,2.0,0,0],[0.5,0,0,0.3,0,0])]
    for lbl,(ode_fn,ic_base,ic_noise) in enumerate(odes):
        count=0; i=0
        while count<n_per and i<5000:
            ic=[b+rng.randn()*n for b,n in zip(ic_base,ic_noise)]
            ic[1]=1.0; ic[2]=0.0; ic[4]=0.0; ic[5]=0.0
            traj=simulate(ic,ode_fn,seed=seed*1000+i); i+=1
            if traj is None or len(traj)<t_in+5: continue
            seg=traj[:t_in]
            mu=seg.mean(0); sig=seg.std(0)+1e-8
            X_list.append(((seg-mu)/sig).astype(np.float32))
            L_list.append(lbl); count+=1
    X=torch.from_numpy(np.stack(X_list))
    L=torch.tensor(L_list,dtype=torch.long)
    if verbose:
        print(f'  数据集: {len(X)} 样本  '
              f'stable={(L==0).sum()}  changing={(L==1).sum()}')
    return X,L

def momentum_change(traj):
    """
    动量变化率（L2范数均值，越低越守恒）
    traj: (T, STATE_DIM) 时域轨迹
    """
    vel=traj[:,3:]; dp=np.diff(vel,axis=0)
    return float(np.linalg.norm(dp,axis=-1).mean())

def real_physics_baseline(n=50, seed=42):
    """
    真实物理基准：stable_ode 生成轨迹的动量变化率
    用 t[T_IN:T_IN+T_OUT] 预测段，和模型输出量纲一致
    """
    rng=np.random.RandomState(seed); moms=[]
    for i in range(n):
        ic=[0,1.0,0,1.0+rng.randn()*0.1,0,0]
        t=simulate(ic,stable_ode,seed=i)
        if t is not None and len(t)>=T_IN+T_OUT:
            moms.append(momentum_change(t[T_IN:T_IN+T_OUT]))
    return float(np.mean(moms))

# ── 模型组件 ──────────────────────────────────────────────────
class MinkowskiLN(nn.Module):
    """
    闵可夫斯基 LayerNorm（光锥归一化）
    ⟨x,x⟩_η = ‖x_s‖² − ‖x_t‖²（保留符号）
    t_dim：类时维度数量（对应 TIME_RATIO）
    """
    def __init__(self, dim, t_dim, eps=1e-5):
        super().__init__()
        self.td=t_dim; self.eps=eps
        self.pre=nn.LayerNorm(dim)
        self.g=nn.Parameter(torch.ones(dim))
        self.b=nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x=self.pre(x)
        t=x[...,:self.td]; s=x[...,self.td:]
        mq=(s**2).sum(-1,keepdim=True)-(t**2).sum(-1,keepdim=True)
        eps=torch.clamp(0.01*mq.abs().mean(), min=self.eps)
        return self.g*(x/(torch.sqrt(mq.abs()+eps)+eps))+self.b

class Attn(nn.Module):
    """
    光锥注意力（F3公式）
    F3: score = cat(−σ·Q_t Kᵀ_t, Q_s Kᵀ_s) / √d
    负号让类时方向互相排斥 → 信息沿光锥边界传播
    σ = sigmoid(w_sigma)，训练中自适应收敛
    """
    def __init__(self, dim, n_heads, time_ratio, mode='f3'):
        super().__init__()
        self.nh=n_heads; self.hd=dim//n_heads
        self.mode=mode; self.scale=self.hd**-0.5
        self.nt=max(1,int(n_heads*time_ratio))
        self.ns=n_heads-self.nt
        if mode=='euclidean':
            self.q=nn.Linear(dim,dim,bias=False)
            self.k=nn.Linear(dim,dim,bias=False)
        else:
            self.qt=nn.Linear(dim,self.nt*self.hd,bias=False)
            self.kt=nn.Linear(dim,self.nt*self.hd,bias=False)
            self.qs=nn.Linear(dim,self.ns*self.hd,bias=False)
            self.ks=nn.Linear(dim,self.ns*self.hd,bias=False)
            if mode=='f3':
                self.w_sigma=nn.Parameter(torch.zeros(1))
        self.v=nn.Linear(dim,dim,bias=False)
        self.out=nn.Linear(dim,dim)

    def forward(self, x):
        B,T,D=x.shape; hd=self.hd
        if self.mode=='euclidean':
            Q=self.q(x).view(B,T,self.nh,hd)
            K=self.k(x).view(B,T,self.nh,hd)
            score=torch.einsum('bthd,bshd->bths',Q,K)*self.scale
        else:
            Qt=self.qt(x).view(B,T,self.nt,hd)
            Kt=self.kt(x).view(B,T,self.nt,hd)
            Qs=self.qs(x).view(B,T,self.ns,hd)
            Ks=self.ks(x).view(B,T,self.ns,hd)
            st=torch.einsum('bthd,bshd->bths',Qt,Kt)
            ss=torch.einsum('bthd,bshd->bths',Qs,Ks)
            if self.mode=='f3':
                score=torch.cat(
                    [-torch.sigmoid(self.w_sigma)*st, ss],
                    dim=2)*self.scale
            else:
                score=torch.cat([-st,ss],dim=2)*self.scale
        attn=F.softmax(score,dim=-1)
        V=self.v(x).view(B,T,self.nh,hd)
        return self.out(
            torch.einsum('bths,bshd->bthd',attn,V).reshape(B,T,D))

    @property
    def sigma(self):
        if self.mode=='f3':
            return float(torch.sigmoid(self.w_sigma).item())
        return None

class LLCMBackbone(nn.Module):
    """
    LLCM 完整模型（五模块）
    ─────────────────────────────────────────────
    模块1: embed + blocks + norm（物理预训练 backbone）
    模块3: lang_gen（方向A：物理→语言）
           cls（分类头）
    模块4: lang_aligner（方向B第一步：语言→洛伦兹）
    模块5: phys_decoder（方向B第二步：洛伦兹→轨迹）

    参数：
      mode: 'f3'（默认）| 'euclidean' | 'f1'
      embed_dim, n_heads, n_layers, time_ratio：架构参数
      state_dim, lang_dim, t_in, t_out：数据参数
    """
    def __init__(self,
                 mode='f3',
                 embed_dim=EMBED_DIM,
                 n_heads=N_HEADS,
                 n_layers=N_LAYERS,
                 time_ratio=TIME_RATIO,
                 state_dim=STATE_DIM,
                 lang_dim=LANG_DIM,
                 n_labels=N_LABELS,
                 t_in=T_IN,
                 t_out=T_OUT):
        super().__init__()
        dim=embed_dim
        td=max(1,int(n_heads*time_ratio))*(dim//n_heads)
        use_mln=(mode!='euclidean')
        self.mode=mode; self.t_dim=td; self.t_out=t_out
        self.state_dim=state_dim; self.lang_dim=lang_dim

        # 模块1: backbone
        self.embed=nn.Linear(state_dim, dim)
        self.pos=nn.Embedding(t_in+t_out, dim)
        NC=lambda:(MinkowskiLN(dim,td) if use_mln else nn.LayerNorm(dim))
        self.blocks=nn.ModuleList([nn.ModuleDict({
            'attn': Attn(dim, n_heads, time_ratio, mode),
            'n1':   NC(), 'n2': NC(),
            'ff':   nn.Sequential(
                        nn.Linear(dim,dim*4), nn.GELU(),
                        nn.Linear(dim*4,dim)),
        }) for _ in range(n_layers)])
        self.norm=NC()
        self.traj_head=nn.Linear(dim, state_dim*t_out)

        # 模块3: 方向A
        self.lang_gen=nn.Sequential(
            nn.Linear(dim,dim*2), nn.GELU(),
            nn.LayerNorm(dim*2), nn.Linear(dim*2,lang_dim))
        self.cls=nn.Sequential(
            nn.Linear(dim,dim//2), nn.GELU(),
            nn.LayerNorm(dim//2), nn.Linear(dim//2,n_labels))

        # 模块4: 方向B 第一步
        self.lang_aligner=nn.Sequential(
            nn.Linear(lang_dim,dim*2), nn.GELU(),
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2,dim), nn.GELU(), nn.LayerNorm(dim))

        # 模块5: 方向B 第二步
        self.phys_decoder=nn.Sequential(
            nn.Linear(dim,dim*2), nn.GELU(),
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, state_dim*t_out))

    # ── 前向方法 ─────────────────────────────────────────────
    def embed_seq(self, x):
        """物理轨迹 → 洛伦兹嵌入（核心表示）"""
        B,T,_=x.shape
        h=self.embed(x)+self.pos(torch.arange(T,device=x.device))
        for blk in self.blocks:
            h=h+blk['attn'](blk['n1'](h))
            h=h+blk['ff'](blk['n2'](h))
        return self.norm(h)[:,-1,:]

    def forward_pretrain(self, x):
        """模块1：轨迹预测（多步 MSE 预训练）"""
        return self.traj_head(self.embed_seq(x)).view(
            x.shape[0], self.t_out, self.state_dim)

    def forward_A_gen(self, x):
        """模块3：物理嵌入 → 语言空间（方向A）"""
        return self.lang_gen(self.embed_seq(x))

    def forward_A_cls(self, x):
        """模块3：物理嵌入 → 分类标签"""
        return self.cls(self.embed_seq(x))

    def forward_B(self, lang_emb):
        """模块4+5：语言 → 洛伦兹 → 时域轨迹（方向B完整路径）"""
        return self.phys_decoder(
            self.lang_aligner(lang_emb)).view(
            -1, self.t_out, self.state_dim)

    # ── 层2测量 ───────────────────────────────────────────────
    def measure_lorentz(self, x_emb):
        """
        测量物理嵌入的洛伦兹几何量
        返回：tl_ratio（类时比例），mq_mean（平均 mq 值）
        mq = ‖s‖² − ‖t‖²，mq<0 = 类时区域
        对应 K=1 Chronogeometrodynamics 的 K = x⊤Gx
        """
        t_p=x_emb[:,:self.t_dim]; s_p=x_emb[:,self.t_dim:]
        mq=(s_p**2).sum(-1)-(t_p**2).sum(-1)
        return {
            'tl_ratio': (mq<0).float().mean().item(),
            'mq_mean':  mq.mean().item(),
            'mq_std':   mq.std().item(),
        }

    def compute_dc(self, x_emb):
        """
        计算局部稳定边界 dc（Theorem 2）
        dc = α·√(−1/det G)，det G < 0 → dc > 0（洛伦兹）
        使用对角近似 G = diag(−‖t‖², ‖s‖²)
        """
        t_norm=(x_emb[:,:self.t_dim]**2).sum(-1).mean().item()
        s_norm=(x_emb[:,self.t_dim:]**2).sum(-1).mean().item()
        det_G=(-t_norm)*s_norm  # 对角近似
        if det_G<0:
            return float(np.sqrt(-1.0/det_G))
        return 0.0  # 欧氏退化（Theorem 3）

    # ── 权重管理 ──────────────────────────────────────────────
    def freeze_backbone(self):
        """冻结 backbone，只训练头部模块"""
        for name,p in self.named_parameters():
            if not any(k in name for k in
                       ['lang_gen','cls','lang_aligner','phys_decoder']):
                p.requires_grad_(False)

    def freeze_all_except_B(self):
        """只训练方向B（模块4+5）"""
        for name,p in self.named_parameters():
            if not any(k in name for k in
                       ['lang_aligner','phys_decoder']):
                p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad_(True)

    @property
    def sigma(self):
        """当前光锥强度（F3 专属）"""
        if self.mode=='f3':
            return float(torch.sigmoid(
                self.blocks[0]['attn'].w_sigma).item())
        return None

# ── 标准预训练流程 ─────────────────────────────────────────────
def pretrain(model, seed=0, ep=None, lr=None, verbose=True):
    """
    模块1：多步轨迹预测预训练
    建立洛伦兹感知流形，激活 F3 的光锥几何
    """
    if ep is None:  ep=EP_PRE
    if lr is None:  lr=LR_PRE
    model.unfreeze_all()
    rng=np.random.RandomState(seed)
    odes=[(stable_ode,[0,1.0,0,1.0,0,0]),
          (running_ode,[0,1.0,0,2.0,0,0])]
    trajs=[]
    for i in range(200):
        ode_fn,ic_base=odes[i%2]; ic=list(ic_base)
        ic[0]+=rng.randn()*0.5; ic[3]+=rng.randn()*0.2; ic[1]=1.0
        t=simulate(ic,ode_fn,seed=seed+i)
        if t is not None: trajs.append(t)
    total=T_IN+T_OUT; X_list,Y_list=[],[]
    for traj in trajs:
        for s in range(0,len(traj)-total,4):
            seg=traj[s:s+total]
            if seg.shape[0]<total: break
            mu=seg.mean(0); sig=seg.std(0)+1e-8; sn=(seg-mu)/sig
            X_list.append(sn[:T_IN].astype(np.float32))
            Y_list.append(sn[T_IN:].astype(np.float32))
    X=torch.from_numpy(np.stack(X_list))
    Y=torch.from_numpy(np.stack(Y_list))
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,ep)
    loader=DataLoader(TensorDataset(X,Y),BS,shuffle=True)
    model.train()
    for e in range(ep):
        tl=0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            loss=F.mse_loss(model.forward_pretrain(xb),yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.3)
            opt.step(); tl+=loss.item()
        sched.step()
    s=f'  sigma={model.sigma:.3f}' if model.sigma else ''
    if verbose:
        print(f'  预训练完成  loss={tl/len(loader):.4f}{s}')
    return tl/len(loader)

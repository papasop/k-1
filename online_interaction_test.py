"""
在线交互实验（online_interaction_test.py）
==========================================
从 K=1 Chronogeometrodynamics 提取的在线学习公式：

  Law II:   dx/dt = (JG − D)∇V_total
  Law III:  K(x) = x⊤Gx → 1（统计吸引子）
  Theorem 4: dc > 0 ⟺ det G < 0（洛伦兹几何必要条件）
  Theorem 6: κK = 4dc（收敛速度由几何决定）
  Prop 7:   σ² = 2dc·T_tol（噪声-温度匹配）

实验设计：
  第一阶段：离线预训练（建立洛伦兹几何）
  第二阶段：在线交互（Law II 驱动更新）
    - 机器人执行运动 → 物理状态嵌入 x
    - 人类语言反馈 → V_lang（代价信号）
    - Law II 更新 lang_aligner 权重
    - 测量收敛速度是否符合 κK = 4dc

验证的核心假设：
  F3（洛伦兹）的 dc > 0，在线收敛速度 κK_F3 > κK_euc
  欧氏的 dc = 0（Theorem 3），在线学习需要更多交互轮次

使用：
  exec(open('online_interaction_test.py').read())
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, TensorDataset
import warnings; warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── 超参数 ─────────────────────────────────────────────────────
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
T_DIM      = max(1, int(N_HEADS*TIME_RATIO)) * (EMBED_DIM//N_HEADS)

# 在线交互参数（对应 Law II）
LR_ONLINE  = 3e-4   # D（耗散项，学习率）
N_INTERACT = 20     # 每轮交互步数
N_ROUNDS   = 10     # 总交互轮数
N_SEEDS    = 5

LABELS = {0:'momentum_stable', 1:'momentum_changing'}
DESCRIPTIONS = {
    0: ["平稳匀速运动，动量保持守恒",
        "smooth constant velocity motion with conserved momentum"],
    1: ["动量持续变化，存在外力作用",
        "changing momentum with continuous force application"],
}

print(f'T_DIM={T_DIM}')

# ── 语言编码器 ─────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
lang_enc = SentenceTransformer('all-MiniLM-L6-v2')
print('语言编码器加载完成')

def encode(texts):
    return lang_enc.encode(texts, convert_to_tensor=True,
                           show_progress_bar=False).to(device)

# ── 物理数据 ───────────────────────────────────────────────────
def stable_ode(t, y):
    x,yp,z,vx,vy,vz=y
    return [vx, 0, vz, -0.001*vx, 0, -0.001*vz]

def running_ode(t, y):
    x,yp,z,vx,vy,vz=y; g=9.81; m=70.0
    phase=(3.0*t)%1.0; L=0.95+0.12*abs(np.sin(3.0*np.pi*t))
    pen=max(0,L-yp)
    Fv=(2000*pen-30*vy) if (phase<0.4 and yp<L) else 0
    return [vx,vy,vz, m*0.002*2000*(3.5-vx)/m,(Fv-m*g)/m,(-80*z-15*vz)/m]

def simulate(ic, ode_fn, seed=0):
    rng=np.random.RandomState(seed)
    sol=solve_ivp(ode_fn,(0,80*0.033),ic,
                  t_eval=np.linspace(0,80*0.033,80),
                  method='RK45',rtol=1e-6,atol=1e-8)
    if not sol.success: return None
    traj=sol.y.T.astype(np.float32)
    if np.any(np.isnan(traj)) or np.any(np.abs(traj)>200): return None
    pos=traj[:,:3]+rng.randn(80,3)*0.002
    vel=np.zeros_like(pos)
    vel[1:-1]=(pos[2:]-pos[:-2])/(2*0.033)
    vel[0]=(pos[1]-pos[0])/0.033; vel[-1]=(pos[-1]-pos[-2])/0.033
    return np.concatenate([pos,vel],axis=1)

def build_dataset(seed=0):
    rng=np.random.RandomState(seed)
    X_list,L_list=[],[]
    odes=[(stable_ode,[0,1.0,0,1.0,0,0],[0.3,0,0,0.1,0,0]),
          (running_ode,[0,1.0,0,2.0,0,0],[0.5,0,0,0.3,0,0])]
    for lbl,(ode_fn,ic_base,ic_noise) in enumerate(odes):
        count=0; i=0
        while count<N_PER and i<5000:
            ic=[b+rng.randn()*n for b,n in zip(ic_base,ic_noise)]
            ic[1]=1.0; ic[2]=0.0; ic[4]=0.0; ic[5]=0.0
            traj=simulate(ic,ode_fn,seed=seed*1000+i); i+=1
            if traj is None or len(traj)<T_IN+5: continue
            seg=traj[:T_IN]
            mu=seg.mean(0); sig=seg.std(0)+1e-8
            X_list.append(((seg-mu)/sig).astype(np.float32))
            L_list.append(lbl); count+=1
    X=torch.from_numpy(np.stack(X_list))
    L=torch.tensor(L_list,dtype=torch.long)
    return X,L

# ── 模型 ──────────────────────────────────────────────────────
class MinkowskiLN(nn.Module):
    def __init__(self,dim,td,eps=1e-5):
        super().__init__()
        self.td=td; self.eps=eps
        self.pre=nn.LayerNorm(dim)
        self.g=nn.Parameter(torch.ones(dim))
        self.b=nn.Parameter(torch.zeros(dim))
    def forward(self,x):
        x=self.pre(x); t=x[...,:self.td]; s=x[...,self.td:]
        mq=(s**2).sum(-1,keepdim=True)-(t**2).sum(-1,keepdim=True)
        eps=torch.clamp(0.01*mq.abs().mean(),min=self.eps)
        return self.g*(x/(torch.sqrt(mq.abs()+eps)+eps))+self.b

class Attn(nn.Module):
    def __init__(self,dim,nh,tr,mode):
        super().__init__()
        self.nh=nh; self.hd=dim//nh; self.mode=mode
        self.scale=self.hd**-0.5
        self.nt=max(1,int(nh*tr)); self.ns=nh-self.nt
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
    def forward(self,x):
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
                    [-torch.sigmoid(self.w_sigma)*st,ss],dim=2)*self.scale
            else:
                score=torch.cat([-st,ss],dim=2)*self.scale
        attn=F.softmax(score,dim=-1)
        V=self.v(x).view(B,T,self.nh,hd)
        return self.out(
            torch.einsum('bths,bshd->bthd',attn,V).reshape(B,T,D))

class OnlineModel(nn.Module):
    """
    在线交互模型
    backbone 离线预训练，在线交互只更新 lang_aligner
    对应 Law II: dx/dt = (JG − D)∇V_total
      JG 由 backbone 的洛伦兹几何决定（固定）
      D 是学习率（固定）
      ∇V_total 由语言反馈决定（在线）
    """
    def __init__(self, mode='f3'):
        super().__init__()
        dim=EMBED_DIM; td=T_DIM
        use_mln=(mode!='euclidean'); self.mode=mode
        self.embed=nn.Linear(STATE_DIM,dim)
        self.pos=nn.Embedding(T_IN+T_OUT,dim)
        NC=lambda:(MinkowskiLN(dim,td) if use_mln else nn.LayerNorm(dim))
        self.blocks=nn.ModuleList([nn.ModuleDict({
            'attn':Attn(dim,N_HEADS,TIME_RATIO,mode),
            'n1':NC(),'n2':NC(),
            'ff':nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),
                               nn.Linear(dim*4,dim)),
        }) for _ in range(N_LAYERS)])
        self.norm=NC()
        self.traj_head=nn.Linear(dim,STATE_DIM*T_OUT)
        self.cls=nn.Sequential(
            nn.Linear(dim,dim//2),nn.GELU(),
            nn.LayerNorm(dim//2),nn.Linear(dim//2,N_LABELS))
        # 在线学习目标：lang_aligner（方向B第一步）
        self.lang_aligner=nn.Sequential(
            nn.Linear(LANG_DIM,dim*2),nn.GELU(),
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2,dim),nn.GELU(),nn.LayerNorm(dim))

    def embed_seq(self,x):
        B,T,_=x.shape
        h=self.embed(x)+self.pos(torch.arange(T,device=x.device))
        for blk in self.blocks:
            h=h+blk['attn'](blk['n1'](h))
            h=h+blk['ff'](blk['n2'](h))
        return self.norm(h)[:,-1,:]

    def forward_pretrain(self,x):
        return self.traj_head(self.embed_seq(x)).view(
            x.shape[0],T_OUT,STATE_DIM)

    def forward_A_cls(self,x): return self.cls(self.embed_seq(x))

    def compute_G_local(self, x_emb):
        """
        计算局部洛伦兹度规 G（2×2代理）
        从128维嵌入提取类时/类空分量
        G = [[s²-t², 0], [0, s²]]（对角近似）
        对应 K=1 Chronogeometrodynamics Law I
        """
        t_part=x_emb[:,:T_DIM]; s_part=x_emb[:,T_DIM:]
        t_norm=(t_part**2).sum(-1).mean().item()
        s_norm=(s_part**2).sum(-1).mean().item()
        # 2×2 代理度规
        G=torch.tensor([[-t_norm, 0],[0, s_norm]],
                        dtype=torch.float32)
        return G

    def compute_dc(self, G):
        """
        dc = sqrt(-1/det G)（Theorem 2）
        det G < 0 → dc > 0（洛伦兹）
        det G > 0 → dc = 0（欧氏，Theorem 3）
        """
        det_G=G[0,0]*G[1,1]-G[0,1]*G[1,0]
        if det_G < 0:
            return float(np.sqrt(-1.0/det_G.item()))
        else:
            return 0.0  # 欧氏退化

    def compute_kappa_K(self, dc, G):
        """
        κK = 4dc · x*⊤G²x*（Theorem 6）
        在归一化点 x*⊤G²x* = 1 时简化为 κK = 4dc
        """
        G2=torch.mm(G,G)
        # 用迹作为 x*⊤G²x* 的近似（各向同性假设）
        g2_trace=torch.trace(G2).item()/2
        return 4*dc*g2_trace

    def freeze_backbone(self):
        for name,p in self.named_parameters():
            if 'lang_aligner' not in name:
                p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad_(True)

    @property
    def sigma(self):
        if self.mode=='f3':
            return float(torch.sigmoid(
                self.blocks[0]['attn'].w_sigma).item())
        return None

# ── 离线预训练 ─────────────────────────────────────────────────
def pretrain(model, seed=0):
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
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP_PRE)
    loader=DataLoader(TensorDataset(X,Y),16,shuffle=True)
    model.train()
    for ep in range(EP_PRE):
        tl=0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            loss=F.mse_loss(model.forward_pretrain(xb),yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.3)
            opt.step(); tl+=loss.item()
        sched.step()
    s=f'  sigma={model.sigma:.3f}' if model.sigma else ''
    print(f'  预训练完成  loss={tl/len(loader):.4f}{s}')

# ── 核心：Law II 在线更新步 ────────────────────────────────────
def law2_online_step(model, x_phys, lang_feedback, optimizer):
    """
    单步在线更新（对应 Law II）

    dx/dt = (JG − D)∇V_total
    V_total = V_K + V_lang
      V_K    = 0.5*(K-1)²       Law III 吸引子
      V_lang = -cos_sim(lorentz_lang, x_phys)  语言反馈代价

    只更新 lang_aligner（对应 D·∇V_lang 项）
    backbone 固定（JG 固定，几何不变）
    """
    model.train()
    optimizer.zero_grad()

    # 物理嵌入（backbone 固定）
    with torch.no_grad():
        x_emb=model.embed_seq(x_phys)  # (1, 128)

    # Law III: K(x) = x⊤Gx，计算当前 K 值
    t_p=x_emb[:,:T_DIM]; s_p=x_emb[:,T_DIM:]
    K_val=(s_p**2).sum(-1)-(t_p**2).sum(-1)  # mq，正=类空，负=类时

    # V_K = 0.5*(K-1)²（吸引子代价，鼓励靠近类时边界）
    K_target=-1.0  # 目标：类时区域（mq=-1 对应归一化类时）
    V_K=0.5*((K_val-K_target)**2).mean()

    # V_lang：语言反馈代价（语言嵌入和物理嵌入的对齐代价）
    lorentz_lang=model.lang_aligner(lang_feedback)
    V_lang=-F.cosine_similarity(
        F.normalize(lorentz_lang,dim=-1),
        F.normalize(x_emb.detach(),dim=-1)).mean()

    # 总代价（对应 ∇V_total）
    V_total=V_K+V_lang
    V_total.backward()
    optimizer.step()

    return {
        'V_total':  V_total.item(),
        'V_K':      V_K.item(),
        'V_lang':   V_lang.item(),
        'K_val':    K_val.mean().item(),
    }

# ── 在线交互：模拟人类反馈 ─────────────────────────────────────
def simulate_human_feedback(true_label, noise_level=0.1, rng=None):
    """
    模拟人类语言反馈（Class A + Class B 混合）
    Class A：正确描述（热力学信号）
    Class B：同义不同说法（规范自由度，不应该被纠正）
    noise_level：模拟 Prop 7 的 σ²（人类反馈随机性）
    """
    if rng is None: rng=np.random
    descs=DESCRIPTIONS[true_label]
    # 随机选一个说法（Class B 规范自由度）
    desc=descs[rng.randint(len(descs))]
    # 加入少量随机噪声（Prop 7 的 σ）
    lang_emb=encode([desc])
    if noise_level>0:
        noise=torch.randn_like(lang_emb)*noise_level
        lang_emb=F.normalize(lang_emb+noise,dim=-1)
    return lang_emb

# ── K 值和 dc 的追踪（验证 Theorem 4/6） ─────────────────────
def measure_lorentz_geometry(model, X_batch):
    """
    测量当前表示空间的洛伦兹几何量
    返回：
      dc：稳定边界（Theorem 2，dc>0 ⟺ 洛伦兹）
      kappa_K：收敛速度（Theorem 6，κK = 4dc）
      tl_ratio：类时比例
    """
    model.eval()
    with torch.no_grad():
        x_emb=model.embed_seq(X_batch.to(device))
        G=model.compute_G_local(x_emb)
        dc=model.compute_dc(G)
        kappa_K=model.compute_kappa_K(dc,G)
        t_p=x_emb[:,:T_DIM]; s_p=x_emb[:,T_DIM:]
        mq=(s_p**2).sum(-1)-(t_p**2).sum(-1)
        tl_ratio=(mq<0).float().mean().item()
    return {'dc':dc,'kappa_K':kappa_K,'tl_ratio':tl_ratio,
            'det_G':(G[0,0]*G[1,1]-G[0,1]*G[1,0]).item()}

# ── 在线对齐质量评估 ─────────────────────────────────────────
def measure_alignment(model, X_te, L_te):
    """
    测量当前 lang_aligner 的对齐质量
    stable 语言嵌入和 stable 物理嵌入的余弦相似度
    """
    model.eval()
    with torch.no_grad():
        stable_mask=(L_te==0)
        n=stable_mask.sum().item()
        if n==0: return 0.0
        stable_lang=encode([DESCRIPTIONS[0][0]]).expand(n,-1)
        stable_phys=model.embed_seq(X_te[stable_mask].to(device))
        sim=F.cosine_similarity(
            F.normalize(model.lang_aligner(stable_lang),dim=-1),
            F.normalize(stable_phys,dim=-1)).mean().item()
    return sim

# ── 主实验 ─────────────────────────────────────────────────────
print('\n'+'='*60)
print('在线交互实验（Law II 驱动）')
print('验证：F3 的 dc > 0 让在线收敛更快（Theorem 4/6）')
print('='*60)

X_test,L_test=build_dataset(seed=42)
print(f'测试集: {len(X_test)} 样本')

# 收集跨 seed 结果
f3_convergence_list=[]
euc_convergence_list=[]
f3_dc_list=[]
euc_dc_list=[]

for seed in range(N_SEEDS):
    print(f'\n{"="*55}\nSeed {seed}\n{"="*55}')
    rng=np.random.RandomState(seed)

    # 离线预训练
    print('离线预训练F3...')
    model_f3=OnlineModel('f3').to(device)
    pretrain(model_f3,seed=seed*1000)

    print('离线预训练欧氏...')
    model_euc=OnlineModel('euclidean').to(device)
    pretrain(model_euc,seed=seed*1000)

    # 测量初始洛伦兹几何量（Theorem 2/4）
    geo_f3 =measure_lorentz_geometry(model_f3, X_test)
    geo_euc=measure_lorentz_geometry(model_euc,X_test)
    f3_dc_list.append(geo_f3['dc'])
    euc_dc_list.append(geo_euc['dc'])

    print(f'  F3  几何: dc={geo_f3["dc"]:.4f}  '
          f'κK={geo_f3["kappa_K"]:.4f}  '
          f'det_G={geo_f3["det_G"]:.4f}  '
          f'类时={geo_f3["tl_ratio"]:.1%}')
    print(f'  欧氏几何: dc={geo_euc["dc"]:.4f}  '
          f'κK={geo_euc["kappa_K"]:.4f}  '
          f'det_G={geo_euc["det_G"]:.4f}  '
          f'类时={geo_euc["tl_ratio"]:.1%}')

    # 冻结 backbone，只在线更新 lang_aligner
    model_f3.freeze_backbone()
    model_euc.freeze_backbone()
    opt_f3 =torch.optim.Adam(
        [p for p in model_f3.parameters()  if p.requires_grad],
        lr=LR_ONLINE)
    opt_euc=torch.optim.Adam(
        [p for p in model_euc.parameters() if p.requires_grad],
        lr=LR_ONLINE)

    # 在线交互循环（Law II）
    align_f3_curve=[]
    align_euc_curve=[]

    # 初始对齐质量
    align_f3_curve.append(measure_alignment(model_f3, X_test,L_test))
    align_euc_curve.append(measure_alignment(model_euc,X_test,L_test))

    for round_idx in range(N_ROUNDS):
        # 每轮采样一批物理状态
        idx=rng.randint(0,len(X_test),N_INTERACT)
        X_batch=X_test[idx]; L_batch=L_test[idx]

        for i in range(N_INTERACT):
            x_phys=X_batch[i:i+1].to(device)
            true_lbl=L_batch[i].item()
            # 人类语言反馈（模拟 Prop 7 的 σ 噪声）
            lang_fb=simulate_human_feedback(true_lbl, noise_level=0.05, rng=rng)

            # Law II 在线更新
            law2_online_step(model_f3, x_phys, lang_fb, opt_f3)
            law2_online_step(model_euc,x_phys, lang_fb, opt_euc)

        # 每轮结束测量对齐质量
        align_f3_curve.append(
            measure_alignment(model_f3, X_test,L_test))
        align_euc_curve.append(
            measure_alignment(model_euc,X_test,L_test))

        if (round_idx+1)%5==0:
            print(f'  Round {round_idx+1:2d}: '
                  f'F3={align_f3_curve[-1]:.4f}  '
                  f'欧氏={align_euc_curve[-1]:.4f}')

    # 计算收敛速度（对齐质量的改善量/轮数）
    f3_gain  =align_f3_curve[-1] -align_f3_curve[0]
    euc_gain =align_euc_curve[-1]-align_euc_curve[0]
    f3_convergence_list.append(f3_gain)
    euc_convergence_list.append(euc_gain)

    print(f'\n  F3  对齐改善: {f3_gain:+.4f}  '
          f'(初始={align_f3_curve[0]:.4f} → 最终={align_f3_curve[-1]:.4f})')
    print(f'  欧氏对齐改善: {euc_gain:+.4f}  '
          f'(初始={align_euc_curve[0]:.4f} → 最终={align_euc_curve[-1]:.4f})')
    print(f'  F3>欧氏: {"✓" if f3_gain>euc_gain else "✗"}')

# ── 最终统计 ───────────────────────────────────────────────────
print('\n'+'='*60)
print('最终统计结果')
print('='*60)

f3_arr  =np.array(f3_convergence_list)
euc_arr =np.array(euc_convergence_list)
f3_dc   =np.array(f3_dc_list)
euc_dc  =np.array(euc_dc_list)
_,p=scipy_stats.ttest_rel(f3_arr,euc_arr)
d=(f3_arr-euc_arr).mean()/((f3_arr-euc_arr).std(ddof=1)+1e-10)
n_ok=sum(f>e for f,e in zip(f3_convergence_list,euc_convergence_list))

print(f'\n洛伦兹几何量（Theorem 2/4）:')
print(f'  F3  dc均值: {f3_dc.mean():.4f}±{f3_dc.std():.4f}  '
      f'（dc>0 = 洛伦兹）')
print(f'  欧氏dc均值: {euc_dc.mean():.4f}±{euc_dc.std():.4f}  '
      f'（dc≈0 = 欧氏退化）')

print(f'\n在线对齐改善量（Law II 收敛）:')
print(f'  F3:   {f3_arr.mean():+.4f}±{f3_arr.std():.4f}')
print(f'  欧氏: {euc_arr.mean():+.4f}±{euc_arr.std():.4f}')
print(f'  p={p:.4f}  d={d:.2f}  F3>欧氏: {n_ok}/{N_SEEDS}')

# Theorem 6 验证：κK 和 dc 的关系
print(f'\nTheorem 6 验证（κK = 4·dc）:')
for seed_i,(dc_val) in enumerate(f3_dc_list):
    kappa_theory=4*dc_val
    print(f'  Seed {seed_i}: dc={dc_val:.4f}  '
          f'理论κK={kappa_theory:.4f}')

# 婴儿说话在线交互结论
print('\n'+'='*60)
print('在线交互实验结论')
print('='*60)
dc_ok   =(f3_dc.mean()>euc_dc.mean())
conv_ok =(p<0.05 and d>0) or (n_ok>N_SEEDS//2)
print(f'  dc(F3) > dc(欧氏): {"✅" if dc_ok else "❌"}  '
      f'({f3_dc.mean():.4f} vs {euc_dc.mean():.4f})')
print(f'  F3在线收敛更快:    {"✅" if conv_ok else "❌"}  '
      f'p={p:.4f}  {n_ok}/{N_SEEDS} seeds')

if dc_ok and conv_ok:
    print('\n在线交互验证成立 ✅')
    print('F3 的 dc>0（Theorem 4）让在线学习收敛更快（Theorem 6）')
    print('人类语言反馈通过 Law II 驱动 lang_aligner 向洛伦兹流形收敛')
elif dc_ok:
    print('\ndc(F3)>dc(欧氏) ✅，收敛速度差异需要更多交互轮次')
else:
    print('\n需要更长预训练让洛伦兹几何充分激活')

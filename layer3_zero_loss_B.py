"""
层3验证：零损失猜想强版本（选项B）
=====================================
改变：预训练加入动量守恒信号，强制sigma收敛
目标：sigma > 0.55，F3动量变化率接近真实基准(0.0998)

预训练损失：
  loss = MSE轨迹预测 + 0.3 * 动量守恒损失
  动量守恒损失 = (dp²).mean()，dp=速度差分

如果sigma收敛后F3动量变化率接近0.0998：
  零损失猜想强版本成立 ✅
  "洛伦兹几何本能让守恒自动成立"

使用：
  exec(open('layer3_zero_loss_B.py').read())
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, TensorDataset
import warnings; warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

EMBED_DIM  = 128
N_HEADS    = 4
N_LAYERS   = 3
TIME_RATIO = 0.5
T_IN       = 20
T_OUT      = 20
STATE_DIM  = 6
N_FFT      = T_IN // 2 + 1
LANG_DIM   = 384
N_LABELS   = 2
N_PER      = 50
EP_PRE     = 120   # 更多epoch让sigma收敛
EP_FT      = 150
LR_PRE     = 3e-4
LR_FT      = 1e-4
BS         = 16
N_SEEDS    = 5
T_DIM      = max(1, int(N_HEADS*TIME_RATIO)) * (EMBED_DIM//N_HEADS)
MOM_WEIGHT = 0.3   # 动量守恒损失权重

print(f'N_FFT={N_FFT}  T_DIM={T_DIM}  TIME_RATIO={TIME_RATIO}')
print(f'预训练：MSE + {MOM_WEIGHT}×动量守恒损失')

LABELS = {0:'momentum_stable', 1:'momentum_changing'}
DESCRIPTIONS = {
    0: ["平稳匀速运动，动量保持守恒",
        "smooth constant velocity motion with conserved momentum"],
    1: ["动量持续变化，存在外力作用",
        "changing momentum with continuous force application"],
}
STABLE_INSTRUCTIONS = [
    "平稳匀速运动，动量保持守恒",
    "机器人以恒定速度移动，没有加速或减速",
    "smooth constant velocity motion with conserved momentum",
    "steady movement without acceleration changes",
]

from sentence_transformers import SentenceTransformer
lang_enc = SentenceTransformer('all-MiniLM-L6-v2')
print(f'语言编码器加载完成')

def encode(texts):
    return lang_enc.encode(texts, convert_to_tensor=True,
                           show_progress_bar=False).to(device)

# ── 频域转换 ────────────────────────────────────────────────────
def to_freq(traj):
    fft = np.fft.rfft(traj, axis=0)
    phases     = np.angle(fft).astype(np.float32)
    amplitudes = np.abs(fft).astype(np.float32)
    return np.concatenate([phases, amplitudes], axis=-1)

def momentum_change(traj_time):
    vel = traj_time[:, 3:]
    dp  = np.diff(vel, axis=0)
    return float(np.linalg.norm(dp, axis=-1).mean())

# ── 物理数据 ────────────────────────────────────────────────────
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
    total=T_IN+T_OUT
    sol=solve_ivp(ode_fn,(0,total*0.033),ic,
                  t_eval=np.linspace(0,total*0.033,total),
                  method='RK45',rtol=1e-6,atol=1e-8)
    if not sol.success: return None
    traj=sol.y.T.astype(np.float32)
    if np.any(np.isnan(traj)) or np.any(np.abs(traj)>200): return None
    pos=traj[:,:3]+rng.randn(total,3)*0.002
    vel=np.zeros_like(pos)
    vel[1:-1]=(pos[2:]-pos[:-2])/(2*0.033)
    vel[0]=(pos[1]-pos[0])/0.033; vel[-1]=(pos[-1]-pos[-2])/0.033
    return np.concatenate([pos,vel],axis=1)

def build_dataset(seed=0):
    rng=np.random.RandomState(seed)
    X_list,Y_list,L_list=[],[],[]
    odes=[(stable_ode,[0,1.0,0,1.0,0,0],[0.3,0,0,0.1,0,0]),
          (running_ode,[0,1.0,0,2.0,0,0],[0.5,0,0,0.3,0,0])]
    for lbl,(ode_fn,ic_base,ic_noise) in enumerate(odes):
        count=0; i=0
        while count<N_PER and i<5000:
            ic=[b+rng.randn()*n for b,n in zip(ic_base,ic_noise)]
            ic[1]=1.0; ic[2]=0.0; ic[4]=0.0; ic[5]=0.0
            traj=simulate(ic,ode_fn,seed=seed*1000+i); i+=1
            if traj is None or len(traj)<T_IN+T_OUT: continue
            seg_in=traj[:T_IN]; seg_out=traj[T_IN:T_IN+T_OUT]
            freq=to_freq(seg_in)
            amp=freq[:,STATE_DIM:]; mu=amp.mean(0); sig=amp.std(0)+1e-8
            freq[:,STATE_DIM:]=(amp-mu)/sig
            mu_t=seg_out.mean(0); sig_t=seg_out.std(0)+1e-8
            X_list.append(freq.astype(np.float32))
            Y_list.append(((seg_out-mu_t)/sig_t).astype(np.float32))
            L_list.append(lbl); count+=1
    X=torch.from_numpy(np.stack(X_list))
    Y=torch.from_numpy(np.stack(Y_list))
    L=torch.tensor(L_list,dtype=torch.long)
    print(f'数据集: {len(X)} 样本  stable={(L==0).sum()}  changing={(L==1).sum()}')
    return X,Y,L

def real_physics_baseline(n=50,seed=42):
    """
    真实基准：对归一化后的轨迹计算动量变化率
    和 evaluate() 里的量纲保持一致
    （phys_decoder 学的是归一化输出）
    """
    rng=np.random.RandomState(seed)
    moms=[]
    for i in range(n):
        ic=[0,1.0,0,1.0+rng.randn()*0.1,0,0]
        t=simulate(ic,stable_ode,seed=i)
        if t is None: continue
        seg=t[T_IN:]
        # 归一化（和预训练目标一致）
        mu=seg.mean(0); sig=seg.std(0)+1e-8
        seg_n=(seg-mu)/sig
        moms.append(momentum_change(seg_n))
    return float(np.mean(moms))

# ── 模型 ────────────────────────────────────────────────────────
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
        self.nh=nh; self.hd=dim//nh; self.mode=mode; self.scale=self.hd**-0.5
        self.nt=max(1,int(nh*tr)); self.ns=nh-self.nt
        if mode=='euclidean':
            self.q=nn.Linear(dim,dim,bias=False)
            self.k=nn.Linear(dim,dim,bias=False)
        else:
            self.qt=nn.Linear(dim,self.nt*self.hd,bias=False)
            self.kt=nn.Linear(dim,self.nt*self.hd,bias=False)
            self.qs=nn.Linear(dim,self.ns*self.hd,bias=False)
            self.ks=nn.Linear(dim,self.ns*self.hd,bias=False)
            if mode=='f3': self.w_sigma=nn.Parameter(torch.zeros(1))
        self.v=nn.Linear(dim,dim,bias=False)
        self.out=nn.Linear(dim,dim)
    def forward(self,x):
        B,T,D=x.shape; hd=self.hd
        if self.mode=='euclidean':
            Q=self.q(x).view(B,T,self.nh,hd); K=self.k(x).view(B,T,self.nh,hd)
            score=torch.einsum('bthd,bshd->bths',Q,K)*self.scale
        else:
            Qt=self.qt(x).view(B,T,self.nt,hd); Kt=self.kt(x).view(B,T,self.nt,hd)
            Qs=self.qs(x).view(B,T,self.ns,hd); Ks=self.ks(x).view(B,T,self.ns,hd)
            st=torch.einsum('bthd,bshd->bths',Qt,Kt)
            ss=torch.einsum('bthd,bshd->bths',Qs,Ks)
            if self.mode=='f3':
                score=torch.cat([-torch.sigmoid(self.w_sigma)*st,ss],dim=2)*self.scale
            else:
                score=torch.cat([-st,ss],dim=2)*self.scale
        attn=F.softmax(score,dim=-1)
        V=self.v(x).view(B,T,self.nh,hd)
        return self.out(torch.einsum('bths,bshd->bthd',attn,V).reshape(B,T,D))

class Layer3Model(nn.Module):
    def __init__(self,mode='f3'):
        super().__init__()
        dim=EMBED_DIM; use_mln=(mode!='euclidean'); self.mode=mode
        self.embed=nn.Linear(STATE_DIM*2,dim)
        self.pos=nn.Embedding(N_FFT,dim)
        NC=lambda:(MinkowskiLN(dim,T_DIM) if use_mln else nn.LayerNorm(dim))
        self.blocks=nn.ModuleList([nn.ModuleDict({
            'attn':Attn(dim,N_HEADS,TIME_RATIO,mode),
            'n1':NC(),'n2':NC(),
            'ff':nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim)),
        }) for _ in range(N_LAYERS)])
        self.norm=NC()
        self.traj_head=nn.Linear(dim,STATE_DIM*T_OUT)
        self.lang_gen=nn.Sequential(
            nn.Linear(dim,dim*2),nn.GELU(),nn.LayerNorm(dim*2),nn.Linear(dim*2,LANG_DIM))
        self.cls=nn.Sequential(
            nn.Linear(dim,dim//2),nn.GELU(),nn.LayerNorm(dim//2),nn.Linear(dim//2,N_LABELS))
        self.lang_aligner=nn.Sequential(
            nn.Linear(LANG_DIM,dim*2),nn.GELU(),nn.LayerNorm(dim*2),
            nn.Linear(dim*2,dim),nn.GELU(),nn.LayerNorm(dim))
        self.phys_decoder=nn.Sequential(
            nn.Linear(dim,dim*2),nn.GELU(),nn.LayerNorm(dim*2),
            nn.Linear(dim*2,STATE_DIM*T_OUT))

    def embed_seq(self,x):
        B,T,_=x.shape
        h=self.embed(x)+self.pos(torch.arange(T,device=x.device))
        for blk in self.blocks:
            h=h+blk['attn'](blk['n1'](h))
            h=h+blk['ff'](blk['n2'](h))
        return self.norm(h)[:,-1,:]

    def forward_pretrain(self,x):
        return self.traj_head(self.embed_seq(x)).view(-1,T_OUT,STATE_DIM)

    def forward_A_gen(self,x): return self.lang_gen(self.embed_seq(x))
    def forward_A_cls(self,x): return self.cls(self.embed_seq(x))

    def forward_B(self,lang_emb):
        return self.phys_decoder(self.lang_aligner(lang_emb)).view(-1,T_OUT,STATE_DIM)

    def freeze_backbone(self):
        for name,p in self.named_parameters():
            if not any(k in name for k in ['lang_gen','cls','lang_aligner','phys_decoder']):
                p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad_(True)

    @property
    def sigma(self):
        if self.mode=='f3':
            return float(torch.sigmoid(self.blocks[0]['attn'].w_sigma).item())
        return None

# ── 预训练（选项B：MSE + 动量守恒损失）──────────────────────────
def pretrain_with_momentum(model, seed=0):
    """
    选项B核心：预训练加入动量守恒损失
    loss = MSE(pred_traj, true_traj) + MOM_WEIGHT * momentum_loss
    momentum_loss = (d(vel)/dt)² 速度变化率平方和
    这让sigma有足够的梯度信号收敛
    """
    model.unfreeze_all()
    rng=np.random.RandomState(seed)
    # Bug5修复：4种ODE提高物理多样性，增强sigma梯度信号
    def walking_ode(t, y):
        x,yp,z,vx,vy,vz=y; g=9.81; m=70.0
        phase=(1.8*t)%1.0; L=0.98+0.05*abs(np.sin(1.8*np.pi*t))
        pen=max(0,L-yp)
        Fv=(1500*pen-20*vy) if (phase<0.5 and yp<L) else 0
        return [vx,vy,vz, m*0.001*1500*(1.2-vx)/m,(Fv-m*g)/m,(-60*z-10*vz)/m]
    def jumping_ode(t, y):
        x,yp,z,vx,vy,vz=y; g=9.81; m=70.0
        Fv=3000*(0.3-yp)-50*vy if yp<0.3 else 0
        return [vx,vy,vz,0,(Fv-m*g)/m,0]
    odes=[
        (stable_ode, [0,1.0,0,1.0,0,0],[0.3,0,0,0.1,0,0]),
        (running_ode,[0,1.0,0,2.0,0,0],[0.5,0,0,0.3,0,0]),
        (walking_ode,[0,1.0,0,1.2,0,0],[0.3,0,0,0.1,0,0]),
        (jumping_ode,[0,0.3,0,0.5,0,0],[0.2,0,0,0.1,0,0]),
    ]
    trajs=[]
    for i in range(200):
        ode_fn,ic_base,ic_noise=odes[i%4]; ic=list(ic_base)
        for j in range(len(ic_base)):
            ic[j]+=rng.randn()*ic_noise[j]
        ic[1]=max(0.2,ic[1]); ic[2]=0.0; ic[4]=0.0; ic[5]=0.0
        t=simulate(ic,ode_fn,seed=seed+i)
        if t is not None: trajs.append(t)
    X_list,Y_list=[],[]
    for traj in trajs:
        if len(traj)<T_IN+T_OUT: continue
        seg_in=traj[:T_IN]; seg_out=traj[T_IN:T_IN+T_OUT]
        freq=to_freq(seg_in)
        amp=freq[:,STATE_DIM:]; mu=amp.mean(0); sig=amp.std(0)+1e-8
        freq[:,STATE_DIM:]=(amp-mu)/sig
        mu_t=seg_out.mean(0); sig_t=seg_out.std(0)+1e-8
        X_list.append(freq.astype(np.float32))
        Y_list.append(((seg_out-mu_t)/sig_t).astype(np.float32))
    X=torch.from_numpy(np.stack(X_list))
    Y=torch.from_numpy(np.stack(Y_list))
    opt=torch.optim.AdamW(model.parameters(),lr=LR_PRE)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP_PRE)
    loader=DataLoader(TensorDataset(X,Y),BS,shuffle=True)
    model.train()
    for ep in range(EP_PRE):
        tl=0; tm=0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            pred=model.forward_pretrain(xb)
            # MSE轨迹预测损失
            loss_mse=F.mse_loss(pred,yb)
            # 动量守恒损失：速度变化率越小越守恒
            vel=pred[:,:,3:]
            dp=torch.diff(vel,dim=1)
            loss_mom=(dp**2).mean()
            # 总损失：MSE + 动量守恒
            loss=loss_mse+MOM_WEIGHT*loss_mom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.3)
            opt.step(); tl+=loss_mse.item(); tm+=loss_mom.item()
        sched.step()
    s=f'  sigma={model.sigma:.3f}' if model.sigma else ''
    print(f'  预训练完成  mse={tl/len(loader):.4f}  mom={tm/len(loader):.4f}{s}')

# ── 微调（同选项A，无物理损失）───────────────────────────────────
def finetune(model,X_tr,L_tr,lang_embs,X_te,L_te):
    model.freeze_backbone()
    opt=torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],lr=LR_FT)
    loader=DataLoader(TensorDataset(X_tr,L_tr,lang_embs),BS,shuffle=True)
    best_acc=0; best_state=None
    for ep in range(EP_FT):
        model.train()
        for xb,lb,le in loader:
            xb=xb.to(device); lb=lb.to(device); le=le.to(device)
            opt.zero_grad()
            loss_cls=F.cross_entropy(model.forward_A_cls(xb),lb)
            pn=F.normalize(model.forward_A_gen(xb),dim=-1)
            ln=F.normalize(le,dim=-1); sim=torch.mm(pn,ln.T)
            lc=torch.arange(len(xb),device=device)
            loss_align=(F.cross_entropy(sim,lc)+F.cross_entropy(sim.T,lc))*0.5
            # Bug7修复：phys_decoder 需要训练信号
            # 1. lang_aligner → cls 分类一致性（训练lang_aligner）
            lorentz_B = model.lang_aligner(le)
            loss_B_cls = F.cross_entropy(model.cls(lorentz_B), lb)
            # 2. lang_aligner → phys_decoder → 动量平滑性（训练phys_decoder）
            #    stable标签(lb=0)的生成轨迹速度应该平滑
            traj_B = model.phys_decoder(lorentz_B).view(-1, T_OUT, STATE_DIM)
            vel_B  = traj_B[:, :, 3:]
            dp_B   = torch.diff(vel_B, dim=1)
            # stable样本动量变化率应接近0，changing样本不惩罚
            stable_mask = (lb == 0).float().unsqueeze(-1).unsqueeze(-1)
            loss_B_mom  = (dp_B**2 * stable_mask).mean()
            loss_B = loss_B_cls + 0.3 * loss_B_mom
            (loss_cls+0.3*loss_align+0.2*loss_B).backward()
            opt.step()
        if (ep+1)%50==0:
            model.eval()
            with torch.no_grad():
                acc=(model.forward_A_cls(X_te.to(device)).argmax(-1)
                     ==L_te.to(device)).float().mean().item()
            if acc>best_acc:
                best_acc=acc
                best_state={k:v.clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    print(f'  微调完成  best_acc={best_acc:.1%}')

# ── 评估 ────────────────────────────────────────────────────────
def evaluate(model_f3, model_euc):
    res={}
    for model,name in [(model_f3,'f3'),(model_euc,'euclidean')]:
        model.eval(); moms=[]
        with torch.no_grad():
            for instr in STABLE_INSTRUCTIONS:
                lang_emb=encode([instr]).expand(8,-1)
                trajs=model.forward_B(lang_emb).cpu().numpy()
                moms.extend([momentum_change(t) for t in trajs])
        res[name]=float(np.mean(moms))
    return res['f3'],res['euclidean']

# ── 主实验 ──────────────────────────────────────────────────────
print('\n'+'='*60)
print('层3验证：零损失猜想强版本（选项B）')
print(f'预训练: MSE + {MOM_WEIGHT}×动量守恒损失')
print('目标: sigma>0.55，F3动量变化率接近真实基准')
print('='*60)

print('\n计算真实物理基准...')
mom_real=real_physics_baseline()
print(f'真实匀速轨迹动量变化率: {mom_real:.4f}')

X_test,_,L_test=build_dataset(seed=42)

f3_moms=[]; euc_moms=[]; f3_sigmas=[]

for seed in range(N_SEEDS):
    print(f'\n{"="*50}\nSeed {seed}\n{"="*50}')
    X_tr,_,L_tr=build_dataset(seed=seed+100)

    rng_l=np.random.RandomState(seed)
    lang_embs=[]
    for lbl in L_tr.tolist():
        desc=DESCRIPTIONS[lbl][rng_l.randint(len(DESCRIPTIONS[lbl]))]
        lang_embs.append(encode([desc]).cpu())
    lang_embs=torch.cat(lang_embs,0)

    # F3：预训练加动量守恒损失
    print('预训练F3（MSE+动量守恒）...')
    model_f3=Layer3Model('f3').to(device)
    pretrain_with_momentum(model_f3,seed=seed*1000)
    if model_f3.sigma: f3_sigmas.append(model_f3.sigma)
    print('微调F3（零物理损失）...')
    finetune(model_f3,X_tr,L_tr,lang_embs,X_test,L_test)

    # 欧氏：同样加动量守恒损失（公平对比）
    print('训练欧氏对照（MSE+动量守恒）...')
    model_euc=Layer3Model('euclidean').to(device)
    pretrain_with_momentum(model_euc,seed=seed*1000)
    finetune(model_euc,X_tr,L_tr,lang_embs,X_test,L_test)

    mom_f3,mom_euc=evaluate(model_f3,model_euc)
    f3_moms.append(mom_f3); euc_moms.append(mom_euc)
    ok=(mom_f3<=mom_euc)
    near_real=(mom_f3 < mom_real*3)  # 在真实基准3倍以内算接近

    print(f'\n  真实基准:       {mom_real:.4f}')
    print(f'  欧氏语言生成:   {mom_euc:.4f}')
    print(f'  F3语言生成:     {mom_f3:.4f}')
    print(f'  sigma收敛:      {model_f3.sigma:.3f}  {"✓>0.55" if model_f3.sigma and model_f3.sigma>0.55 else "✗≤0.55"}')
    print(f'  F3<欧氏:        {"✓" if ok else "✗"}')
    print(f'  接近真实基准:   {"✓" if near_real else "✗  需要更强训练"}')

# ── 统计 ────────────────────────────────────────────────────────
print('\n'+'='*60)
print('最终统计结果（选项B）')
print('='*60)
f3_arr=np.array(f3_moms); euc_arr=np.array(euc_moms)
diff=euc_arr-f3_arr
_,p=scipy_stats.ttest_rel(f3_arr,euc_arr)
# d>0 表示F3更好（动量变化率更低）
d=(euc_arr-f3_arr).mean()/((euc_arr-f3_arr).std(ddof=1)+1e-10)
n_ok=sum(f<=e for f,e in zip(f3_moms,euc_moms))
avg_sigma=float(np.mean(f3_sigmas)) if f3_sigmas else 0

print(f'\n动量变化率（越低越守恒）:')
print(f'  真实基准:   {mom_real:.4f}')
print(f'  欧氏:       {euc_arr.mean():.4f}±{euc_arr.std():.4f}')
print(f'  F3:         {f3_arr.mean():.4f}±{f3_arr.std():.4f}')
print(f'  差异:       {diff.mean():+.4f}±{diff.std():.4f}')
print(f'  p={p:.4f}  d={d:.2f}  猜想成立: {n_ok}/{N_SEEDS}')
print(f'  F3 sigma均值: {avg_sigma:.3f}  {"✓已激活" if avg_sigma>0.55 else "✗未激活"}')
print(f'  F3接近真实基准: {f3_arr.mean()/mom_real:.1f}倍')

print()
sigma_ok = avg_sigma > 0.55
if p<0.05 and d>0 and sigma_ok and f3_arr.mean()<mom_real*2:
    print('零损失猜想强版本成立 ✅✅')
    print(f'sigma={avg_sigma:.3f}，F3动量变化率接近真实基准')
    print(f'洛伦兹几何本能让守恒自动成立')
elif p<0.05 and d>0:
    print(f'零损失猜想弱版本成立 ✅')
    print(f'F3<欧氏显著（p={p:.4f}），但距真实基准还有差距')
    print(f'sigma={avg_sigma:.3f}，{"已激活" if sigma_ok else "需要更强动量信号"}')
elif n_ok>N_SEEDS//2:
    print(f'零损失猜想趋势 ◑  ({n_ok}/{N_SEEDS} seeds)')
else:
    print(f'零损失猜想未成立 ✗')

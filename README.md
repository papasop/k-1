

# 克隆仓库 | Clone repository
git clone https://github.com/YOUR_USERNAME/k1-transformer.git
cd k1-transformer

# 安装依赖 (仅需NumPy) | Install dependencies
pip install numpy matplotlib

# 或安装PyTorch版本 | Or install PyTorch version
pip install torch numpy matplotlib






文件结构 | Repository Structure

k1-transformer/
├── README.md              ⭐ 最重要
├── LICENSE                ⭐ 必需
├── .gitignore             ⭐ 必需
│
├── k1_unified.py          ⭐ 核心代码
├── k1_train_test.py       ⭐ 训练测试
├── k1_colab.py            ⭐ Colab版本
│
├── assets/
│   ├── k1_training.png    ⭐ 训练图表
│   └── k1_claims_vs_reality.png
│
└── examples/
    ├── quick_demo.py
    └── text_test.py


三层架构 | 3-Layer Architecture

┌─────────────────────────────────┐
│  K1Transformer (完整系统)      │
│  = Base + Monitor              │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  K1Monitor (理论监控)          │
│  - 计算 K, Sig(G), ΔV         │
│  - 验证 Law I-III              │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  BaseTransformer (标准实现)    │
│  - 可独立使用                   │
│  - 无K=1依赖                    │
└─────────────────────────────────┘





K=1 Chronogeometrodynamics
https://doi.org/10.5281/zenodo.18949565

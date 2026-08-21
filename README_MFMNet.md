# MFMNet: 基于三层图金字塔与跨尺度注意力路由的协同药物组合预测

## 项目简介

药物组合疗法能够通过同时作用于不同分子靶点或信号通路提高治疗效果，并在一定程度上降低耐药风险。然而，潜在药物组合数量巨大，完全依赖实验方法筛选候选组合需要较高的时间和实验成本。

本项目提出 **MFMNet**，一种用于抗癌协同药物组合预测的多模态深度学习框架。对于给定的药物对–细胞系记录，MFMNet同时利用药物分子图、药物嵌入特征和细胞系转录组特征。

MFMNet采用 **Tri-Layer Graph Pyramid（TGP）**，通过 GCN、GIN 和 GATv2 三个并行图神经网络分支学习互补的药物分子表示，并利用 **Cross-Scale Attention Router（CSAR）** 对不同图表示进行自适应特征级融合。融合后的药物图表示进一步与药物嵌入和细胞系特征联合输入分类器，用于预测药物组合是否具有协同作用。

在初始五折交叉验证分析中，MFMNet取得了 **0.9301 ± 0.0036** 的平均 AUROC。在更严格的 fold-matched evaluation protocol 下，MFMNet取得了 **0.9227 ± 0.0055** 的 AUROC 和 **0.9127 ± 0.0084** 的 PR-AUC。

**关键词：** 药物组合；图神经网络；多模态特征融合；Tri-Layer Graph Pyramid；Cross-Scale Attention Router；药物协同预测

---

## 项目结构

```text
MFMNet/
├── data/                       # 原始数据及预处理数据
├── result/                     # 五折结果、统计分析、消融实验及图表
├── model.py                    # 基础模型代码
├── enhanced_model.py           # 当前 MFMNet 主模型结构
├── enhanced_main.py            # MFMNet 训练与评估入口
├── model_analysis.py           # 模型性能分析与比较
├── attention.py                # 注意力相关辅助模块
├── muti_head_attention.py      # 早期多头注意力实验模块
├── transformer.py              # 早期 Transformer 实验模块
├── loader.py                   # 数据读取与预处理
└── main.py                     # 基础模型训练入口
```

> `muti_head_attention.py` 和 `transformer.py` 保留用于早期实验和代码复现。当前论文报告的 MFMNet 主干结构采用 **TGP + CSAR**，不将 Transformer 或多头注意力模块作为当前核心模型组成部分。

---

## 环境要求

MFMNet基于 **PyTorch** 和 **PyTorch Geometric** 构建，推荐使用 Python 3.11。

```bash
conda create -n MFMNet python=3.11
conda activate MFMNet

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu121

conda install pyg -c pyg
conda install pandas
conda install rdkit
conda install scipy==1.11.4
conda install tqdm
conda install matplotlib scikit-learn
pip install scikit-optimize
```

如需运行部分已有可视化脚本，可额外安装：

```bash
pip install seaborn
```

---

## 兼容性说明

当前代码针对不同版本的 PyTorch、PyTorch Geometric 和 SciPy 进行了兼容性处理，主要包括：

1. 移除部分对新版 PyTorch 特有接口的依赖；
2. 增加必要的兼容函数以适配不同 PyTorch 版本；
3. 将已弃用的 `scipy.interp` 替换为 `numpy.interp`；
4. 将旧版 `torch_geometric.data.DataLoader` 调整为 `torch_geometric.loader.DataLoader`；
5. 增加设备检测逻辑，可根据运行环境自动选择 CUDA 或 CPU；
6. 对固定长度分子图中的填充节点进行有效位置标记，使无效填充位置不参与图归一化和图级平均池化。

---

## 数据集

本项目使用公开药物组合数据进行训练与评估。

当前主要实验数据包含：

- 13,243 条药物对–细胞系记录；
- 38 种不同药物；
- 31 个癌细胞系；
- 6,188 条协同样本；
- 7,055 条非协同样本。

标签定义：

- synergy score > 10：协同；
- synergy score < 0：非协同；
- synergy score 位于 0–10 的记录不纳入当前二分类实验。

由于部分数据文件体积较大，未直接上传至 GitHub。数据可从以下地址下载：

https://drive.google.com/drive/folders/10M9KRnyQR-XR1VStL3RdayFSr81paEI2?usp=drive_link

下载后请将相应文件放入：

```text
data/
```

---

## 数据预处理

对于每个药物对–细胞系记录：

1. 使用 RDKit 将两个药物的 SMILES 转换为分子图；
2. 每个原子编码为 78 维原子特征向量；
3. 每个分子图最多保留 100 个原子位置；
4. 不足部分采用零填充，并生成 validity mask；
5. 填充位置不参与 GraphNorm 和图级 masked mean pooling；
6. 每个药物同时使用 768 维药物嵌入特征；
7. 每个细胞系使用 954 维转录组特征。

---

## 模型架构

### 1. Tri-Layer Graph Pyramid

对于每个药物分子图，TGP并行采用三个图神经网络分支：

```text
GCN:   78 → 512 → 128
GIN:   78 → 512 → 128
GATv2: 78 → 512 → 128
```

三个分支分别学习不同类型的分子结构表示，并输出 128 维节点特征。

GIN 第一层后使用 GraphNorm，零填充位置不参与归一化统计。

### 2. Cross-Scale Attention Router

三个图分支的节点表示进行拼接：

```text
128 + 128 + 128 = 384
```

CSAR通过特征级门控网络生成自适应权重：

```text
384 → 192 → 384 → Sigmoid
```

门控后的特征进一步投影为：

```text
384 → 128
```

随后采用 masked mean pooling 得到每个药物的 128 维图级表示。

### 3. Drug Embedding Processor

每个药物包含 768 维辅助药物嵌入：

```text
768 → 512 → 256
```

### 4. Cell-Line Encoder

每个癌细胞系由 954 维转录组特征表示：

```text
954 → 512 → 256
```

### 5. Multimodal Feature Fusion

最终分类输入为：

```text
Drug A embedding       256
Drug B embedding       256
Cell-line feature      256
Drug A graph feature   128
Drug B graph feature   128
--------------------------------
Total                 1024
```

### 6. Classification Module

分类器结构为：

```text
1024 → 512 → 128 → 32 → 2
```

最终通过 Softmax 输出协同与非协同类别概率。

---

## 模型训练

运行当前 MFMNet：

```bash
python enhanced_main.py
```

当前正式实验采用五折 outer-test evaluation。对于每个外层折，从剩余训练数据中进一步划分验证集，用于模型选择和 early stopping，外层测试折不参与模型选择。

主要训练参数：

```text
Maximum epochs: 300
Batch size: 64
Optimizer: Adam
Learning rate: 5e-4
Weight decay: 1e-5
Dropout: 0.3
Early-stopping patience: 20
```

模型选择依据验证集 AUROC。

如需运行基础模型：

```bash
python main.py
```

---

## 小样本测试模式

为方便代码调试，可在 `enhanced_main.py` 中启用小样本测试：

```python
small_sample_test = True
sample_size = 100

quick_test_mode = True
```

小样本模式仅用于：

- 检查代码是否能够正常运行；
- 快速验证数据读取流程；
- 调试网络结构；
- 在资源受限环境中测试运行环境。

> 小样本测试结果不作为论文正式实验结果。

---

## 模型分析

运行：

```bash
python model_analysis.py
```

可用于分析：

- AUROC；
- PR-AUC；
- Accuracy；
- Balanced Accuracy；
- Precision；
- Recall；
- F1-score；
- 不同模型之间的性能差异。

相关输出保存在：

```text
result/
```

---

## 主要实验结果

### 初始五折交叉验证

```text
AUROC = 0.9301 ± 0.0036
```

### Fold-matched evaluation

```text
AUROC  = 0.9227 ± 0.0055
PR-AUC = 0.9127 ± 0.0084
```

MFMNet进一步与以下模型进行了相同外层测试折比较：

- DeepDDS-GCN；
- DeepDDS-GAT；
- XGBoost。

AUROC 和 PR-AUC 的配对比较采用 two-sided paired t-test，并分别进行 Holm multiple-comparison correction。

---

## 消融实验

主要受控消融变体包括：

- w/o GCN；
- w/o GIN；
- w/o GAT；
- w/o CSAR, mean fusion；
- simple concatenation + projection。

| Variant | AUROC mean ± SD | PR-AUC mean ± SD |
|---|---:|---:|
| Full MFMNet | 0.9227 ± 0.0055 | 0.9127 ± 0.0084 |
| w/o GCN | 0.9213 ± 0.0039 | 0.9141 ± 0.0046 |
| w/o GIN | 0.9206 ± 0.0024 | 0.9120 ± 0.0064 |
| w/o GAT | 0.9225 ± 0.0032 | 0.9147 ± 0.0047 |
| w/o CSAR, mean fusion | 0.9192 ± 0.0041 | 0.9099 ± 0.0054 |
| Simple concatenation + projection | 0.9182 ± 0.0010 | 0.9102 ± 0.0044 |

这些消融比较在 Holm correction 后未达到统计显著水平，因此相关结果主要用于分析不同图分支及特征融合方法的描述性贡献。

---

## 外部验证

### AstraZeneca

```text
Accuracy = 93.3%
```

### FRIEDMAN

```text
AUROC  = 0.7704
PR-AUC = 0.6602
```

### NCI-ALMANAC

```text
AUROC  = 0.5407
PR-AUC = 0.2455
```

FRIEDMAN结果提供了一定的跨数据集预测支持，而NCI-ALMANAC上的性能明显降低，说明模型跨数据集泛化能力仍会受到数据来源、样本组成和协同作用评分定义差异的影响。

---

## 计算效率

当前 MFMNet 主干模型约包含：

```text
2.65 million trainable parameters
```

五折实验平均训练时间约为：

```text
27.6 min / fold
```

训练阶段平均峰值加速器显存占用约为：

```text
565.3 MB
```

当 batch size = 64 时：

```text
Inference latency ≈ 1.00 ms/sample
Throughput ≈ 1003 samples/s
```

随着 batch size 从 32 增加至 512，吞吐量约从 804 samples/s 提高至 1256 samples/s，同时显存占用相应增加。

---

## 模型特点

1. **异构图表示学习**：并行使用 GCN、GIN 和 GATv2 获取不同类型的分子结构表示。
2. **Tri-Layer Graph Pyramid**：将多个图神经网络分支组织为统一的分子图表示模块。
3. **Cross-Scale Attention Router**：使用可学习的特征级门控机制对多分支图表示进行自适应融合。
4. **多模态信息融合**：联合使用分子图、药物嵌入和细胞系转录组特征。
5. **Masked graph processing**：分子图零填充位置不参与 GraphNorm 和图级平均池化。
6. **受控消融与统计分析**：使用相同 outer-test folds 对不同模型和消融变体进行配对比较。
7. **跨数据集验证**：在额外独立数据来源上评估模型的外部泛化表现。

---

## 可重复性

本项目提供：

- 模型训练代码；
- 数据处理代码；
- 五折实验脚本；
- baseline comparison scripts；
- controlled ablation scripts；
- statistical analysis scripts；
- computational-resource measurement scripts；
- 对应实验结果文件。

建议正式复现实验时保持相同的数据划分、随机种子、训练参数和模型选择方法。

---

## 致谢

本项目基于 PyTorch、PyTorch Geometric、RDKit、scikit-learn 及相关开源科学计算工具开发。

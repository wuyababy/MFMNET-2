## import module
from loader import *
from model import MDSyn
from enhanced_model import EnhancedMDSyn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import random
import numpy as np
import pandas as pd
# 使用numpy的interp函数替代scipy的interp
from numpy import interp
import matplotlib.pylab as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    balanced_accuracy_score, f1_score


def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write(','.join(map(str, AUCs)) + '\n')


# 设置随机种子以确保结果可重现
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, device, drug1_loader_train, drug2_loader_train, linc, optimizer, scheduler=None, epoch=1):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    total_loss = 0
    LOG_INTERVAL = 100

    # 定义损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        lincs = linc.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)

        optimizer.zero_grad()
        output, weight = model(data1, data2, lincs)
        loss = loss_fn(output, y)

        # 添加 L2 正则化
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)

        loss += 1e-5 * l2_reg  # 正则化系数

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))

    avg_loss = total_loss / len(drug1_loader_train)
    print(f'Average training loss: {avg_loss:.6f}')

    # 如果使用学习率调度器，根据平均损失更新学习率
    if scheduler is not None:
        scheduler.step(avg_loss)

    return avg_loss


def predicting(model, device, drug1_loader_test, drug2_loader_test, linc):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            lincs = linc.to(device)
            output, weight = model(data1, data2, lincs)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten(), weight


def ensemble_predict(models, device, drug1_loader_test, drug2_loader_test, linc):
    """集成多个模型的预测结果"""
    all_preds = []
    all_prelabels = []
    total_labels = None

    for model in models:
        model.eval()
        total_preds = torch.Tensor()
        total_prelabels = torch.Tensor()
        labels = torch.Tensor()

        with torch.no_grad():
            for data in zip(drug1_loader_test, drug2_loader_test):
                data1 = data[0]
                data2 = data[1]
                data1 = data1.to(device)
                data2 = data2.to(device)
                lincs = linc.to(device)
                output, _ = model(data1, data2, lincs)
                ys = F.softmax(output, 1).to('cpu').data.numpy()
                predicted_labels = list(map(lambda x: np.argmax(x), ys))
                predicted_scores = list(map(lambda x: x[1], ys))
                total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
                total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
                labels = torch.cat((labels, data1.y.view(-1, 1).cpu()), 0)

        all_preds.append(total_preds.numpy().flatten())
        all_prelabels.append(total_prelabels.numpy().flatten())

        if total_labels is None:
            total_labels = labels.numpy().flatten()

    # 加权融合预测分数
    weights = [1.0 / len(models)] * len(models)  # 平均权重
    ensemble_preds = np.zeros_like(all_preds[0])
    for i, preds in enumerate(all_preds):
        ensemble_preds += weights[i] * preds

    # 根据融合的预测分数生成标签
    ensemble_prelabels = (ensemble_preds > 0.5).astype(int)

    return total_labels, ensemble_preds, ensemble_prelabels, None


def train_model(model_class, device, train_nums, test_nums, drug1_data, drug2_data, linc,
                batch_size=64, lr=5e-4, num_epochs=300, model_params=None):
    """训练单个模型并返回"""
    # 准备数据加载器
    drug1_data_train = drug1_data[train_nums]
    drug1_data_test = drug1_data[test_nums]
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=batch_size, shuffle=None)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=batch_size, shuffle=None)

    drug2_data_test = drug2_data[test_nums]
    drug2_data_train = drug2_data[train_nums]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=batch_size, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=batch_size, shuffle=None)

    # 创建模型
    if model_params is None:
        model = model_class().to(device)
    else:
        model = model_class(**model_params).to(device)

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_auc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        train(model, device, drug1_loader_train, drug2_loader_train, linc, optimizer, scheduler, epoch)
        T, S, Y, _ = predicting(model, device, drug1_loader_test, drug2_loader_test, linc)

        # 计算指标
        AUC = roc_auc_score(T, S)

        if AUC > best_auc:
            best_auc = AUC
            best_epoch = epoch
            # 保存最佳模型
            torch.save(model.state_dict(), f'./result/models/{model_class.__name__}_best.pt')

    print(f'Best AUC: {best_auc:.4f} at epoch {best_epoch}')

    # 加载最佳模型
    model.load_state_dict(torch.load(f'./result/models/{model_class.__name__}_best.pt'))

    return model


def train_ensemble(model_classes, device, train_nums, test_nums, drug1_data, drug2_data, linc,
                   batch_size=64, lr=5e-4, num_epochs=300, model_params_list=None):
    """训练多个模型并返回集成"""
    models = []

    for i, model_class in enumerate(model_classes):
        print(f"\nTraining model {i + 1}/{len(model_classes)}: {model_class.__name__}")

        model_params = None if model_params_list is None else model_params_list[i]
        model = train_model(
            model_class, device, train_nums, test_nums, drug1_data, drug2_data, linc,
            batch_size, lr, num_epochs, model_params
        )
        models.append(model)

    return models


def evaluate_ensemble(models, device, test_nums, drug1_data, drug2_data, linc, batch_size=64, fold_idx=0):
    """评估集成模型的性能"""
    # 准备数据加载器
    drug1_data_test = drug1_data[test_nums]
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=batch_size, shuffle=None)

    drug2_data_test = drug2_data[test_nums]
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=batch_size, shuffle=None)

    # 集成预测
    T, S, Y, _ = ensemble_predict(models, device, drug1_loader_test, drug2_loader_test, linc)

    # 计算指标
    AUC = roc_auc_score(T, S)
    precision, recall, threshold = metrics.precision_recall_curve(T, S)
    PR_AUC = metrics.auc(recall, precision)
    BACC = balanced_accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    TPR = tp / (tp + fn)
    FPR = fp / (fp + tn)
    PREC = precision_score(T, Y)
    ACC = accuracy_score(T, Y)
    KAPPA = cohen_kappa_score(T, Y)
    recall = recall_score(T, Y)
    precision = precision_score(T, Y)
    F1 = f1_score(T, Y)

    # 保存指标
    metrics_file = f'./result/metric/Enhanced_MD-Syn_fold{fold_idx}.csv'
    AUCs = ['Ensemble', AUC, PR_AUC, ACC, BACC, precision, TPR, KAPPA, recall, F1]
    save_AUCs(AUCs, metrics_file)

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(T, S)
    roc_auc = auc(fpr, tpr)

    print(f"\nEnsemble Model Evaluation (Fold {fold_idx}):")
    print(f"AUC: {AUC:.4f}")
    print(f"PR-AUC: {PR_AUC:.4f}")
    print(f"Accuracy: {ACC:.4f}")
    print(f"Balanced Accuracy: {BACC:.4f}")
    print(f"F1 Score: {F1:.4f}")

    return fpr, tpr, roc_auc


# 主程序
if __name__ == "__main__":
    # 配置参数
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    LR = 5e-4
    LOG_INTERVAL = 20
    NUM_EPOCHS = 300

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    # 检查是否有可用的GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建结果目录
    import os

    os.makedirs('./result/models', exist_ok=True)
    os.makedirs('./result/metric', exist_ok=True)
    os.makedirs('./result/figure', exist_ok=True)

    # 加载数据
    drug1_data = DrugcombDataset(root='./data', dataset='ONeil_Drug1')
    drug2_data = DrugcombDataset(root='./data', dataset='ONeil_Drug2')
    row_lincs = pd.read_csv("data/raw/gene_vector.csv", index_col=0, header=None)
    lincs_array = row_lincs.to_numpy()
    linc = torch.tensor(lincs_array, dtype=torch.float32)

    # 数据集大小
    lenth = len(drug1_data)
    pot = int(lenth / 5)
    print('Dataset size:', lenth)
    print('Fold size:', pot)

    # 准备交叉验证
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    num_folds = 5
    random_num = random.sample(range(0, lenth), lenth)

    # 定义模型类和参数
    model_classes = [
        MDSyn,  # 原始模型
        EnhancedMDSyn  # 增强模型
    ]

    model_params_list = [
        None,  # 原始模型使用默认参数
        {  # 增强模型使用自定义参数
            'molecule_channels': 78,
            'hidden_channels': 128,
            'middle_channels': 64,
            'dropout_rate': 0.3,
            'n_heads': 4,
            'transformer_layers': 2
        }
    ]

    # 是否使用小样本测试模式
    small_sample_test = False  # 设置为True以使用小样本进行测试
    sample_size = 100 if small_sample_test else None  # 小样本大小

    # 快速测试模式（极小样本和极少轮数）
    quick_test_mode = False  # 设置为True启用快速测试
    if quick_test_mode:
        sample_size = 50  # 极小样本
        NUM_EPOCHS = 2  # 极少轮数
        print("QUICK TEST MODE ENABLED: Using 50 samples and 2 epochs only")

    # 进行交叉验证
    for i in range(num_folds):
        print(f"\n===== Fold {i + 1}/{num_folds} =====")

        # 划分训练集和测试集
        test_num = random_num[pot * i:pot * (i + 1)]
        train_num = random_num[:pot * i] + random_num[pot * (i + 1):]

        # 如果使用小样本测试，随机选择一部分数据
        if small_sample_test or quick_test_mode:
            print(f"Using small sample test mode with {sample_size} samples")
            indices = torch.randperm(len(train_num))[:sample_size]
            train_num = [train_num[i] for i in indices]
            indices = torch.randperm(len(test_num))[:sample_size]
            test_num = [test_num[i] for i in indices]

        # 训练集成模型
        models = train_ensemble(
            model_classes, device, train_num, test_num, drug1_data, drug2_data, linc,
            TRAIN_BATCH_SIZE, LR, NUM_EPOCHS, model_params_list
        )

        # 评估集成模型
        fpr, tpr, roc_auc = evaluate_ensemble(
            models, device, test_num, drug1_data, drug2_data, linc,
            TEST_BATCH_SIZE, i
        )

        # 保存ROC曲线数据
        tprs.append(interp(mean_fpr, fpr, tpr))  # 使用numpy的interp函数
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, label=f'Fold {i + 1} (AUC = {roc_auc:.3f})')

        # 如果是小样本测试模式，只运行一折
        if small_sample_test or quick_test_mode:
            print("Small sample test completed for first fold. Exiting...")
            break

    # 绘制ROC曲线
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.3f )' % (mean_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Enhanced MD-Syn: 5-Fold Cross-Validation')
    plt.legend(loc="lower right")
    plt.savefig("./result/figure/Enhanced_MD-Syn_ROC.png", dpi=600)
    plt.show()

    print(f"\nCross-validation complete. Mean AUC: {mean_auc:.4f}")

# -*- coding: utf-8 -*- 
# @Time    : 2025/6/28 16:08
# @Author  : yigeoooo
# @FileName: enhanced_trainer.py
# @Software: PyCharm
"""
增强训练器 - 专门解决两类分类问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from datetime import datetime

from trainer import HandGestureTrainer
from data_preprocessor import HandGesturePreprocessor


def enhanced_training():
    """增强的训练流程，专门针对两类分类问题"""
    print("🚀 增强训练器 - 专门解决两类分类问题")
    print("=" * 60)

    # 1. 检查数据
    data_dir = "data/processed"
    processed_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if not processed_files:
        print("❌ 找不到预处理数据文件")
        return

    latest_file = sorted(processed_files)[-1]
    data_path = os.path.join(data_dir, latest_file)
    print(f"✅ 使用数据文件: {latest_file}")

    # 2. 加载数据
    preprocessor = HandGesturePreprocessor()
    data_splits = preprocessor.load_processed_data(data_path)

    # 3. 分析数据质量
    X_train = data_splits['X_train']
    y_train = data_splits['y_gesture_train']

    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print(f"\n📊 数据分析:")
    print(f"   类别数: {len(unique_classes)}")
    print(f"   训练样本: {len(X_train)}")

    for cls, count in zip(unique_classes, class_counts):
        gesture_name = preprocessor.label_decoder['gesture'].get(cls, f'未知_{cls}')
        print(f"   类别{cls}({gesture_name}): {count} 样本")

    # 计算类别权重来处理不平衡
    class_weights = len(y_train) / (len(unique_classes) * class_counts)
    weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}
    print(f"\n⚖️ 计算类别权重: {weight_dict}")

    # 4. 创建多个训练实验
    experiments = [
        {
            'name': 'LSTM_高学习率',
            'model_type': 'lstm',
            'model_params': {'hidden_dim': 256, 'num_layers': 3, 'dropout': 0.5},
            'lr': 0.01,
            'epochs': 150,
            'scheduler': 'step'
        },
        {
            'name': 'CNN_LSTM_混合',
            'model_type': 'cnn_lstm',
            'model_params': {'cnn_hidden_dims': [256, 512], 'lstm_hidden_dim': 256, 'dropout': 0.4},
            'lr': 0.005,
            'epochs': 150,
            'scheduler': 'cosine'
        },
        {
            'name': 'Transformer_注意力',
            'model_type': 'transformer',
            'model_params': {'d_model': 512, 'nhead': 8, 'num_layers': 4},
            'lr': 0.001,
            'epochs': 200,
            'scheduler': 'plateau'
        }
    ]

    best_result = {'accuracy': 0, 'model_name': '', 'model_path': ''}

    for i, exp in enumerate(experiments):
        print(f"\n{'=' * 20} 实验 {i + 1}: {exp['name']} {'=' * 20}")

        try:
            # 创建训练器
            trainer = HandGestureTrainer(model_type=exp['model_type'])
            trainer.prepare_data(data_splits, batch_size=8)  # 小批次适合小数据集

            # 构建模型
            trainer.build_model(**exp['model_params'])

            # 设置带类别权重的损失函数
            if hasattr(trainer, 'criterion') and hasattr(trainer.criterion, 'weight'):
                weights = torch.FloatTensor([weight_dict.get(i, 1.0) for i in range(len(unique_classes))])
                trainer.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

            # 设置训练参数
            trainer.setup_training(
                learning_rate=exp['lr'],
                optimizer_type="adam",
                scheduler_type=exp['scheduler'],
                weight_decay=1e-3,
                use_early_stopping=True,
                patience=30
            )

            # 训练
            print(f"开始训练 {exp['name']}...")
            trainer.train(epochs=exp['epochs'], verbose=True)

            # 评估
            results = trainer.evaluate()
            accuracy = results['accuracy']

            print(f"\n✅ {exp['name']} 完成:")
            print(f"   最终准确率: {accuracy:.4f}")
            print(f"   平均损失: {results['avg_loss']:.4f}")

            # 保存模型
            model_path = f"data/models/{exp['name']}_final.pth"
            trainer.save_checkpoint(exp['epochs'], is_best=True)

            # 记录最佳结果
            if accuracy > best_result['accuracy']:
                best_result = {
                    'accuracy': accuracy,
                    'model_name': exp['name'],
                    'model_path': model_path
                }

            # 分析训练质量
            analyze_training_quality(trainer, exp['name'])

        except Exception as e:
            print(f"❌ {exp['name']} 训练失败: {e}")
            continue

    # 显示最终结果
    print(f"\n🏆 最佳模型结果:")
    print(f"   模型: {best_result['model_name']}")
    print(f"   准确率: {best_result['accuracy']:.4f}")

    if best_result['accuracy'] > 0.8:
        print(f"✅ 训练成功! 模型准确率超过80%")
    elif best_result['accuracy'] > 0.6:
        print(f"⚠️ 训练一般，准确率{best_result['accuracy']:.1%}，可能需要更多数据")
    else:
        print(f"❌ 训练效果差，准确率仅{best_result['accuracy']:.1%}")
        print(f"💡 建议:")
        print(f"   1. 检查手势是否足够不同")
        print(f"   2. 重新收集更多高质量数据")
        print(f"   3. 确保手势动作标准化")


def analyze_training_quality(trainer, model_name):
    """分析训练质量"""
    history = trainer.metrics_tracker.history

    if not history.get('train_acc') or not history.get('val_acc'):
        return

    final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
    final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0

    print(f"\n📈 {model_name} 训练质量分析:")
    print(f"   最终训练准确率: {final_train_acc:.4f}")
    print(f"   最终验证准确率: {final_val_acc:.4f}")

    # 过拟合检查
    overfitting = final_train_acc - final_val_acc
    if overfitting > 0.2:
        print(f"   ⚠️ 严重过拟合 (差距: {overfitting:.3f})")
    elif overfitting > 0.1:
        print(f"   ⚠️ 轻微过拟合 (差距: {overfitting:.3f})")
    else:
        print(f"   ✅ 泛化良好 (差距: {overfitting:.3f})")

    # 收敛性检查
    if len(history['train_acc']) > 20:
        recent_improvement = history['train_acc'][-1] - history['train_acc'][-10]
        if abs(recent_improvement) < 0.01:
            print(f"   ✅ 已收敛")
        else:
            print(f"   🔄 仍在改善")

    # 学习率检查
    if history.get('learning_rate'):
        final_lr = history['learning_rate'][-1]
        print(f"   最终学习率: {final_lr:.6f}")


def quick_fix_training():
    """快速修复训练 - 针对当前问题的简化版本"""
    print("🔧 快速修复训练")
    print("=" * 40)

    # 1. 加载数据
    data_dir = "data/processed"
    processed_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    latest_file = sorted(processed_files)[-1]
    data_path = os.path.join(data_dir, latest_file)

    preprocessor = HandGesturePreprocessor()
    data_splits = preprocessor.load_processed_data(data_path)

    # 2. 简单但有效的训练
    trainer = HandGestureTrainer(model_type="lstm")
    trainer.prepare_data(data_splits, batch_size=4)  # 很小的批次

    # 3. 小模型，高学习率
    trainer.build_model(
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    )

    # 4. 激进的训练设置
    trainer.optimizer = optim.SGD(trainer.model.parameters(), lr=0.1, momentum=0.9)
    trainer.criterion = nn.CrossEntropyLoss()
    trainer.scheduler = optim.lr_scheduler.StepLR(trainer.optimizer, step_size=50, gamma=0.1)

    print("开始激进训练...")

    # 5. 训练更多轮次
    for epoch in range(300):
        trainer.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(trainer.train_loader):
            inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)

            trainer.optimizer.zero_grad()
            outputs = trainer.model(inputs)
            loss = trainer.criterion(outputs, labels)
            loss.backward()
            trainer.optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        if trainer.scheduler:
            trainer.scheduler.step()

        accuracy = correct / total

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss={total_loss / len(trainer.train_loader):.4f}, Acc={accuracy:.4f}")

        # 如果准确率达到95%以上就停止
        if accuracy > 0.95:
            print(f"✅ 在第{epoch}轮达到95%准确率，停止训练")
            break

    # 保存模型
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_type': 'lstm'
    }, 'data/models/quick_fix_model.pth')

    # 评估
    results = trainer.evaluate()
    print(f"\n最终结果: 准确率 {results['accuracy']:.4f}")

    return results['accuracy']


def main():
    """主函数"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        accuracy = quick_fix_training()
        if accuracy > 0.8:
            print("✅ 快速修复成功!")
        else:
            print("❌ 快速修复失败，尝试完整增强训练")
            enhanced_training()
    else:
        enhanced_training()


if __name__ == "__main__":
    main()
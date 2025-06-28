import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from model_definition import ModelFactory, count_parameters


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class MetricsTracker:
    """指标跟踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }

    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def get_best_epoch(self, metric: str = 'val_acc', mode: str = 'max') -> int:
        if metric not in self.history or not self.history[metric]:
            return 0

        if mode == 'max':
            return np.argmax(self.history[metric])
        else:
            return np.argmin(self.history[metric])

    def plot_history(self, save_path: str = None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss', color='blue')
        if self.history['val_loss']:
            axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 准确率曲线
        axes[0, 1].plot(self.history['train_acc'], label='Training Accuracy', color='blue')
        if self.history['val_acc']:
            axes[0, 1].plot(self.history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 学习率曲线
        if self.history['learning_rate']:
            axes[1, 0].plot(self.history['learning_rate'], label='Learning Rate', color='green')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            axes[1, 0].set_yscale('log')

        # 验证损失vs训练损失
        if self.history['val_loss']:
            axes[1, 1].scatter(self.history['train_loss'], self.history['val_loss'], alpha=0.6)
            axes[1, 1].plot([0, max(self.history['train_loss'])], [0, max(self.history['train_loss'])], 'r--')
            axes[1, 1].set_title('Validation vs Training Loss')
            axes[1, 1].set_xlabel('Training Loss')
            axes[1, 1].set_ylabel('Validation Loss')
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存: {save_path}")

        plt.show()


class HandGestureTrainer:
    """手语识别训练器"""

    def __init__(self, model_type: str = "lstm", device: str = None, save_dir: str = "data/models"):
        self.model_type = model_type
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir

        # 创建保存目录（确保路径存在）
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"模型保存目录: {os.path.abspath(self.save_dir)}")

        # 模型和优化器
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # 训练状态
        self.is_multitask = False
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = None

        # 数据
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        print(f"训练器初始化完成，使用设备: {self.device}")

    def prepare_data(self, data_splits: Dict, batch_size: int = 32, shuffle: bool = True):
        """准备数据加载器"""
        print("准备数据加载器...")

        # 训练数据
        train_dataset = TensorDataset(
            torch.FloatTensor(data_splits['X_train']),
            torch.LongTensor(data_splits['y_gesture_train'])
        )
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        # 验证数据
        if 'X_val' in data_splits:
            val_dataset = TensorDataset(
                torch.FloatTensor(data_splits['X_val']),
                torch.LongTensor(data_splits['y_gesture_val'])
            )
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 测试数据
        test_dataset = TensorDataset(
            torch.FloatTensor(data_splits['X_test']),
            torch.LongTensor(data_splits['y_gesture_test'])
        )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 获取数据维度信息
        sample_batch = next(iter(self.train_loader))
        self.input_dim = sample_batch[0].shape[-1]
        self.sequence_length = sample_batch[0].shape[1]
        self.num_classes = len(torch.unique(sample_batch[1]))

        print(f"数据加载器准备完成:")
        print(f"  训练集: {len(self.train_loader.dataset)} 样本")
        if self.val_loader:
            print(f"  验证集: {len(self.val_loader.dataset)} 样本")
        print(f"  测试集: {len(self.test_loader.dataset)} 样本")
        print(f"  输入维度: {self.input_dim}")
        print(f"  序列长度: {self.sequence_length}")
        print(f"  类别数: {self.num_classes}")

    def prepare_multitask_data(self, data_splits: Dict, batch_size: int = 32, shuffle: bool = True):
        """准备多任务数据加载器"""
        print("准备多任务数据加载器...")

        self.is_multitask = True

        # 训练数据
        train_dataset = TensorDataset(
            torch.FloatTensor(data_splits['X_train']),
            torch.LongTensor(data_splits['y_gesture_train']),
            torch.LongTensor(data_splits['y_chinese_train']),
            torch.LongTensor(data_splits['y_english_train'])
        )
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        # 验证数据
        if 'X_val' in data_splits:
            val_dataset = TensorDataset(
                torch.FloatTensor(data_splits['X_val']),
                torch.LongTensor(data_splits['y_gesture_val']),
                torch.LongTensor(data_splits['y_chinese_val']),
                torch.LongTensor(data_splits['y_english_val'])
            )
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 测试数据
        test_dataset = TensorDataset(
            torch.FloatTensor(data_splits['X_test']),
            torch.LongTensor(data_splits['y_gesture_test']),
            torch.LongTensor(data_splits['y_chinese_test']),
            torch.LongTensor(data_splits['y_english_test'])
        )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 获取数据维度信息
        sample_batch = next(iter(self.train_loader))
        self.input_dim = sample_batch[0].shape[-1]
        self.sequence_length = sample_batch[0].shape[1]
        self.num_gesture_classes = len(torch.unique(sample_batch[1]))
        self.num_chinese_classes = len(torch.unique(sample_batch[2]))
        self.num_english_classes = len(torch.unique(sample_batch[3]))

        print(f"多任务数据加载器准备完成:")
        print(f"  手势类别数: {self.num_gesture_classes}")
        print(f"  中文类别数: {self.num_chinese_classes}")
        print(f"  英文类别数: {self.num_english_classes}")

    def build_model(self, **model_kwargs):
        """构建模型"""
        print(f"构建{self.model_type}模型...")

        if self.model_type == "multitask":
            if not self.is_multitask:
                raise ValueError("多任务模型需要使用prepare_multitask_data准备数据")

            self.model = ModelFactory.create_model(
                self.model_type,
                input_dim=self.input_dim,
                num_gesture_classes=self.num_gesture_classes,
                num_chinese_classes=self.num_chinese_classes,
                num_english_classes=self.num_english_classes,
                **model_kwargs
            )
        else:
            self.model = ModelFactory.create_model(
                self.model_type,
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                **model_kwargs
            )

        self.model.to(self.device)

        # 打印模型信息
        total_params = count_parameters(self.model)
        print(f"模型构建完成:")
        print(f"  模型类型: {self.model_type}")
        print(f"  总参数数: {total_params:,}")
        print(f"  模型大小: {total_params * 4 / (1024 * 1024):.2f} MB")

    def setup_training(self, learning_rate: float = 0.01, optimizer_type: str = "adam",
                       scheduler_type: str = "cosine", weight_decay: float = 1e-4,
                       use_early_stopping: bool = True, patience: int = 20):
        """设置训练参数"""
        print("设置训练参数...")

        # 损失函数 - 使用标签平滑减少过拟合
        if self.is_multitask:
            self.criterion = {
                'gesture': nn.CrossEntropyLoss(label_smoothing=0.1),
                'chinese': nn.CrossEntropyLoss(label_smoothing=0.1),
                'english': nn.CrossEntropyLoss(label_smoothing=0.1)
            }
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 优化器 - 使用更高的学习率
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                                       momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

        # 学习率调度器 - 使用更激进的调度
        if scheduler_type.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        elif scheduler_type.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        elif scheduler_type.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                  factor=0.5, patience=8, min_lr=1e-6)
        elif scheduler_type.lower() == "none":
            self.scheduler = None
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")

        # 早停 - 增加耐心值
        if use_early_stopping:
            self.early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

        print(f"训练设置完成:")
        print(f"  优化器: {optimizer_type}")
        print(f"  学习率: {learning_rate}")
        print(f"  调度器: {scheduler_type}")
        print(f"  早停耐心: {patience}")
        print(f"  标签平滑: 0.1")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        if self.is_multitask:
            task_losses = {'gesture': 0.0, 'chinese': 0.0, 'english': 0.0}
            task_correct = {'gesture': 0, 'chinese': 0, 'english': 0}

        progress_bar = tqdm(self.train_loader, desc="训练中")

        for batch_idx, batch in enumerate(progress_bar):
            if self.is_multitask:
                inputs, gesture_labels, chinese_labels, english_labels = batch
                inputs = inputs.to(self.device)
                gesture_labels = gesture_labels.to(self.device)
                chinese_labels = chinese_labels.to(self.device)
                english_labels = english_labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)

                # 计算损失
                losses = {
                    'gesture': self.criterion['gesture'](outputs['gesture'], gesture_labels),
                    'chinese': self.criterion['chinese'](outputs['chinese'], chinese_labels),
                    'english': self.criterion['english'](outputs['english'], english_labels)
                }

                # 总损失（可以加权）
                total_batch_loss = losses['gesture'] + losses['chinese'] + losses['english']

                # 计算准确率
                for task in ['gesture', 'chinese', 'english']:
                    pred = torch.argmax(outputs[task], dim=1)
                    if task == 'gesture':
                        correct = (pred == gesture_labels).sum().item()
                    elif task == 'chinese':
                        correct = (pred == chinese_labels).sum().item()
                    else:
                        correct = (pred == english_labels).sum().item()

                    task_correct[task] += correct
                    task_losses[task] += losses[task].item()

                correct_predictions += task_correct['gesture']  # 以手势任务为主要指标

            else:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                total_batch_loss = self.criterion(outputs, labels)

                # 计算准确率
                pred = torch.argmax(outputs, dim=1)
                correct_predictions += (pred == labels).sum().item()

            # 反向传播
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            total_samples += inputs.size(0)

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Acc': f'{correct_predictions / total_samples:.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct_predictions / total_samples

        metrics = {'train_loss': avg_loss, 'train_acc': avg_acc}

        if self.is_multitask:
            for task in ['gesture', 'chinese', 'english']:
                metrics[f'train_{task}_loss'] = task_losses[task] / len(self.train_loader)
                metrics[f'train_{task}_acc'] = task_correct[task] / total_samples

        return metrics

    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        if not self.val_loader:
            return {}

        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        if self.is_multitask:
            task_losses = {'gesture': 0.0, 'chinese': 0.0, 'english': 0.0}
            task_correct = {'gesture': 0, 'chinese': 0, 'english': 0}

        with torch.no_grad():
            for batch in self.val_loader:
                if self.is_multitask:
                    inputs, gesture_labels, chinese_labels, english_labels = batch
                    inputs = inputs.to(self.device)
                    gesture_labels = gesture_labels.to(self.device)
                    chinese_labels = chinese_labels.to(self.device)
                    english_labels = english_labels.to(self.device)

                    outputs = self.model(inputs)

                    losses = {
                        'gesture': self.criterion['gesture'](outputs['gesture'], gesture_labels),
                        'chinese': self.criterion['chinese'](outputs['chinese'], chinese_labels),
                        'english': self.criterion['english'](outputs['english'], english_labels)
                    }

                    total_batch_loss = losses['gesture'] + losses['chinese'] + losses['english']

                    for task in ['gesture', 'chinese', 'english']:
                        pred = torch.argmax(outputs[task], dim=1)
                        if task == 'gesture':
                            correct = (pred == gesture_labels).sum().item()
                        elif task == 'chinese':
                            correct = (pred == chinese_labels).sum().item()
                        else:
                            correct = (pred == english_labels).sum().item()

                        task_correct[task] += correct
                        task_losses[task] += losses[task].item()

                    correct_predictions += task_correct['gesture']

                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    total_batch_loss = self.criterion(outputs, labels)

                    pred = torch.argmax(outputs, dim=1)
                    correct_predictions += (pred == labels).sum().item()

                total_loss += total_batch_loss.item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct_predictions / total_samples

        metrics = {'val_loss': avg_loss, 'val_acc': avg_acc}

        if self.is_multitask:
            for task in ['gesture', 'chinese', 'english']:
                metrics[f'val_{task}_loss'] = task_losses[task] / len(self.val_loader)
                metrics[f'val_{task}_acc'] = task_correct[task] / total_samples

        return metrics

    def train(self, epochs: int = 100, verbose: bool = True):
        """训练模型"""
        print(f"开始训练，共{epochs}个epoch")
        print(f"模型保存目录: {os.path.abspath(self.save_dir)}")
        print("=" * 50)

        start_time = time.time()
        best_val_acc = 0.0
        best_train_acc = 0.0  # 如果没有验证集，使用训练准确率

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate_epoch()

            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step(train_metrics['train_loss'])
                else:
                    self.scheduler.step()

            # 记录指标
            current_lr = self.optimizer.param_groups[0]['lr']
            all_metrics = {**train_metrics, **val_metrics, 'learning_rate': current_lr}
            self.metrics_tracker.update(all_metrics)

            # 决定是否保存模型
            should_save = False
            current_acc = 0.0

            if val_metrics and 'val_acc' in val_metrics:
                current_acc = val_metrics['val_acc']
                if current_acc > best_val_acc:
                    best_val_acc = current_acc
                    should_save = True
            else:
                # 没有验证集时使用训练准确率
                current_acc = train_metrics['train_acc']
                if current_acc > best_train_acc:
                    best_train_acc = current_acc
                    should_save = True

            # 保存检查点（每10个epoch或最佳模型）
            if should_save:
                self.save_checkpoint(epoch, is_best=True)
            elif epoch % 10 == 0 or epoch == epochs - 1:
                self.save_checkpoint(epoch, is_best=False)

            # 早停检查
            if self.early_stopping and val_metrics:
                if self.early_stopping(val_metrics['val_loss'], self.model):
                    print(f"\n早停触发，在第{epoch + 1}个epoch停止训练")
                    break

            # 打印进度
            if verbose:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch [{epoch + 1}/{epochs}] - {epoch_time:.2f}s")
                print(f"  训练: Loss={train_metrics['train_loss']:.4f}, Acc={train_metrics['train_acc']:.4f}")
                if val_metrics:
                    print(f"  验证: Loss={val_metrics['val_loss']:.4f}, Acc={val_metrics['val_acc']:.4f}")
                print(f"  学习率: {current_lr:.6f}")
                if should_save:
                    print(f"  💾 保存最佳模型 (准确率: {current_acc:.4f})")

                # 检查训练质量
                if epoch > 10:
                    if train_metrics['train_acc'] > 0.95 and (not val_metrics or val_metrics.get('val_acc', 0) < 0.7):
                        print(
                            f"  ⚠️ 可能过拟合：训练准确率{train_metrics['train_acc']:.3f}，验证准确率{val_metrics.get('val_acc', 0):.3f}")
                    elif train_metrics['train_acc'] < 0.6 and epoch > 50:
                        print(f"  ⚠️ 训练困难：准确率仍然很低，考虑调整学习率或数据质量")

                print("-" * 30)

        # 训练结束后保存最终模型
        final_save_path = os.path.join(self.save_dir, f"final_{self.model_type}_model.pth")
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'final_epoch': epochs,
                'best_acc': max(best_val_acc, best_train_acc),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }, final_save_path)
            print(f"✅ 最终模型已保存: {os.path.abspath(final_save_path)}")
        except Exception as e:
            print(f"❌ 最终模型保存失败: {e}")

        total_time = time.time() - start_time
        final_best = max(best_val_acc, best_train_acc)
        print(f"\n🎉 训练完成!")
        print(f"   总用时: {total_time:.2f}秒")
        print(f"   最佳准确率: {final_best:.4f}")
        print(f"   模型文件保存在: {os.path.abspath(self.save_dir)}")

        # 列出保存的文件
        try:
            saved_files = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
            if saved_files:
                print(f"\n📁 已保存的模型文件:")
                for file in saved_files:
                    filepath = os.path.join(self.save_dir, file)
                    size = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"   - {file} ({size:.2f} MB)")
            else:
                print(f"⚠️  警告: 模型目录中没有找到保存的文件")
        except Exception as e:
            print(f"无法列出保存的文件: {e}")

    def evaluate(self, loader: DataLoader = None) -> Dict[str, Any]:
        """评估模型"""
        if loader is None:
            loader = self.test_loader

        self.model.eval()

        all_predictions = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(loader, desc="评估中"):
                if self.is_multitask:
                    inputs, gesture_labels, chinese_labels, english_labels = batch
                    inputs = inputs.to(self.device)
                    gesture_labels = gesture_labels.to(self.device)

                    outputs = self.model(inputs)
                    pred = torch.argmax(outputs['gesture'], dim=1)

                    all_predictions.extend(pred.cpu().numpy())
                    all_labels.extend(gesture_labels.cpu().numpy())

                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    pred = torch.argmax(outputs, dim=1)

                    all_predictions.extend(pred.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()

        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)

        # 分类报告
        class_report = classification_report(all_labels, all_predictions, output_dict=True)

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)

        results = {
            'accuracy': accuracy,
            'avg_loss': total_loss / len(loader),
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'true_labels': all_labels
        }

        print(f"评估结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  平均损失: {results['avg_loss']:.4f}")

        return results

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点 - 包含完整模型配置"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        # 构建模型配置
        model_config = {}

        if hasattr(self.model, '__dict__'):
            # 尝试提取模型参数
            if self.model_type == "cnn_lstm":
                if hasattr(self.model, 'cnn_extractor') and hasattr(self.model.cnn_extractor, 'cnn_layers'):
                    # 提取CNN通道配置
                    cnn_layers = self.model.cnn_extractor.cnn_layers
                    cnn_channels = []
                    for i, layer in enumerate(cnn_layers):
                        if isinstance(layer, nn.Conv1d):
                            cnn_channels.append(layer.out_channels)
                    model_config['cnn_channels'] = cnn_channels

                if hasattr(self.model, 'lstm'):
                    model_config['lstm_hidden_size'] = self.model.lstm.hidden_size
                    model_config['lstm_num_layers'] = self.model.lstm.num_layers
                    model_config['bidirectional'] = self.model.lstm.bidirectional

                if hasattr(self.model, 'classifier') and len(self.model.classifier) > 1:
                    if isinstance(self.model.classifier[1], nn.Linear):
                        model_config['classifier_hidden_size'] = self.model.classifier[1].out_features

            elif self.model_type == "lstm":
                if hasattr(self.model, 'lstm'):
                    model_config['hidden_dim'] = self.model.lstm.hidden_size
                    model_config['num_layers'] = self.model.lstm.num_layers
                    model_config['bidirectional'] = self.model.lstm.bidirectional

        checkpoint = {
            'epoch': epoch,
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'model_config': model_config,  # 重要：保存模型配置
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_history': self.metrics_tracker.history,
            'is_multitask': self.is_multitask,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'timestamp': timestamp
        }

        if is_best:
            filename = f"best_{self.model_type}_model.pth"
        else:
            filename = f"{self.model_type}_checkpoint_epoch_{epoch + 1}.pth"

        filepath = os.path.join(self.save_dir, filename)

        try:
            torch.save(checkpoint, filepath)
            print(f"✅ 模型已保存: {os.path.abspath(filepath)}")
            print(f"   保存的配置: {model_config}")

            # 验证文件是否真的存在
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"   文件大小: {file_size:.2f} MB")

        except Exception as e:
            print(f"❌ 模型保存失败: {e}")

            # 尝试在当前目录保存
            try:
                fallback_path = f"backup_{filename}"
                torch.save(checkpoint, fallback_path)
                print(f"✅ 备用保存成功: {os.path.abspath(fallback_path)}")
            except Exception as e2:
                print(f"❌ 备用保存也失败: {e2}")

    # ==================== 修复5: 快速修复脚本 ====================

    def quick_fix_model_loading():
        """快速修复现有模型的加载问题"""
        print("🔧 快速修复模型加载问题")
        print("=" * 40)

        models_dir = "data/models"
        if not os.path.exists(models_dir):
            print("❌ 模型目录不存在")
            return

        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not model_files:
            print("❌ 没有找到模型文件")
            return

        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            print(f"\n检查模型: {model_file}")

            try:
                # 加载模型检查点
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # 分析模型架构
                state_dict = checkpoint['model_state_dict']

                # 推断配置
                config = {}

                if 'cnn_extractor.cnn_layers.0.weight' in state_dict:
                    first_cnn = state_dict['cnn_extractor.cnn_layers.0.weight'].shape[0]
                    config['cnn_channels'] = [first_cnn]

                    if 'cnn_extractor.cnn_layers.5.weight' in state_dict:
                        second_cnn = state_dict['cnn_extractor.cnn_layers.5.weight'].shape[0]
                        config['cnn_channels'].append(second_cnn)

                if 'lstm.weight_ih_l0' in state_dict:
                    lstm_hidden = state_dict['lstm.weight_ih_l0'].shape[0] // 4
                    config['lstm_hidden_size'] = lstm_hidden
                    config['lstm_num_layers'] = 2 if 'lstm.weight_ih_l1' in state_dict else 1
                    config['bidirectional'] = 'lstm.weight_ih_l0_reverse' in state_dict

                if 'classifier.1.weight' in state_dict:
                    classifier_hidden = state_dict['classifier.1.weight'].shape[0]
                    config['classifier_hidden_size'] = classifier_hidden

                print(f"推断配置: {config}")

                # 更新模型检查点
                if config and 'model_config' not in checkpoint:
                    checkpoint['model_config'] = config

                    # 保存修复后的模型
                    fixed_path = model_path.replace('.pth', '_fixed.pth')
                    torch.save(checkpoint, fixed_path)
                    print(f"✅ 修复后模型已保存: {fixed_path}")
                else:
                    print("✅ 模型已包含配置信息")

            except Exception as e:
                print(f"❌ 处理失败: {e}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics_tracker.history = checkpoint['metrics_history']

        print(f"模型检查点已加载: {filepath}")
        return checkpoint['epoch']

    def plot_training_history(self, save_path: str = None):
        """绘制训练历史"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"training_history_{self.model_type}_{timestamp}.png")

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            self.metrics_tracker.plot_history(save_path)
        except Exception as e:
            print(f"❌ 训练历史图保存失败: {e}")
            # 尝试保存到当前目录
            try:
                fallback_path = f"training_history_{self.model_type}.png"
                self.metrics_tracker.plot_history(fallback_path)
                print(f"✅ 训练历史图备用保存: {os.path.abspath(fallback_path)}")
            except Exception as e2:
                print(f"❌ 备用保存也失败: {e2}")


def main():
    """主函数 - 完整的训练流程"""
    print("手语识别模型训练器")
    print("=" * 50)

    # 1. 检查预处理数据文件
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print("❌ 找不到预处理数据目录")
        print("请先运行 data_preprocessor.py 生成预处理数据")
        return

    # 查找最新的预处理数据文件
    processed_files = [f for f in os.listdir(data_dir) if f.startswith("processed_data_") and f.endswith(".pkl")]
    if not processed_files:
        print("❌ 找不到预处理数据文件")
        print("请先运行 data_preprocessor.py 生成预处理数据")
        return

    # 使用最新的数据文件
    latest_file = sorted(processed_files)[-1]
    data_path = os.path.join(data_dir, latest_file)
    print(f"✅ 找到预处理数据文件: {latest_file}")

    # 2. 加载预处理数据
    try:
        from data_preprocessor import HandGesturePreprocessor
        preprocessor = HandGesturePreprocessor()
        data_splits = preprocessor.load_processed_data(data_path)
        print(f"✅ 数据加载成功")

        # 显示数据统计
        print(f"\n📊 数据统计:")
        print(f"   训练集: {data_splits['X_train'].shape[0]} 样本")
        if 'X_val' in data_splits:
            print(f"   验证集: {data_splits['X_val'].shape[0]} 样本")
        print(f"   测试集: {data_splits['X_test'].shape[0]} 样本")
        print(f"   特征维度: {data_splits['X_train'].shape[1]} × {data_splits['X_train'].shape[2]}")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 3. 选择模型类型
    available_models = ["lstm", "gru", "transformer", "cnn_lstm", "attention_lstm", "resnet1d"]
    print(f"\n🤖 可用的模型类型:")
    for i, model in enumerate(available_models, 1):
        print(f"   {i}. {model}")

    # 这里可以自动选择或让用户选择
    model_type = "lstm"  # 默认使用LSTM，你可以修改这里
    print(f"🎯 选择模型类型: {model_type}")

    # 4. 创建训练器
    trainer = HandGestureTrainer(model_type=model_type)

    # 5. 准备数据
    print(f"\n📦 准备数据...")
    trainer.prepare_data(data_splits, batch_size=16, shuffle=True)

    # 6. 构建模型
    print(f"\n🏗️  构建模型...")
    if model_type == "lstm":
        trainer.build_model(
            hidden_dim=256,  # 增加模型容量
            num_layers=3,  # 增加层数
            dropout=0.5,  # 增加dropout防止过拟合
            bidirectional=True
        )
    elif model_type == "transformer":
        trainer.build_model(
            d_model=512,
            nhead=8,
            num_layers=6,
            dropout=0.2
        )
    elif model_type == "cnn_lstm":
        trainer.build_model(
            cnn_hidden_dims=[256, 512],
            lstm_hidden_dim=256,
            lstm_layers=3,
            dropout=0.5
        )
    else:
        trainer.build_model()  # 使用默认参数

    # 7. 设置训练参数
    print(f"\n⚙️  设置训练参数...")
    trainer.setup_training(
        learning_rate=0.01,  # 提高学习率
        optimizer_type="adam",
        scheduler_type="step",  # 使用阶梯式学习率
        weight_decay=1e-3,  # 增加正则化
        use_early_stopping=True,
        patience=25  # 增加耐心值
    )

    # 8. 开始训练
    print(f"\n🚀 开始训练...")
    epochs = 200  # 大幅增加训练轮数
    trainer.train(epochs=epochs, verbose=True)

    # 9. 评估模型
    print(f"\n📊 评估模型...")
    test_results = trainer.evaluate()

    print(f"\n✅ 训练完成!")
    print(f"📈 最终测试结果:")
    print(f"   准确率: {test_results['accuracy']:.4f}")
    print(f"   平均损失: {test_results['avg_loss']:.4f}")

    # 10. 绘制训练历史
    print(f"\n📊 生成训练历史图...")
    trainer.plot_training_history()

    # 11. 保存最终统计
    stats = {
        'model_type': model_type,
        'epochs': epochs,
        'final_accuracy': test_results['accuracy'],
        'final_loss': test_results['avg_loss'],
        'train_samples': len(data_splits['X_train']),
        'test_samples': len(data_splits['X_test']),
        'timestamp': datetime.now().isoformat()
    }

    # 确保保存目录存在
    os.makedirs(trainer.save_dir, exist_ok=True)
    stats_file = os.path.join(trainer.save_dir, f"training_stats_{model_type}.json")

    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ 训练统计已保存: {os.path.abspath(stats_file)}")
    except Exception as e:
        print(f"❌ 统计保存失败: {e}")
        # 尝试保存到当前目录
        try:
            fallback_stats = f"training_stats_{model_type}.json"
            with open(fallback_stats, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"✅ 统计备用保存: {os.path.abspath(fallback_stats)}")
        except Exception as e2:
            print(f"❌ 备用统计保存失败: {e2}")

    print(f"\n🎉 模型训练流程全部完成!")
    print(f"📁 训练结果保存目录: {os.path.abspath(trainer.save_dir)}")

    # 列出所有保存的文件
    try:
        print(f"\n📋 生成的文件列表:")
        all_files = []

        # 检查模型文件
        if os.path.exists(trainer.save_dir):
            model_files = [f for f in os.listdir(trainer.save_dir) if f.endswith(('.pth', '.png', '.json'))]
            for file in model_files:
                filepath = os.path.join(trainer.save_dir, file)
                size = os.path.getsize(filepath) / (1024 * 1024)
                print(f"   📄 {file} ({size:.2f} MB)")
                all_files.append(file)

        # 检查当前目录的备用文件
        current_files = [f for f in os.listdir('.') if
                         f.startswith(('backup_', 'training_', 'final_')) and f.endswith(('.pth', '.png', '.json'))]
        for file in current_files:
            if file not in all_files:
                size = os.path.getsize(file) / (1024 * 1024)
                print(f"   📄 {file} ({size:.2f} MB) [当前目录]")

        if not all_files and not current_files:
            print("   ⚠️  没有找到保存的文件")

    except Exception as e:
        print(f"无法列出文件: {e}")


def train_multiple_models():
    """训练多个模型进行对比"""
    print("多模型对比训练")
    print("=" * 50)

    # 加载数据
    data_dir = "data/processed"
    processed_files = [f for f in os.listdir(data_dir) if f.startswith("processed_data_") and f.endswith(".pkl")]
    if not processed_files:
        print("❌ 找不到预处理数据文件")
        return

    latest_file = sorted(processed_files)[-1]
    data_path = os.path.join(data_dir, latest_file)

    from data_preprocessor import HandGesturePreprocessor
    preprocessor = HandGesturePreprocessor()
    data_splits = preprocessor.load_processed_data(data_path)

    # 要训练的模型列表
    models_to_train = ["lstm", "gru", "cnn_lstm"]
    results = {}

    for model_type in models_to_train:
        print(f"\n{'=' * 20} 训练 {model_type.upper()} 模型 {'=' * 20}")

        try:
            # 创建训练器
            trainer = HandGestureTrainer(model_type=model_type)
            trainer.prepare_data(data_splits, batch_size=16)
            trainer.build_model()
            trainer.setup_training(learning_rate=0.001, use_early_stopping=True, patience=10)

            # 训练
            trainer.train(epochs=30, verbose=False)

            # 评估
            test_results = trainer.evaluate()
            results[model_type] = {
                'accuracy': test_results['accuracy'],
                'loss': test_results['avg_loss']
            }

            print(f"✅ {model_type} 完成 - 准确率: {test_results['accuracy']:.4f}")

        except Exception as e:
            print(f"❌ {model_type} 训练失败: {e}")
            results[model_type] = {'accuracy': 0.0, 'loss': float('inf')}

    # 显示对比结果
    print(f"\n📊 模型对比结果:")
    print("-" * 60)
    print(f"{'模型类型':<15} {'准确率':<10} {'损失':<10}")
    print("-" * 60)

    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])

    for model_type, metrics in results.items():
        marker = " 🏆" if model_type == best_model else ""
        print(f"{model_type:<15} {metrics['accuracy']:<10.4f} {metrics['loss']:<10.4f}{marker}")

    print(f"\n🏆 最佳模型: {best_model} (准确率: {results[best_model]['accuracy']:.4f})")


def quick_train():
    """快速训练（使用默认参数）"""
    print("快速训练模式")
    print("=" * 30)

    try:
        # 查找数据
        data_dir = "data/processed"
        processed_files = [f for f in os.listdir(data_dir) if f.startswith("processed_data_") and f.endswith(".pkl")]
        latest_file = sorted(processed_files)[-1]
        data_path = os.path.join(data_dir, latest_file)

        # 加载数据
        from data_preprocessor import HandGesturePreprocessor
        preprocessor = HandGesturePreprocessor()
        data_splits = preprocessor.load_processed_data(data_path)

        # 快速训练
        trainer = HandGestureTrainer(model_type="lstm")
        trainer.prepare_data(data_splits, batch_size=32)
        trainer.build_model(hidden_dim=64, num_layers=1)  # 简化模型
        trainer.setup_training(learning_rate=0.01, use_early_stopping=True, patience=5)
        trainer.train(epochs=20, verbose=True)

        # 评估
        results = trainer.evaluate()
        print(f"\n✅ 快速训练完成!")
        print(f"准确率: {results['accuracy']:.4f}")

    except Exception as e:
        print(f"❌ 快速训练失败: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "multi":
            train_multiple_models()
        elif sys.argv[1] == "quick":
            quick_train()
        else:
            print("用法:")
            print("  python trainer.py        # 完整训练")
            print("  python trainer.py multi  # 多模型对比")
            print("  python trainer.py quick  # 快速训练")
    else:
        main()
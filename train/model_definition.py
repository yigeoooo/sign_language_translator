# -*- coding: utf-8 -*- 
# @Time    : 2025/6/28 14:07
# @Author  : yigeoooo
# @FileName: model_definition.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from typing import Dict, List, Tuple, Optional
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码模块 - 用于Transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 256, 512]):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim

        self.cnn_layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.cnn_layers(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        return x


class LSTMModel(nn.Module):
    """LSTM模型 - 基础的循环神经网络模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 10, dropout: float = 0.3, bidirectional: bool = True,
                 # 添加兼容性参数
                 lstm_hidden_size: int = None, lstm_num_layers: int = None):
        super().__init__()

        # 参数兼容性处理
        if lstm_hidden_size is not None:
            hidden_dim = lstm_hidden_size
        if lstm_num_layers is not None:
            num_layers = lstm_num_layers

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # 输出维度
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)

        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)

        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 连接前向和后向的最后隐藏状态
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # 分类
        output = self.classifier(hidden)
        return output


class GRUModel(nn.Module):
    """GRU模型 - 门控循环单元模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 10, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # GRU层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # 输出维度
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)

        # GRU前向传播
        gru_out, hidden = self.gru(x)

        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 连接前向和后向的最后隐藏状态
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        # 分类
        output = self.classifier(hidden)
        return output


class TransformerModel(nn.Module):
    """Transformer模型 - 基于注意力机制的模型"""

    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # 分类层
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # 输入投影
        x = self.input_projection(x)

        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # Transformer编码
        transformer_out = self.transformer_encoder(x)

        # 全局平均池化
        pooled = torch.mean(transformer_out, dim=1)

        # 分类
        output = self.classifier(pooled)
        return output


class CNN_LSTM_Model(nn.Module):
    """CNN-LSTM混合模型 - 结合卷积和循环神经网络"""

    def __init__(self, input_dim: int, cnn_channels: List[int] = None,
                 lstm_hidden_size: int = 128, lstm_num_layers: int = 2,
                 num_classes: int = 10, dropout: float = 0.3,
                 bidirectional: bool = True, classifier_hidden_size: int = None):
        super().__init__()

        # 兼容旧的参数名称
        if cnn_channels is None:
            cnn_channels = [128, 256]

        if classifier_hidden_size is None:
            classifier_hidden_size = lstm_hidden_size

        # CNN特征提取器
        self.cnn_extractor = CNNFeatureExtractor(input_dim, cnn_channels)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.cnn_extractor.output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # 分类层
        lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_classes)
        )

    def forward(self, x):
        # CNN特征提取
        cnn_features = self.cnn_extractor(x)

        # LSTM处理
        lstm_out, (hidden, _) = self.lstm(cnn_features)

        # 使用最后隐藏状态
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # 分类
        output = self.classifier(hidden)
        return output


class AttentionLSTM(nn.Module):
    """带注意力机制的LSTM模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        lstm_output_dim = hidden_dim * 2  # 双向LSTM

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)

        # 计算注意力权重
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # 加权求和
        weighted_output = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_dim*2)

        # 分类
        output = self.classifier(weighted_output)
        return output


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.layers(x)


class ResNet1D(nn.Module):
    """1D ResNet模型 - 用于序列数据的残差网络"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_blocks: int = 4,
                 num_classes: int = 10, dropout: float = 0.1):
        super().__init__()

        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(num_blocks)
        ])

        # 全局平均池化和分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)

        # 残差块
        for block in self.residual_blocks:
            x = block(x)

        # 转换为1D卷积格式并池化
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        output = self.classifier(x)
        return output


class MultiTaskModel(nn.Module):
    """多任务学习模型 - 同时预测手势、中文含义、英文含义"""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_gesture_classes: int = 10, num_chinese_classes: int = 10,
                 num_english_classes: int = 10, backbone: str = "lstm"):
        super().__init__()

        # 选择骨干网络
        if backbone == "lstm":
            self.backbone = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                dropout=0.3,
                bidirectional=True,
                batch_first=True
            )
            feature_dim = hidden_dim * 2
        elif backbone == "transformer":
            self.backbone = TransformerModel(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=8,
                num_layers=4,
                num_classes=hidden_dim  # 作为特征提取器使用
            )
            feature_dim = hidden_dim
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")

        self.backbone_type = backbone

        # 共享特征层
        self.shared_features = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 任务特定的分类头
        self.gesture_classifier = nn.Linear(hidden_dim, num_gesture_classes)
        self.chinese_classifier = nn.Linear(hidden_dim, num_chinese_classes)
        self.english_classifier = nn.Linear(hidden_dim, num_english_classes)

    def forward(self, x):
        # 特征提取
        if self.backbone_type == "lstm":
            lstm_out, (hidden, _) = self.backbone(x)
            # 使用最后隐藏状态
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:  # transformer
            # 使用transformer的特征提取部分
            features = self.backbone.input_projection(x)
            features = features.transpose(0, 1)
            features = self.backbone.pos_encoder(features)
            features = features.transpose(0, 1)
            features = self.backbone.transformer_encoder(features)
            features = torch.mean(features, dim=1)

        # 共享特征
        shared_features = self.shared_features(features)

        # 多任务输出
        gesture_output = self.gesture_classifier(shared_features)
        chinese_output = self.chinese_classifier(shared_features)
        english_output = self.english_classifier(shared_features)

        return {
            "gesture": gesture_output,
            "chinese": chinese_output,
            "english": english_output
        }


class ModelFactory:
    """模型工厂 - 用于创建不同类型的模型"""

    @staticmethod
    def create_model(model_type: str, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
        """创建指定类型的模型"""

        if model_type == "lstm":
            return LSTMModel(input_dim, num_classes=num_classes, **kwargs)

        elif model_type == "gru":
            return GRUModel(input_dim, num_classes=num_classes, **kwargs)

        elif model_type == "transformer":
            return TransformerModel(input_dim, num_classes=num_classes, **kwargs)

        elif model_type == "cnn_lstm":
            return CNN_LSTM_Model(input_dim, num_classes=num_classes, **kwargs)

        elif model_type == "attention_lstm":
            return AttentionLSTM(input_dim, num_classes=num_classes, **kwargs)

        elif model_type == "resnet1d":
            return ResNet1D(input_dim, num_classes=num_classes, **kwargs)

        elif model_type == "multitask":
            return MultiTaskModel(input_dim, **kwargs)

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    @staticmethod
    def get_model_info(model_type: str) -> Dict:
        """获取模型信息"""
        model_info = {
            "lstm": {
                "name": "LSTM模型",
                "description": "基础的长短期记忆网络，适合序列数据处理",
                "complexity": "中等",
                "training_speed": "快",
                "accuracy": "中等"
            },
            "gru": {
                "name": "GRU模型",
                "description": "门控循环单元，比LSTM更简单但性能相近",
                "complexity": "中等",
                "training_speed": "快",
                "accuracy": "中等"
            },
            "transformer": {
                "name": "Transformer模型",
                "description": "基于注意力机制的模型，能够捕获长距离依赖",
                "complexity": "高",
                "training_speed": "慢",
                "accuracy": "高"
            },
            "cnn_lstm": {
                "name": "CNN-LSTM混合模型",
                "description": "结合卷积神经网络和LSTM的优势",
                "complexity": "中高",
                "training_speed": "中等",
                "accuracy": "中高"
            },
            "attention_lstm": {
                "name": "注意力LSTM模型",
                "description": "在LSTM基础上加入注意力机制",
                "complexity": "中高",
                "training_speed": "中等",
                "accuracy": "中高"
            },
            "resnet1d": {
                "name": "1D ResNet模型",
                "description": "适用于序列数据的残差网络",
                "complexity": "中高",
                "training_speed": "中等",
                "accuracy": "中高"
            },
            "multitask": {
                "name": "多任务学习模型",
                "description": "同时学习多个相关任务，提高泛化能力",
                "complexity": "高",
                "training_speed": "慢",
                "accuracy": "高"
            }
        }

        return model_info.get(model_type, {"name": "未知模型"})


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> str:
    """生成模型摘要"""
    total_params = count_parameters(model)

    # 计算模型大小（MB）
    param_size = total_params * 4 / (1024 * 1024)  # 假设每个参数4字节

    summary = f"""
模型摘要:
========================================
模型类型: {model.__class__.__name__}
输入形状: {input_shape}
总参数数: {total_params:,}
模型大小: {param_size:.2f} MB
========================================
"""
    return summary


def main():
    """主函数 - 演示模型创建"""
    print("手语识别模型定义")
    print("=" * 50)

    # 模型参数
    input_dim = 401  # 假设特征维度
    sequence_length = 30
    num_classes = 10
    batch_size = 32

    # 创建不同类型的模型
    models = {
        "LSTM": ModelFactory.create_model("lstm", input_dim, num_classes),
        "GRU": ModelFactory.create_model("gru", input_dim, num_classes),
        "Transformer": ModelFactory.create_model("transformer", input_dim, num_classes),
        "CNN-LSTM": ModelFactory.create_model("cnn_lstm", input_dim, num_classes),
        "Attention-LSTM": ModelFactory.create_model("attention_lstm", input_dim, num_classes),
        "ResNet1D": ModelFactory.create_model("resnet1d", input_dim, num_classes),
    }

    # 多任务模型
    multitask_model = ModelFactory.create_model(
        "multitask", input_dim,
        num_gesture_classes=10,
        num_chinese_classes=10,
        num_english_classes=10
    )

    # 测试输入
    test_input = torch.randn(batch_size, sequence_length, input_dim)

    print("模型测试结果:")
    print("-" * 50)

    # 测试单任务模型
    for name, model in models.items():
        try:
            output = model(test_input)
            print(f"{name:15s} | 输出形状: {output.shape} | 参数数: {count_parameters(model):,}")
        except Exception as e:
            print(f"{name:15s} | 错误: {e}")

    # 测试多任务模型
    try:
        multitask_output = multitask_model(test_input)
        print(
            f"{'Multi-task':15s} | 输出: {[f'{k}:{v.shape}' for k, v in multitask_output.items()]} | 参数数: {count_parameters(multitask_model):,}")
    except Exception as e:
        print(f"{'Multi-task':15s} | 错误: {e}")

    print("\n模型信息:")
    print("-" * 50)
    for model_type in ["lstm", "transformer", "multitask"]:
        info = ModelFactory.get_model_info(model_type)
        print(f"{info['name']}: {info['description']}")


if __name__ == "__main__":
    main()
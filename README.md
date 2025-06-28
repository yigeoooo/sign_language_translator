# 手语翻译系统

基于Leap Motion 2的手语识别与翻译系统，支持中英文双语翻译和语音播报。

## 项目简介

本项目是一个完整的手语识别训练框架，从数据收集、预处理、模型训练到实时推理的端到端解决方案。系统能够：

- **数据收集**: 使用Leap Motion设备收集手语数据
- **数据预处理**: 特征提取、数据清洗、标准化
- **模型训练**: 支持多种深度学习模型架构
- **实时推理**: 实时手语识别与中英文翻译

---

##  环境配置


- **Python版本：  3.8（必须）**

- **leapmotion2环境配置：**  
  * 参考官方Python重构配置环境，由于leapmotion的SDK是基于C语言进行二次开发，所以用Python时需要特殊配置环境。
  * 具体参考Github官方项目。[leapc-python-bindings](https://github.com/ultraleap/leapc-python-bindings)。
  * 当根据流程配置好环境，运行项目中visualiser.py，看到如下效果，则证明环境配置成功。（前提是拥有leapmotion2设备，并且已经连接）</br>
  ![图片](/tests/video.gif "leapmotion2测试")

## 项目目录结构

```
tests/                           # 测试类文档
train/
├── data_collector.py           # 数据收集器
├── data_preprocessor.py        # 数据预处理器
├── model_definition.py         # 模型定义
├── enhanced_trainer.py         # 增强训练器
├── trainer.py                  # 训练器
├── inference.py                # 推理器
└── data/                       # 数据目录
    ├── raw/                    # 原始数据文件
    │   └── ...
    ├── annotations/            # 标注文件
    │   └── ...
    ├── processed/              # 处理后的数据
    │   └── ...
    ├── models/                 # 训练好的模型
    │   └── ...
    └── gesture_labels.json     # 手势标签配置
```

## 快速开始

### 1. 数据收集

首先收集手语数据：

```bash
cd train
python data_collector.py
```

**操作说明：**
- 按数字键（0-9）开始录制对应手势
- 按空格键停止录制
- 按q键退出

**默认手势标签：**
- 1: 你好 (hello)
- 2: 谢谢 (thank you)
- 3: 再见 (goodbye)
- 4: 是 (yes)
- 5: 不 (no)
- 6: 我 (I)
- 7: 你 (you)
- 8: 爱 (love)
- 9: 家 (home)
- 0: 水 (water)

### 2. 数据预处理

处理收集的原始数据：

```bash
python data_preprocessor.py
```

这将生成：
- 标准化的特征数据
- 训练/验证/测试集分割
- 数据统计信息和可视化

### 3. 模型训练

选择模型架构进行训练：

```bash
# 使用LSTM模型
python -c "
from trainer import HandGestureTrainer
from data_preprocessor import HandGesturePreprocessor

# 加载数据
preprocessor = HandGesturePreprocessor()
data_splits = preprocessor.load_processed_data('data/processed/processed_data_latest.pkl')

# 创建训练器
trainer = HandGestureTrainer(model_type='lstm')
trainer.prepare_data(data_splits)
trainer.build_model(hidden_dim=128, num_layers=2)
trainer.setup_training(learning_rate=0.001)

# 开始训练
trainer.train(epochs=100)

# 评估模型
results = trainer.evaluate()
trainer.plot_training_history()
"
```

**支持的模型类型：**
- `lstm`: LSTM循环神经网络
- `gru`: GRU门控循环单元
- `transformer`: Transformer注意力模型
- `cnn_lstm`: CNN-LSTM混合模型
- `attention_lstm`: 带注意力的LSTM
- `resnet1d`: 1D残差网络
- `multitask`: 多任务学习模型

### 4. 实时推理

启动实时手语识别：

```bash
python inference.py
```

**快捷键：**
- `q`: 退出程序
- `s`: 保存预测历史
- `c`: 清空预测历史
- `h`: 显示帮助信息

## 模型性能对比

| 模型类型 | 参数量 | 训练速度 | 准确率 | 推理速度 | 特点 |
|---------|--------|----------|--------|----------|------|
| LSTM | 中等 | 快 | 85-90% | 快 | 基础序列模型 |
| GRU | 中等 | 快 | 85-88% | 快 | 更简单的LSTM |
| Transformer | 高 | 慢 | 90-95% | 中等 | 注意力机制 |
| CNN-LSTM | 中高 | 中等 | 88-92% | 中等 | 特征提取+序列 |
| Attention-LSTM | 中高 | 中等 | 88-91% | 中等 | 带注意力LSTM |
| ResNet1D | 中高 | 中等 | 87-90% | 中等 | 残差连接 |
| MultiTask | 高 | 慢 | 90-93% | 慢 | 多任务学习 |

## 配置说明

### 数据收集配置

在 `data_collector.py` 中修改：

```python
# 每个手势录制帧数
max_frames_per_gesture = 60

# 手势标签配置
gesture_labels = {
    "1": {"chinese": "你好", "english": "hello"},
    # 添加更多手势...
}
```

### 预处理配置

在 `data_preprocessor.py` 中修改：

```python
feature_config = {
    "sequence_length": 30,      # 序列长度
    "palm_features": True,      # 是否包含手掌特征
    "arm_features": True,       # 是否包含手臂特征
    "digit_features": True,     # 是否包含手指特征
    "velocity_features": True,  # 是否包含速度特征
    "angle_features": True,     # 是否包含角度特征
    "distance_features": True   # 是否包含距离特征
}
```

### 训练配置

在训练时可以调整：

```python
# 模型参数
trainer.build_model(
    hidden_dim=128,         # 隐藏层维度
    num_layers=2,           # 层数
    dropout=0.3,            # Dropout率
    bidirectional=True      # 是否双向（LSTM/GRU）
)

# 训练参数
trainer.setup_training(
    learning_rate=0.001,    # 学习率
    optimizer_type="adam",  # 优化器类型
    scheduler_type="cosine", # 学习率调度器
    use_early_stopping=True, # 是否使用早停
    patience=10             # 早停耐心值
)
```

### 推理配置

在 `inference.py` 中修改：

```python
# 手势缓冲器配置
gesture_buffer = GestureBuffer(
    max_length=30,          # 最大缓冲长度
    min_length=10,          # 最小预测长度
    motion_threshold=0.1,   # 运动检测阈值
    stillness_duration=1.0  # 静止持续时间
)

# 置信度跟踪器配置
confidence_tracker = ConfidenceTracker(
    window_size=5,          # 滑动窗口大小
    threshold=0.7           # 置信度阈值
)
```

## 高级用法

### 自定义模型

创建自定义模型架构：

```python
import torch.nn as nn
from model_definition import ModelFactory

class CustomModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # 定义你的模型架构
        
    def forward(self, x):
        # 定义前向传播
        return output

# 注册到工厂
ModelFactory.register_model("custom", CustomModel)
```

### 多任务训练

训练多任务模型：

```python
trainer = HandGestureTrainer(model_type="multitask")
trainer.prepare_multitask_data(data_splits)
trainer.build_model(
    num_gesture_classes=10,
    num_chinese_classes=10,
    num_english_classes=10
)
trainer.train(epochs=100)
```

### 数据增强

在预处理阶段添加数据增强：

```python
# 在data_preprocessor.py中添加
def augment_sequence(self, sequence):
    # 时间扭曲
    # 噪声添加
    # 旋转变换
    return augmented_sequence
```

### 模型融合

使用多个模型进行集成预测：

```python
models = [
    load_model("best_lstm_model.pth"),
    load_model("best_transformer_model.pth"),
    load_model("best_cnn_lstm_model.pth")
]

# 集成预测
ensemble_prediction = ensemble_predict(models, input_data)
```

## 性能优化

### 训练优化

1. **混合精度训练**：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
# 在训练循环中使用autocast
```

2. **学习率调度**：
```python
# 使用余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

3. **梯度裁剪**：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 推理优化

1. **模型量化**：
```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

2. **批处理推理**：
```python
# 收集多个手势序列进行批量推理
batch_predictions = model(batch_input)
```

## 常见问题

### Q1: Leap Motion设备无法连接

**解决方案：**
1. 确保Leap Motion驱动已正确安装
2. 检查USB连接是否稳定
3. 重启Leap Motion服务
4. 检查设备管理器中是否有未知设备

### Q2: 训练过程中显存不足

**解决方案：**
1. 减少批处理大小
2. 使用梯度累积
3. 减少模型参数
4. 使用混合精度训练

### Q3: 识别准确率低

**解决方案：**
1. 增加训练数据量
2. 调整特征提取参数
3. 尝试不同的模型架构
4. 调整预处理超参数
5. 使用数据增强技术

### Q4: 语音播报不工作

**解决方案：**
1. 检查pyttsx3是否正确安装
2. 确认系统有可用的TTS引擎
3. 检查音频设备设置
4. 尝试不同的语音引擎

### Q5: 实时推理延迟高

**解决方案：**
1. 使用GPU加速
2. 减少模型复杂度
3. 优化特征提取过程
4. 使用模型量化
5. 调整缓冲区大小

## 相关资源

- [Leap Motion开发者文档](https://developer.leapmotion.com/)
- [PyTorch官方文档](https://pytorch.org/docs/)
- [手语识别论文集](https://github.com/topics/sign-language-recognition)
- [深度学习最佳实践](https://www.deeplearningbook.org/)


## 作者
  * yigeoooo
  * XXK
---

**注意**: 本项目仅供学习和研究使用，商业用途请遵循相关许可证条款。



import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Callable
from collections import deque
from datetime import datetime
import copy
from model_definition import ModelFactory
from data_preprocessor import HandGesturePreprocessor


class GestureBuffer:
    def __init__(self, max_length=30, no_hand_duration=1.0):
        """
        max_length: 每次采集的最大帧数
        no_hand_duration: 连续检测不到手超过此秒数时结束采集
        """
        self.buffer = []
        self.max_length = max_length
        self.no_hand_duration = no_hand_duration
        self.last_hand_time = None
        self.is_collecting = False

    def add_frame(self, frame_data: Dict) -> bool:
        """
        添加一帧数据
        return True 表示采集完成，可以触发推理
        """
        current_time = time.time()
        has_hands = len(frame_data.get('hands', [])) > 0

        if has_hands:
            self.last_hand_time = current_time
            if not self.is_collecting:
                self.is_collecting = True
                self.buffer.clear()
                print(f"开始收集手势数据...")
            # ★ 深拷贝防止后续引用被修改为空
            self.buffer.append(copy.deepcopy(frame_data))
            print(f"收集中... {len(self.buffer)}/{self.max_length} 帧")

            # ★ 达到最大帧数立即结束收集
            if len(self.buffer) >= self.max_length:
                self.is_collecting = False
                print(f"收集完成! 共收集 {len(self.buffer)} 帧，准备预测...")
                return True

        else:
            # 没有手时，如果已经在采集中且超过 no_hand_duration 并且采集帧数足够，结束采集
            if (self.is_collecting and
                    current_time - self.last_hand_time > self.no_hand_duration and
                    len(self.buffer) >= 10):
                self.is_collecting = False
                print(f"收集完成! 共收集 {len(self.buffer)} 帧，准备预测...")
                return True
            elif self.is_collecting:
                # 没有手，但采集中，仍然存入以保留时间轴信息
                self.buffer.append(copy.deepcopy(frame_data))
                print(f"手势结束，等待静止...")

        return False

    def get_sequence(self) -> List[Dict]:
        """返回当前缓冲区的帧序列"""
        return self.buffer

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.is_collecting = False


class HandGestureInference:
    """手语识别推理引擎"""

    def __init__(self, model_path: str, preprocessor_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载预处理器和模型
        self.preprocessor = HandGesturePreprocessor()
        self.data_splits = self.preprocessor.load_processed_data(preprocessor_path)

        # 检查训练数据分布
        self._check_data_distribution()

        self.model = self._load_model(model_path)
        self.model.eval()

        # 预测控制
        self.last_prediction_time = 0
        self.prediction_cooldown = 2.0

        print(f"推理引擎初始化完成，使用设备: {self.device}")

    def _check_data_distribution(self):
        """检查训练数据分布"""
        print("\n 训练数据分布检查:")

        if hasattr(self.preprocessor, 'label_decoder') and 'gesture' in self.preprocessor.label_decoder:
            gesture_decoder = self.preprocessor.label_decoder['gesture']
            print(f"  总共有 {len(gesture_decoder)} 个手势类别:")

            for idx, gesture in gesture_decoder.items():
                print(f"    类别{idx}: {gesture}")

        # 检查训练集标签分布
        if 'y_gesture_train' in self.data_splits:
            y_train = self.data_splits['y_gesture_train']
            unique, counts = np.unique(y_train, return_counts=True)

            print(f"\n  训练集样本分布:")
            total_samples = len(y_train)
            for class_idx, count in zip(unique, counts):
                if class_idx in gesture_decoder:
                    gesture_name = gesture_decoder[class_idx]
                    percentage = (count / total_samples) * 100
                    print(f"    {gesture_name}: {count} 样本 ({percentage:.1f}%)")

            # 检查数据不平衡
            min_count = np.min(counts)
            max_count = np.max(counts)
            imbalance_ratio = max_count / min_count

            print(f"\n  数据平衡性分析:")
            print(f"    最少样本: {min_count}")
            print(f"    最多样本: {max_count}")
            print(f"    不平衡比率: {imbalance_ratio:.2f}")

            if imbalance_ratio > 5:
                print(f"    严重数据不平衡！某些类别样本过少可能导致识别偏向多数类别")
            elif imbalance_ratio > 2:
                print(f"    轻微数据不平衡")
            else:
                print(f"    数据分布相对平衡")

        print("-" * 50)

    def _load_model(self, model_path: str):
        """加载模型"""
        print(f"正在加载模型: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_type = checkpoint.get('model_type', 'lstm')
        print(f"模型类型: {model_type}")

        sample_data = self.data_splits['X_train']
        input_dim = sample_data.shape[-1]
        num_classes = len(self.preprocessor.label_decoder['gesture'])

        print(f"数据输入维度: {input_dim}")
        print(f"类别数: {num_classes}")

        state_dict = checkpoint['model_state_dict']

        # 直接在这里推断配置，不调用额外方法
        config = {}

        try:
            if model_type == "lstm":
                # 纯LSTM模型
                if 'lstm.weight_ih_l0' in state_dict:
                    hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
                    num_layers = 2 if 'lstm.weight_ih_l1' in state_dict else 1
                    bidirectional = 'lstm.weight_ih_l0_reverse' in state_dict

                    config = {
                        'hidden_dim': hidden_size,  # 使用LSTM模型的正确参数名
                        'num_layers': num_layers,
                        'bidirectional': bidirectional,
                        'dropout': 0.3
                    }
                    print(f"推断LSTM配置: {config}")

            elif model_type == "cnn_lstm":
                # CNN-LSTM模型
                # CNN配置
                if 'cnn_extractor.cnn_layers.0.weight' in state_dict:
                    first_channels = state_dict['cnn_extractor.cnn_layers.0.weight'].shape[0]
                    second_channels = first_channels * 2
                    if 'cnn_extractor.cnn_layers.5.weight' in state_dict:
                        second_channels = state_dict['cnn_extractor.cnn_layers.5.weight'].shape[0]
                    config['cnn_channels'] = [first_channels, second_channels]

                # LSTM配置
                if 'lstm.weight_ih_l0' in state_dict:
                    hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
                    config['lstm_hidden_size'] = hidden_size  # CNN-LSTM使用这个参数名
                    config['lstm_num_layers'] = 2 if 'lstm.weight_ih_l1' in state_dict else 1
                    config['bidirectional'] = 'lstm.weight_ih_l0_reverse' in state_dict

                # 分类器配置
                if 'classifier.1.weight' in state_dict:
                    config['classifier_hidden_size'] = state_dict['classifier.1.weight'].shape[0]

                print(f"推断CNN-LSTM配置: {config}")

            else:
                print(f"使用默认配置创建 {model_type} 模型")

        except Exception as e:
            print(f"推断配置时出错: {e}，将使用默认配置")
            config = {}

        # 创建模型
        try:
            model = ModelFactory.create_model(
                model_type,
                input_dim=input_dim,
                num_classes=num_classes,
                **config
            )
            print("模型创建成功")

        except Exception as e:
            print(f"使用推断配置创建模型失败: {e}")
            print("尝试使用默认配置...")
            model = ModelFactory.create_model(model_type, input_dim=input_dim, num_classes=num_classes)

        # 加载权重
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("模型权重严格加载成功")
        except RuntimeError as e:
            print(f"严格加载失败: {e}")
            print("尝试非严格加载...")
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )

            if missing_keys:
                print(f"缺失参数: {len(missing_keys)} 个")
                if len(missing_keys) <= 5:
                    print(f"  具体: {missing_keys}")
            if unexpected_keys:
                print(f"多余参数: {len(unexpected_keys)} 个")
                if len(unexpected_keys) <= 5:
                    print(f"  具体: {unexpected_keys}")

            # 检查关键参数是否缺失
            critical_missing = [key for key in missing_keys
                                if any(critical in key for critical in ['cnn', 'lstm', 'classifier'])]
            if critical_missing:
                print(f"关键层参数缺失: {critical_missing}")
                raise RuntimeError(f"无法加载模型，关键参数缺失")

            print("部分参数加载成功，继续运行...")

        model.to(self.device)
        return model

    def extract_hand_features(self, hand_data: Dict) -> np.ndarray:
        """提取手部特征"""
        features = []

        # 基本特征
        features.extend([
            1.0 if hand_data.get("hand_type") == "right" else 0.0,
            hand_data.get("confidence", 0),
            hand_data.get("grab_strength", 0),
            hand_data.get("grab_angle", 0),
            hand_data.get("pinch_distance", 0),
            hand_data.get("pinch_strength", 0)
        ])

        # 手掌特征
        palm = hand_data.get("palm", {})
        features.extend(palm.get("position", [0, 0, 0]))
        features.extend(palm.get("direction", [0, 0, 0]))
        features.extend(palm.get("normal", [0, 0, 0]))
        features.extend(palm.get("velocity", [0, 0, 0]))
        features.append(palm.get("width", 0))

        # 手臂特征
        arm = hand_data.get("arm", {})
        features.extend(arm.get("prev_joint", [0, 0, 0]))
        features.extend(arm.get("next_joint", [0, 0, 0]))
        features.extend(arm.get("direction", [0, 0, 0]))
        features.append(arm.get("length", 0))
        features.append(arm.get("width", 0))

        # 手指特征
        digits = hand_data.get("digits", [])
        for digit_idx in range(5):
            if digit_idx < len(digits):
                digit = digits[digit_idx]
                features.append(float(digit.get("is_extended", True)))

                bones = digit.get("bones", [])
                for bone_idx in range(4):
                    if bone_idx < len(bones):
                        bone = bones[bone_idx]
                        features.extend(bone.get("prev_joint", [0, 0, 0]))
                        features.extend(bone.get("next_joint", [0, 0, 0]))
                        features.extend(bone.get("direction", [0, 0, 0]))
                        features.append(bone.get("length", 0))
                        features.append(bone.get("width", 0))
                    else:
                        features.extend([0] * 11)
            else:
                features.append(1.0)
                features.extend([0] * 44)

        return np.array(features[:200])  # 限制特征长度

    def extract_frame_features(self, frame_data: Dict) -> np.ndarray:
        """提取帧特征"""
        hands = frame_data.get("hands", [])

        # 双手特征
        left_hand_features = np.zeros(200)
        right_hand_features = np.zeros(200)

        for hand in hands:
            hand_features = self.extract_hand_features(hand)
            hand_features = np.pad(hand_features, (0, max(0, 200 - len(hand_features))))[:200]

            if hand.get("hand_type") == "left":
                left_hand_features = hand_features
            else:
                right_hand_features = hand_features

        # 合并特征
        features = np.concatenate([left_hand_features, right_hand_features, [len(hands)]])
        return features

    def predict_gesture(self, sequence_data: List[Dict]) -> Optional[Dict]:
        """预测手语"""
        print(f"开始预测，序列长度: {len(sequence_data)}")

        # 过滤掉未检测到手的帧
        valid_sequence = []
        for i, frame in enumerate(sequence_data):
            hands_count = len(frame.get('hands', []))
            if hands_count > 0:
                frame_features = self.extract_frame_features(frame)
                non_zero_ratio = np.count_nonzero(frame_features) / frame_features.size
                print(
                    f"帧{i}: 检测到{hands_count}只手, 特征维度: {len(frame_features)}, 非零比例: {non_zero_ratio:.3f}")
                valid_sequence.append(frame)
            else:
                print(f"帧{i}: 未检测到手，跳过")

        # 检查有效帧数量
        if len(valid_sequence) < 10:
            print(f"有效帧不足（{len(valid_sequence)}/10），请重新录制")
            return None

        try:
            # 提取特征序列（仅用有效帧）
            features = []
            for frame in valid_sequence:
                frame_features = self.extract_frame_features(frame)
                features.append(frame_features)

            # 标准化序列长度到30帧
            features = np.array(features)
            print(f"原始特征形状: {features.shape}")

            target_length = 30
            if len(features) < target_length:
                padding = np.repeat(features[-1:], target_length - len(features), axis=0)
                features = np.vstack([features, padding])
                print(f"序列过短，填充到: {features.shape}")
            elif len(features) > target_length:
                indices = np.linspace(0, len(features) - 1, target_length, dtype=int)
                features = features[indices]
                print(f"序列过长，采样到: {features.shape}")

            # 数据缩放
            original_shape = features.shape
            features_reshaped = features.reshape(-1, features.shape[-1])
            print(f"缩放前形状: {features_reshaped.shape}")

            features_scaled = self.preprocessor.scaler.transform(features_reshaped)
            features = features_scaled.reshape(original_shape)
            print(f"缩放后形状: {features.shape}")

            # 模型预测
            input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            print(f"输入tensor形状: {input_tensor.shape}")

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = torch.max(probs).item()

                print(f"模型输出分析:")
                print(f"  预测类别: {pred_idx}")
                print(f"  最高置信度: {confidence:.3f}")

                # 显示所有类别的概率分布
                print(f"  类别概率分布:")
                probs_np = probs[0].cpu().numpy()
                for i, prob in enumerate(probs_np):
                    if i in self.preprocessor.label_decoder['gesture']:
                        gesture_name = self.preprocessor.label_decoder['gesture'][i]
                        marker = " ← 预测" if i == pred_idx else ""
                        print(f"    类别{i}({gesture_name}): {prob:.4f}{marker}")

                # 对于2分类问题，检查是否有明确的区分
                if len(probs_np) == 2:
                    prob_diff = abs(probs_np[0] - probs_np[1])
                    print(f"  两类概率差值: {prob_diff:.4f}")
                    if prob_diff < 0.2:
                        print(f"    两类概率很接近，模型不确定")
                    elif prob_diff > 0.6:
                        print(f"    预测很确定")
                    else:
                        print(f"    预测较为确定")

                # 检查概率分布是否正常
                prob_std = np.std(probs_np)
                if prob_std < 0.05:
                    print(f"    警告: 概率分布过于平均，模型可能没学到区别")

                print(f"    概率标准差: {prob_std:.4f}")

            gesture_label = self.preprocessor.label_decoder['gesture'][pred_idx]
            print(f"预测的手势标签: {gesture_label}")

            # 获取中英文含义
            gesture_labels_path = os.path.join(self.preprocessor.data_dir, "gesture_labels.json")
            chinese_meaning = "未知"
            english_meaning = "unknown"

            if os.path.exists(gesture_labels_path):
                with open(gesture_labels_path, 'r', encoding='utf-8') as f:
                    gesture_labels = json.load(f)
                    gesture_info = gesture_labels.get(gesture_label, {})
                    chinese_meaning = gesture_info.get('chinese', '未知')
                    english_meaning = gesture_info.get('english', 'unknown')
                    print(f"标签映射: {gesture_label} -> {chinese_meaning}/{english_meaning}")
            else:
                print("警告: gesture_labels.json 文件不存在")

            return {
                'gesture_label': gesture_label,
                'chinese_meaning': chinese_meaning,
                'english_meaning': english_meaning,
                'confidence': confidence,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            print(f"预测出错: {e}")
            import traceback
            traceback.print_exc()
            return None


class GestureRecognitionCore:
    """手语识别核心逻辑类"""

    def __init__(self, model_path: str, preprocessor_path: str):
        self.inference_engine = HandGestureInference(model_path, preprocessor_path)
        self.gesture_buffer = GestureBuffer()

        # 识别状态
        self.current_result = None
        self.result_display_time = 5.0
        self.result_start_time = 0
        self.recognition_status = "WAITING"  # WAITING, COLLECTING, SUCCESS, FAILED

        # 统计
        self.total_attempts = 0
        self.successful_recognitions = 0

        # 状态更新回调
        self.status_callback: Optional[Callable] = None

    def set_status_callback(self, callback: Callable):
        """设置状态更新回调函数"""
        self.status_callback = callback

    def process_prediction(self, frame_data: Dict) -> Optional[Dict]:
        """处理预测"""
        current_time = time.time()

        # 检查冷却时间
        if current_time - self.inference_engine.last_prediction_time < self.inference_engine.prediction_cooldown:
            return None

        # 添加到缓冲区
        should_predict = self.gesture_buffer.add_frame(frame_data)

        if should_predict:
            sequence = self.gesture_buffer.get_sequence()
            if sequence:
                self.total_attempts += 1
                result = self.inference_engine.predict_gesture(sequence)

                if result and result['confidence'] > 0.4:  # 调整为更合理的阈值
                    self.successful_recognitions += 1
                    self.inference_engine.last_prediction_time = current_time

                    print(f"\n 识别成功!")
                    print(f"手势: {result['gesture_label']}")
                    print(f"中文: {result['chinese_meaning']}")
                    print(f"英文: {result['english_meaning']}")
                    print(f"置信度: {result['confidence']:.3f}")
                    print("-" * 40)

                    return result
                else:
                    confidence_str = f"{result['confidence']:.3f}" if result else "无预测结果"
                    print(f" 识别失败: 置信度{confidence_str} (需要>0.4)")

                    # 显示备选预测
                    if result:
                        print(f"   当前预测: {result['chinese_meaning']}({result['english_meaning']})")

                    return {"status": "failed", "reason": f"置信度{confidence_str}"}

        return None

    def update_result(self, result):
        """更新识别结果"""
        if result:
            if result.get("status") == "failed":
                self.recognition_status = "FAILED"
            else:
                self.recognition_status = "SUCCESS"
                self.current_result = result
            self.result_start_time = time.time()

            # 调用状态更新回调
            if self.status_callback:
                self.status_callback(self.recognition_status, result)

    def update_collecting_status(self, has_hands: bool):
        """更新采集状态"""
        current_time = time.time()

        if has_hands and self.recognition_status == "WAITING":
            self.recognition_status = "COLLECTING"
            if self.status_callback:
                self.status_callback(self.recognition_status, None)
        elif (self.recognition_status in ["SUCCESS", "FAILED"] and
              current_time - self.result_start_time > self.result_display_time):
            self.recognition_status = "WAITING"
            self.current_result = None
            if self.status_callback:
                self.status_callback(self.recognition_status, None)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        success_rate = (self.successful_recognitions / max(1, self.total_attempts) * 100)
        return {
            'total_attempts': self.total_attempts,
            'successful_recognitions': self.successful_recognitions,
            'success_rate': success_rate,
            'buffer_length': len(self.gesture_buffer.buffer),
            'max_buffer_length': self.gesture_buffer.max_length
        }

    def get_current_status(self) -> Dict:
        """获取当前状态"""
        return {
            'status': self.recognition_status,
            'result': self.current_result,
            'stats': self.get_stats()
        }
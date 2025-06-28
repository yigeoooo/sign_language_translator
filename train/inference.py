import torch
import torch.nn.functional as F
import numpy as np
import cv2
import leap
import json
import os
import time
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime

from model_definition import ModelFactory
from data_preprocessor import HandGesturePreprocessor


class GestureBuffer:
    """手势数据缓冲器"""

    def __init__(self, max_length: int = 30):
        self.max_length = max_length
        self.buffer = deque(maxlen=max_length)
        self.is_collecting = False
        self.last_hand_time = 0
        self.no_hand_duration = 1.0  # 没有手的持续时间

    def add_frame(self, frame_data: Dict) -> bool:
        """添加帧数据，返回是否应该预测"""
        current_time = time.time()
        has_hands = len(frame_data.get('hands', [])) > 0

        if has_hands:
            self.last_hand_time = current_time
            if not self.is_collecting:
                self.is_collecting = True
                self.buffer.clear()
                print(f"开始收集手势数据...")
            self.buffer.append(frame_data)
            if len(self.buffer) % 5 == 0:  # 每5帧打印一次
                print(f"收集中... {len(self.buffer)}/{self.max_length} 帧")
        else:
            if (self.is_collecting and
                    current_time - self.last_hand_time > self.no_hand_duration and
                    len(self.buffer) >= 10):

                self.is_collecting = False
                print(f"收集完成! 共收集 {len(self.buffer)} 帧，准备预测...")
                return True
            elif self.is_collecting:
                self.buffer.append(frame_data)
                print(f"手势结束，等待静止...")

        return False

    def get_sequence(self) -> List[Dict]:
        return list(self.buffer)


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
        print("\n📊 训练数据分布检查:")

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
                print(f"    ⚠️ 严重数据不平衡！某些类别样本过少可能导致识别偏向多数类别")
            elif imbalance_ratio > 2:
                print(f"    ⚠️ 轻微数据不平衡")
            else:
                print(f"    ✅ 数据分布相对平衡")

        print("-" * 50)

    def _load_model(self, model_path: str):
        """加载模型 - 修复版本，不依赖额外方法"""
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
            print("✅ 模型创建成功")

        except Exception as e:
            print(f"使用推断配置创建模型失败: {e}")
            print("尝试使用默认配置...")
            model = ModelFactory.create_model(model_type, input_dim=input_dim, num_classes=num_classes)

        # 加载权重
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("✅ 模型权重严格加载成功")
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
                print(f"❌ 关键层参数缺失: {critical_missing}")
                raise RuntimeError(f"无法加载模型，关键参数缺失")

            print("⚠️ 部分参数加载成功，继续运行...")

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

        if len(sequence_data) < 10:
            print(f"序列太短: {len(sequence_data)} < 10")
            return None

        try:
            # 提取特征序列
            features = []
            for i, frame in enumerate(sequence_data):
                frame_features = self.extract_frame_features(frame)
                features.append(frame_features)
                if i < 3:  # 只打印前3帧的信息
                    hands_count = len(frame.get('hands', []))
                    print(f"帧{i}: 检测到{hands_count}只手, 特征维度: {len(frame_features)}")

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
                        print(f"    ⚠️ 两类概率很接近，模型不确定")
                    elif prob_diff > 0.6:
                        print(f"    ✅ 预测很确定")
                    else:
                        print(f"    🔶 预测较为确定")

                # 检查概率分布是否正常
                prob_std = np.std(probs_np)
                if prob_std < 0.05:
                    print(f"    ⚠️ 警告: 概率分布过于平均，模型可能没学到区别")

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


class RealTimeGestureRecognizer:
    """实时手语识别器"""

    def __init__(self, model_path: str, preprocessor_path: str):
        self.inference_engine = HandGestureInference(model_path, preprocessor_path)
        self.gesture_buffer = GestureBuffer()

        # 界面设置
        self.screen_size = [600, 800]
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)

        # 识别状态
        self.current_result = None
        self.result_display_time = 5.0
        self.result_start_time = 0
        self.recognition_status = "WAITING"  # WAITING, COLLECTING, SUCCESS, FAILED

        # 统计
        self.total_attempts = 0
        self.successful_recognitions = 0

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

                    print(f"\n✅ 识别成功!")
                    print(f"手势: {result['gesture_label']}")
                    print(f"中文: {result['chinese_meaning']}")
                    print(f"英文: {result['english_meaning']}")
                    print(f"置信度: {result['confidence']:.3f}")
                    print("-" * 40)

                    return result
                else:
                    confidence_str = f"{result['confidence']:.3f}" if result else "无预测结果"
                    print(f"❌ 识别失败: 置信度{confidence_str} (需要>0.4)")

                    # 显示备选预测
                    if result:
                        print(f"   当前预测: {result['chinese_meaning']}({result['english_meaning']})")

                    return {"status": "failed", "reason": f"置信度{confidence_str}"}

        return None

    def render_interface(self, event):
        """渲染界面"""
        # 清空画布
        self.output_image[:] = 0

        # 标题
        cv2.putText(self.output_image, "Sign Language Recognition",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 统计信息
        y = 80
        success_rate = (self.successful_recognitions / max(1, self.total_attempts) * 100)
        stats = [
            f"Attempts: {self.total_attempts}",
            f"Success: {self.successful_recognitions}",
            f"Rate: {success_rate:.1f}%",
            f"Buffer: {len(self.gesture_buffer.buffer)}/{self.gesture_buffer.max_length}"
        ]

        for stat in stats:
            cv2.putText(self.output_image, stat, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 25

        # 状态显示区域
        status_y = y + 20
        status_height = 200

        # 状态颜色
        colors = {
            "SUCCESS": (0, 255, 0),
            "FAILED": (0, 0, 255),
            "COLLECTING": (0, 255, 255),
            "WAITING": (128, 128, 128)
        }
        color = colors.get(self.recognition_status, (128, 128, 128))

        # 状态框
        cv2.rectangle(self.output_image, (20, status_y), (self.screen_size[1] - 20, status_y + status_height), color, 2)

        # 状态内容
        if self.recognition_status == "SUCCESS" and self.current_result:
            cv2.putText(self.output_image, "RECOGNITION SUCCESS!",
                        (30, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(self.output_image, f"Gesture: {self.current_result['gesture_label']}",
                        (30, status_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(self.output_image, f"Chinese: {self.current_result['chinese_meaning']}",
                        (30, status_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(self.output_image, f"English: {self.current_result['english_meaning']}",
                        (30, status_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(self.output_image, f"Confidence: {self.current_result['confidence']:.2f}",
                        (30, status_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif self.recognition_status == "FAILED":
            cv2.putText(self.output_image, "RECOGNITION FAILED!",
                        (30, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(self.output_image, "Try making a clearer gesture",
                        (30, status_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif self.recognition_status == "COLLECTING":
            cv2.putText(self.output_image, "COLLECTING GESTURE...",
                        (30, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 进度条
            progress = len(self.gesture_buffer.buffer) / self.gesture_buffer.max_length
            bar_width = int((self.screen_size[1] - 60) * progress)
            cv2.rectangle(self.output_image, (30, status_y + 70), (30 + bar_width, status_y + 90), (0, 255, 255), -1)
            cv2.rectangle(self.output_image, (30, status_y + 70), (self.screen_size[1] - 30, status_y + 90),
                          (255, 255, 255), 1)

            cv2.putText(self.output_image, f"Progress: {progress * 100:.0f}%",
                        (30, status_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        else:  # WAITING
            cv2.putText(self.output_image, "WAITING FOR GESTURE...",
                        (30, status_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(self.output_image, "Place hand above sensor",
                        (30, status_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制手部骨架
        if hasattr(event, 'hands') and event.hands:
            if self.recognition_status == "WAITING":
                self.recognition_status = "COLLECTING"

            for hand in event.hands:
                self._draw_hand_skeleton(hand)
        else:
            # 状态转换逻辑
            current_time = time.time()
            if (self.recognition_status in ["SUCCESS", "FAILED"] and
                    current_time - self.result_start_time > self.result_display_time):
                self.recognition_status = "WAITING"
                self.current_result = None

        # 控制说明
        cv2.putText(self.output_image, "Press 'q' to quit",
                    (20, self.screen_size[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def _draw_hand_skeleton(self, hand):
        """绘制手部骨架"""
        try:
            # 手臂
            wrist = self._get_joint_position(hand.arm.next_joint)
            elbow = self._get_joint_position(hand.arm.prev_joint)

            if wrist and elbow:
                cv2.line(self.output_image, wrist, elbow, (255, 255, 255), 2)
                cv2.circle(self.output_image, wrist, 4, (255, 255, 255), -1)
                cv2.circle(self.output_image, elbow, 4, (255, 255, 255), -1)

            # 手指
            for digit in hand.digits:
                for bone in digit.bones:
                    start = self._get_joint_position(bone.prev_joint)
                    end = self._get_joint_position(bone.next_joint)

                    if start and end:
                        cv2.line(self.output_image, start, end, (255, 255, 255), 2)
                        cv2.circle(self.output_image, start, 3, (255, 255, 255), -1)
                        cv2.circle(self.output_image, end, 3, (255, 255, 255), -1)
        except:
            pass

    def _get_joint_position(self, joint):
        """获取关节屏幕位置"""
        if joint:
            x = int(joint.x + (self.screen_size[1] / 2))
            y = int(joint.z + (self.screen_size[0] / 2))
            return (x, y)
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


class GestureRecognitionListener(leap.Listener):
    """Leap Motion监听器"""

    def __init__(self, recognizer: RealTimeGestureRecognizer):
        self.recognizer = recognizer

    def on_connection_event(self, event):
        print("Leap Motion连接成功")

    def on_tracking_event(self, event):
        try:
            # 渲染界面
            self.recognizer.render_interface(event)

            # 构建帧数据
            frame_data = {'timestamp': time.time(), 'hands': []}

            # 提取手部数据
            if hasattr(event, 'hands') and event.hands:
                for hand in event.hands:
                    hand_data = self._extract_hand_data(hand)
                    if hand_data:
                        frame_data['hands'].append(hand_data)

            # 处理预测
            result = self.recognizer.process_prediction(frame_data)
            if result:
                self.recognizer.update_result(result)

        except Exception as e:
            print(f"处理跟踪事件出错: {e}")

    def _extract_hand_data(self, hand):
        """提取手部数据"""
        try:
            return {
                "hand_type": "left" if hasattr(hand, 'type') and hand.type == leap.HandType.Left else "right",
                "confidence": getattr(hand, 'confidence', 1.0),
                "grab_strength": getattr(hand, 'grab_strength', 0.0),
                "grab_angle": getattr(hand, 'grab_angle', 0.0),
                "pinch_distance": getattr(hand, 'pinch_distance', 0.0),
                "pinch_strength": getattr(hand, 'pinch_strength', 0.0),
                "palm": {
                    "position": self._get_vector3(getattr(getattr(hand, 'palm', None), 'position', None)),
                    "direction": self._get_vector3(getattr(getattr(hand, 'palm', None), 'direction', None)),
                    "normal": self._get_vector3(getattr(getattr(hand, 'palm', None), 'normal', None)),
                    "velocity": self._get_vector3(getattr(getattr(hand, 'palm', None), 'velocity', None)),
                    "width": getattr(getattr(hand, 'palm', None), 'width', 0.0)
                },
                "arm": {
                    "prev_joint": self._get_vector3(getattr(getattr(hand, 'arm', None), 'prev_joint', None)),
                    "next_joint": self._get_vector3(getattr(getattr(hand, 'arm', None), 'next_joint', None)),
                    "direction": self._get_vector3(getattr(getattr(hand, 'arm', None), 'direction', None)),
                    "length": getattr(getattr(hand, 'arm', None), 'length', 0.0),
                    "width": getattr(getattr(hand, 'arm', None), 'width', 0.0)
                },
                "digits": self._extract_digits(hand)
            }
        except Exception as e:
            print(f"提取手部数据出错: {e}")
            return None

    def _extract_digits(self, hand):
        """提取手指数据"""
        digits = []
        try:
            if hasattr(hand, 'digits') and hand.digits:
                for digit_idx in range(min(5, len(hand.digits))):
                    digit = hand.digits[digit_idx]
                    digit_data = {
                        "digit_type": digit_idx,
                        "is_extended": getattr(digit, 'is_extended', True),
                        "bones": []
                    }

                    if hasattr(digit, 'bones') and digit.bones:
                        for bone_idx in range(min(4, len(digit.bones))):
                            bone = digit.bones[bone_idx]
                            bone_data = {
                                "bone_type": bone_idx,
                                "prev_joint": self._get_vector3(getattr(bone, 'prev_joint', None)),
                                "next_joint": self._get_vector3(getattr(bone, 'next_joint', None)),
                                "direction": self._get_vector3(getattr(bone, 'direction', None)),
                                "length": getattr(bone, 'length', 0.0),
                                "width": getattr(bone, 'width', 0.0)
                            }
                            digit_data["bones"].append(bone_data)

                    # 补齐4个骨骼
                    while len(digit_data["bones"]) < 4:
                        digit_data["bones"].append({
                            "bone_type": len(digit_data["bones"]),
                            "prev_joint": [0.0, 0.0, 0.0],
                            "next_joint": [0.0, 0.0, 0.0],
                            "direction": [0.0, 0.0, 0.0],
                            "length": 0.0,
                            "width": 0.0
                        })

                    digits.append(digit_data)

            # 补齐5个手指
            while len(digits) < 5:
                digits.append({
                    "digit_type": len(digits),
                    "is_extended": True,
                    "bones": [{
                        "bone_type": i,
                        "prev_joint": [0.0, 0.0, 0.0],
                        "next_joint": [0.0, 0.0, 0.0],
                        "direction": [0.0, 0.0, 0.0],
                        "length": 0.0,
                        "width": 0.0
                    } for i in range(4)]
                })

        except Exception as e:
            print(f"提取手指数据出错: {e}")

        return digits

    def _get_vector3(self, vector):
        """安全获取3D向量"""
        if vector is None:
            return [0.0, 0.0, 0.0]
        try:
            if hasattr(vector, 'x') and hasattr(vector, 'y') and hasattr(vector, 'z'):
                return [float(vector.x), float(vector.y), float(vector.z)]
        except:
            pass
        return [0.0, 0.0, 0.0]


def main():
    """主函数"""
    print("启动手语识别系统")
    print("=" * 30)

    # 查找文件
    models_dir = "data/models"
    processed_dir = "data/processed"

    model_path = None
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if model_files:
            model_path = os.path.join(models_dir, model_files[0])

    preprocessor_path = None
    if os.path.exists(processed_dir):
        processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.pkl')]
        if processed_files:
            preprocessor_path = os.path.join(processed_dir, sorted(processed_files)[-1])

    if not model_path or not preprocessor_path:
        print("缺少模型或数据文件，请先运行训练流程")
        return

    print(f"模型: {os.path.basename(model_path)}")
    print(f"数据: {os.path.basename(preprocessor_path)}")

    try:
        # 创建识别器
        recognizer = RealTimeGestureRecognizer(model_path, preprocessor_path)
        listener = GestureRecognitionListener(recognizer)

        connection = leap.Connection()
        connection.add_listener(listener)

        print("系统启动成功! 按'q'退出")

        with connection.open():
            connection.set_tracking_mode(leap.TrackingMode.Desktop)

            while True:
                cv2.imshow("Sign Language Recognition", recognizer.output_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

        # 显示统计
        print(f"\n统计: 尝试{recognizer.total_attempts}次, "
              f"成功{recognizer.successful_recognitions}次, "
              f"成功率{(recognizer.successful_recognitions / max(1, recognizer.total_attempts) * 100):.1f}%")

    except Exception as e:
        print(f"启动失败: {e}")


if __name__ == "__main__":
    main()
import json
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class HandGesturePreprocessor:
    """手语数据预处理器 - 用于处理原始Leap Motion数据"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, "raw")
        self.annotations_dir = os.path.join(data_dir, "annotations")
        self.processed_data_dir = os.path.join(data_dir, "processed")
        self.models_dir = os.path.join(data_dir, "models")

        # 创建处理后数据目录
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # 特征配置
        self.feature_config = {
            "sequence_length": 30,  # 固定序列长度
            "palm_features": True,
            "arm_features": True,
            "digit_features": True,
            "bone_features": True,
            "velocity_features": True,
            "angle_features": True,
            "distance_features": True
        }

        # 数据缩放器
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False

        # 标签编码
        self.label_encoder = {}
        self.label_decoder = {}

        # 统计信息
        self.stats = {}

    def load_raw_data(self) -> List[Dict]:
        """加载所有原始数据文件"""
        raw_data = []

        if not os.path.exists(self.raw_data_dir):
            print(f"原始数据目录不存在: {self.raw_data_dir}")
            return raw_data

        files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.json')]
        print(f"找到 {len(files)} 个原始数据文件")

        for filename in files:
            filepath = os.path.join(self.raw_data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['filename'] = filename
                    raw_data.append(data)
            except Exception as e:
                print(f"加载文件失败 {filename}: {e}")

        print(f"成功加载 {len(raw_data)} 个数据文件")
        return raw_data

    def extract_palm_features(self, palm_data: Dict) -> np.ndarray:
        """提取手掌特征"""
        features = []

        if self.feature_config["palm_features"]:
            # 位置特征
            features.extend(palm_data.get("position", [0, 0, 0]))
            # 方向特征
            features.extend(palm_data.get("direction", [0, 0, 0]))
            # 法向量特征
            features.extend(palm_data.get("normal", [0, 0, 0]))
            # 速度特征
            if self.feature_config["velocity_features"]:
                features.extend(palm_data.get("velocity", [0, 0, 0]))
            # 宽度特征
            features.append(palm_data.get("width", 0))

        return np.array(features)

    def extract_arm_features(self, arm_data: Dict) -> np.ndarray:
        """提取手臂特征"""
        features = []

        if self.feature_config["arm_features"]:
            # 关节位置
            features.extend(arm_data.get("prev_joint", [0, 0, 0]))
            features.extend(arm_data.get("next_joint", [0, 0, 0]))
            # 方向
            features.extend(arm_data.get("direction", [0, 0, 0]))
            # 长度和宽度
            features.append(arm_data.get("length", 0))
            features.append(arm_data.get("width", 0))

        return np.array(features)

    def extract_digit_features(self, digit_data: Dict) -> np.ndarray:
        """提取手指特征"""
        features = []

        if self.feature_config["digit_features"]:
            # 是否伸展
            features.append(float(digit_data.get("is_extended", False)))

            # 骨骼特征
            if self.feature_config["bone_features"]:
                bones = digit_data.get("bones", [])
                for bone in bones:
                    # 关节位置
                    features.extend(bone.get("prev_joint", [0, 0, 0]))
                    features.extend(bone.get("next_joint", [0, 0, 0]))
                    # 方向
                    features.extend(bone.get("direction", [0, 0, 0]))
                    # 长度和宽度
                    features.append(bone.get("length", 0))
                    features.append(bone.get("width", 0))

        return np.array(features)

    def calculate_angles(self, hand_data: Dict) -> np.ndarray:
        """计算手指间角度等几何特征"""
        angles = []

        if not self.feature_config["angle_features"]:
            return np.array(angles)

        digits = hand_data.get("digits", [])

        # 计算相邻手指间的角度
        for i in range(len(digits) - 1):
            digit1 = digits[i]
            digit2 = digits[i + 1]

            # 获取手指基部方向向量
            if digit1.get("bones") and digit2.get("bones"):
                bone1 = digit1["bones"][0]  # 掌骨
                bone2 = digit2["bones"][0]

                dir1 = np.array(bone1.get("direction", [0, 0, 0]))
                dir2 = np.array(bone2.get("direction", [0, 0, 0]))

                # 计算角度
                if np.linalg.norm(dir1) > 0 and np.linalg.norm(dir2) > 0:
                    cos_angle = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
                else:
                    angles.append(0)

        return np.array(angles)

    def calculate_distances(self, hand_data: Dict) -> np.ndarray:
        """计算关键点间距离"""
        distances = []

        if not self.feature_config["distance_features"]:
            return np.array(distances)

        palm_pos = np.array(hand_data.get("palm", {}).get("position", [0, 0, 0]))
        digits = hand_data.get("digits", [])

        # 计算手指尖到手掌的距离
        for digit in digits:
            bones = digit.get("bones", [])
            if bones:
                # 手指尖位置（最后一个骨骼的末端）
                fingertip_pos = np.array(bones[-1].get("next_joint", [0, 0, 0]))
                distance = np.linalg.norm(fingertip_pos - palm_pos)
                distances.append(distance)

        # 计算手指间距离
        fingertips = []
        for digit in digits:
            bones = digit.get("bones", [])
            if bones:
                fingertip_pos = np.array(bones[-1].get("next_joint", [0, 0, 0]))
                fingertips.append(fingertip_pos)

        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                distance = np.linalg.norm(fingertips[i] - fingertips[j])
                distances.append(distance)

        return np.array(distances)

    def extract_hand_features(self, hand_data: Dict) -> np.ndarray:
        """提取单只手的所有特征"""
        features = []

        # 基本信息
        hand_type = 1.0 if hand_data.get("hand_type") == "right" else 0.0
        features.append(hand_type)
        features.append(hand_data.get("confidence", 0))
        features.append(hand_data.get("grab_strength", 0))
        features.append(hand_data.get("grab_angle", 0))
        features.append(hand_data.get("pinch_distance", 0))
        features.append(hand_data.get("pinch_strength", 0))

        # 手掌特征
        palm_features = self.extract_palm_features(hand_data.get("palm", {}))
        features.extend(palm_features)

        # 手臂特征
        arm_features = self.extract_arm_features(hand_data.get("arm", {}))
        features.extend(arm_features)

        # 手指特征
        digits = hand_data.get("digits", [])
        for digit in digits:
            digit_features = self.extract_digit_features(digit)
            features.extend(digit_features)

        # 几何特征
        angle_features = self.calculate_angles(hand_data)
        features.extend(angle_features)

        distance_features = self.calculate_distances(hand_data)
        features.extend(distance_features)

        return np.array(features)

    def extract_frame_features(self, frame_data: Dict) -> np.ndarray:
        """提取单帧特征"""
        features = []

        hands = frame_data.get("hands", [])

        # 处理双手情况
        left_hand_features = np.zeros(200)  # 假设单手特征维度为200
        right_hand_features = np.zeros(200)

        for hand in hands:
            hand_features = self.extract_hand_features(hand)

            # 确保特征维度一致
            if len(hand_features) > 200:
                hand_features = hand_features[:200]
            elif len(hand_features) < 200:
                hand_features = np.pad(hand_features, (0, 200 - len(hand_features)))

            if hand.get("hand_type") == "left":
                left_hand_features = hand_features
            else:
                right_hand_features = hand_features

        # 合并双手特征
        features.extend(left_hand_features)
        features.extend(right_hand_features)

        # 添加手的数量信息
        features.append(len(hands))

        return np.array(features)

    def normalize_sequence_length(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """标准化序列长度"""
        current_length = len(sequence)

        if current_length == target_length:
            return sequence
        elif current_length < target_length:
            # 填充：重复最后一帧
            padding = np.repeat(sequence[-1:], target_length - current_length, axis=0)
            return np.vstack([sequence, padding])
        else:
            # 截断：等间隔采样
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return sequence[indices]

    def process_gesture_data(self, gesture_data: Dict) -> Tuple[np.ndarray, str, str, str]:
        """处理单个手势数据"""
        frames = gesture_data.get("frames", [])
        if not frames:
            return None, None, None, None

        # 提取每帧特征
        sequence_features = []
        for frame in frames:
            frame_features = self.extract_frame_features(frame)
            sequence_features.append(frame_features)

        # 转换为numpy数组
        sequence_features = np.array(sequence_features)

        # 标准化序列长度
        sequence_features = self.normalize_sequence_length(
            sequence_features, self.feature_config["sequence_length"]
        )

        # 获取标签
        gesture_label = gesture_data.get("gesture_label", "")
        chinese_meaning = gesture_data.get("chinese_meaning", "")
        english_meaning = gesture_data.get("english_meaning", "")

        return sequence_features, gesture_label, chinese_meaning, english_meaning

    def build_label_encoders(self, raw_data: List[Dict]):
        """构建标签编码器"""
        gesture_labels = set()
        chinese_meanings = set()
        english_meanings = set()

        for data in raw_data:
            gesture_labels.add(data.get("gesture_label", ""))
            chinese_meanings.add(data.get("chinese_meaning", ""))
            english_meanings.add(data.get("english_meaning", ""))

        # 创建编码映射
        self.label_encoder = {
            "gesture": {label: idx for idx, label in enumerate(sorted(gesture_labels))},
            "chinese": {label: idx for idx, label in enumerate(sorted(chinese_meanings))},
            "english": {label: idx for idx, label in enumerate(sorted(english_meanings))}
        }

        # 创建解码映射
        self.label_decoder = {
            "gesture": {idx: label for label, idx in self.label_encoder["gesture"].items()},
            "chinese": {idx: label for label, idx in self.label_encoder["chinese"].items()},
            "english": {idx: label for label, idx in self.label_encoder["english"].items()}
        }

        print(f"标签编码器构建完成:")
        print(f"  手势标签: {len(self.label_encoder['gesture'])}")
        print(f"  中文含义: {len(self.label_encoder['chinese'])}")
        print(f"  英文含义: {len(self.label_encoder['english'])}")

    def process_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """处理所有数据"""
        print("开始处理所有数据...")

        # 加载原始数据
        raw_data = self.load_raw_data()
        if not raw_data:
            raise ValueError("没有找到原始数据文件")

        print(f"找到 {len(raw_data)} 个原始数据文件")

        # 构建标签编码器
        self.build_label_encoders(raw_data)

        # 处理每个手势数据
        all_features = []
        all_gesture_labels = []
        all_chinese_labels = []
        all_english_labels = []

        valid_count = 0
        invalid_count = 0

        for i, data in enumerate(raw_data):
            try:
                features, gesture_label, chinese_meaning, english_meaning = self.process_gesture_data(data)

                if features is not None and features.size > 0:
                    all_features.append(features)
                    all_gesture_labels.append(self.label_encoder["gesture"][gesture_label])
                    all_chinese_labels.append(self.label_encoder["chinese"][chinese_meaning])
                    all_english_labels.append(self.label_encoder["english"][english_meaning])
                    valid_count += 1
                    print(f"处理文件 {i + 1}/{len(raw_data)}: 成功 - {chinese_meaning}({english_meaning})")
                else:
                    invalid_count += 1
                    print(f"处理文件 {i + 1}/{len(raw_data)}: 失败 - 无效数据")
            except Exception as e:
                invalid_count += 1
                print(f"处理文件 {i + 1}/{len(raw_data)}: 失败 - {e}")

        print(f"数据处理完成: 有效数据 {valid_count}, 无效数据 {invalid_count}")

        if valid_count == 0:
            raise ValueError("没有有效的数据可用于训练")

        # 检查数据收集建议
        if valid_count < 10:
            print("\n⚠️  数据不足警告:")
            print(f"   当前只有 {valid_count} 个有效样本")
            print("   建议:")
            print("   1. 每个手势至少收集 10-20 个样本")
            print("   2. 在不同光照条件下收集数据")
            print("   3. 用不同的手势速度收集数据")
            print("   4. 确保手势动作完整和清晰")
            print()

        # 转换为numpy数组
        X = np.array(all_features)
        y_gesture = np.array(all_gesture_labels)
        y_chinese = np.array(all_chinese_labels)
        y_english = np.array(all_english_labels)

        print(f"数据形状检查:")
        print(f"  特征矩阵: {X.shape}")
        print(f"  手势标签: {y_gesture.shape}")
        print(f"  中文标签: {y_chinese.shape}")
        print(f"  英文标签: {y_english.shape}")

        # 数据标准化
        if X.size > 0:
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])

            if not self.is_scaler_fitted:
                print("拟合数据缩放器...")
                X_scaled = self.scaler.fit_transform(X_reshaped)
                self.is_scaler_fitted = True
            else:
                X_scaled = self.scaler.transform(X_reshaped)

            X = X_scaled.reshape(original_shape)
            print("数据标准化完成")

        # 生成统计信息
        self.generate_statistics(X, y_gesture, y_chinese, y_english)

        return X, y_gesture, y_chinese, y_english

    def generate_statistics(self, X: np.ndarray, y_gesture: np.ndarray,
                            y_chinese: np.ndarray, y_english: np.ndarray):
        """生成数据统计信息"""
        self.stats = {
            "total_samples": len(X),
            "sequence_length": X.shape[1],
            "feature_dimension": X.shape[2],
            "num_gesture_classes": len(self.label_encoder["gesture"]),
            "num_chinese_classes": len(self.label_encoder["chinese"]),
            "num_english_classes": len(self.label_encoder["english"]),
            "feature_mean": np.mean(X),
            "feature_std": np.std(X),
            "feature_min": np.min(X),
            "feature_max": np.max(X),
            "class_distribution": {
                "gesture": {self.label_decoder["gesture"][i]: np.sum(y_gesture == i)
                            for i in range(len(self.label_decoder["gesture"]))},
                "chinese": {self.label_decoder["chinese"][i]: np.sum(y_chinese == i)
                            for i in range(len(self.label_decoder["chinese"]))},
                "english": {self.label_decoder["english"][i]: np.sum(y_english == i)
                            for i in range(len(self.label_decoder["english"]))}
            }
        }

        print("\n数据统计信息:")
        print(f"总样本数: {self.stats['total_samples']}")
        print(f"序列长度: {self.stats['sequence_length']}")
        print(f"特征维度: {self.stats['feature_dimension']}")
        print(f"手势类别数: {self.stats['num_gesture_classes']}")
        print(f"中文类别数: {self.stats['num_chinese_classes']}")
        print(f"英文类别数: {self.stats['num_english_classes']}")

    def split_data(self, X: np.ndarray, y_gesture: np.ndarray, y_chinese: np.ndarray,
                   y_english: np.ndarray, test_size: float = 0.2, val_size: float = 0.1):
        """分割数据集"""
        total_samples = len(X)

        # 检查样本数量
        if total_samples < 3:
            print(f"警告: 样本数量太少 ({total_samples})，无法进行数据分割")
            print("将所有数据用作训练集")
            return {
                "X_train": X, "y_gesture_train": y_gesture,
                "y_chinese_train": y_chinese, "y_english_train": y_english,
                "X_test": X, "y_gesture_test": y_gesture,
                "y_chinese_test": y_chinese, "y_english_test": y_english
            }

        # 调整分割比例以适应小样本
        min_test_samples = 1
        min_val_samples = 1 if val_size > 0 else 0

        # 确保测试集至少有1个样本
        effective_test_size = max(min_test_samples / total_samples, test_size)
        effective_test_size = min(effective_test_size, 0.5)  # 最多50%作为测试集

        # 确保验证集至少有1个样本（如果需要验证集）
        if val_size > 0:
            effective_val_size = max(min_val_samples / (total_samples * (1 - effective_test_size)), val_size)
            effective_val_size = min(effective_val_size, 0.3)  # 最多30%作为验证集
        else:
            effective_val_size = 0

        print(f"数据分割信息:")
        print(f"  总样本数: {total_samples}")
        print(f"  测试集比例: {effective_test_size:.2f}")
        print(f"  验证集比例: {effective_val_size:.2f}")

        try:
            # 检查是否有足够的每个类别的样本进行分层抽样
            unique_classes, class_counts = np.unique(y_gesture, return_counts=True)
            min_class_count = np.min(class_counts)

            if min_class_count < 2:
                print("警告: 某些类别样本数量太少，使用随机分割而非分层抽样")
                stratify = None
            else:
                stratify = y_gesture

            # 首先分离训练集和测试集
            X_train, X_test, y_gesture_train, y_gesture_test, y_chinese_train, y_chinese_test, y_english_train, y_english_test = train_test_split(
                X, y_gesture, y_chinese, y_english,
                test_size=effective_test_size,
                random_state=42,
                stratify=stratify
            )

            # 从训练集中分离验证集
            if effective_val_size > 0 and len(X_train) >= 2:
                # 重新计算验证集在剩余训练数据中的比例
                val_size_adjusted = effective_val_size / (1 - effective_test_size)

                # 检查是否能进行分层抽样
                unique_train_classes, train_class_counts = np.unique(y_gesture_train, return_counts=True)
                min_train_class_count = np.min(train_class_counts)

                if min_train_class_count < 2:
                    train_stratify = None
                else:
                    train_stratify = y_gesture_train

                try:
                    X_train, X_val, y_gesture_train, y_gesture_val, y_chinese_train, y_chinese_val, y_english_train, y_english_val = train_test_split(
                        X_train, y_gesture_train, y_chinese_train, y_english_train,
                        test_size=val_size_adjusted,
                        random_state=42,
                        stratify=train_stratify
                    )

                    result = {
                        "X_train": X_train, "y_gesture_train": y_gesture_train,
                        "y_chinese_train": y_chinese_train, "y_english_train": y_english_train,
                        "X_val": X_val, "y_gesture_val": y_gesture_val,
                        "y_chinese_val": y_chinese_val, "y_english_val": y_english_val,
                        "X_test": X_test, "y_gesture_test": y_gesture_test,
                        "y_chinese_test": y_chinese_test, "y_english_test": y_english_test
                    }

                    print(f"  训练集: {len(X_train)} 样本")
                    print(f"  验证集: {len(X_val)} 样本")
                    print(f"  测试集: {len(X_test)} 样本")

                    return result

                except ValueError as e:
                    print(f"验证集分割失败，跳过验证集: {e}")

            # 没有验证集的情况
            result = {
                "X_train": X_train, "y_gesture_train": y_gesture_train,
                "y_chinese_train": y_chinese_train, "y_english_train": y_english_train,
                "X_test": X_test, "y_gesture_test": y_gesture_test,
                "y_chinese_test": y_chinese_test, "y_english_test": y_english_test
            }

            print(f"  训练集: {len(X_train)} 样本")
            print(f"  测试集: {len(X_test)} 样本")
            print("  注意: 未创建验证集")

            return result

        except ValueError as e:
            print(f"数据分割失败: {e}")
            print("使用所有数据作为训练集和测试集")
            return {
                "X_train": X, "y_gesture_train": y_gesture,
                "y_chinese_train": y_chinese, "y_english_train": y_english,
                "X_test": X, "y_gesture_test": y_gesture,
                "y_chinese_test": y_chinese, "y_english_test": y_english
            }

    def save_processed_data(self, data_splits: Dict, filename: str = None):
        """保存处理后的数据"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_data_{timestamp}.pkl"

        filepath = os.path.join(self.processed_data_dir, filename)

        save_data = {
            "data_splits": data_splits,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "feature_config": self.feature_config,
            "stats": self.stats
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"处理后的数据已保存: {filepath}")
        return filepath

    def load_processed_data(self, filepath: str) -> Dict:
        """加载处理后的数据"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        self.scaler = save_data["scaler"]
        self.label_encoder = save_data["label_encoder"]
        self.label_decoder = save_data["label_decoder"]
        self.feature_config = save_data["feature_config"]
        self.stats = save_data["stats"]
        self.is_scaler_fitted = True

        print(f"处理后的数据已加载: {filepath}")
        return save_data["data_splits"]

    def visualize_data_distribution(self):
        """可视化数据分布"""
        if not self.stats:
            print("没有统计信息可供可视化")
            return

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 手势分布
        gesture_dist = self.stats["class_distribution"]["gesture"]
        if gesture_dist:
            labels = list(gesture_dist.keys())
            values = list(gesture_dist.values())

            bars1 = axes[0].bar(range(len(labels)), values, color='skyblue', alpha=0.8)
            axes[0].set_title("Gesture Label Distribution", fontsize=14, fontweight='bold')
            axes[0].set_xlabel("Gesture Labels", fontsize=12)
            axes[0].set_ylabel("Sample Count", fontsize=12)
            axes[0].set_xticks(range(len(labels)))
            axes[0].set_xticklabels(labels, rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars1, values):
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                             str(value), ha='center', va='bottom', fontweight='bold')

        # 中文含义分布
        chinese_dist = self.stats["class_distribution"]["chinese"]
        if chinese_dist:
            labels = list(chinese_dist.keys())
            values = list(chinese_dist.values())

            bars2 = axes[1].bar(range(len(labels)), values, color='lightcoral', alpha=0.8)
            axes[1].set_title("Chinese Meaning Distribution", fontsize=14, fontweight='bold')
            axes[1].set_xlabel("Chinese Meanings", fontsize=12)
            axes[1].set_ylabel("Sample Count", fontsize=12)
            axes[1].set_xticks(range(len(labels)))
            axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            axes[1].grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars2, values):
                axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                             str(value), ha='center', va='bottom', fontweight='bold')

        # 英文含义分布
        english_dist = self.stats["class_distribution"]["english"]
        if english_dist:
            labels = list(english_dist.keys())
            values = list(english_dist.values())

            bars3 = axes[2].bar(range(len(labels)), values, color='lightgreen', alpha=0.8)
            axes[2].set_title("English Meaning Distribution", fontsize=14, fontweight='bold')
            axes[2].set_xlabel("English Meanings", fontsize=12)
            axes[2].set_ylabel("Sample Count", fontsize=12)
            axes[2].set_xticks(range(len(labels)))
            axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            axes[2].grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars3, values):
                axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                             str(value), ha='center', va='bottom', fontweight='bold')

        # 调整布局
        plt.tight_layout()

        # 保存图表
        chart_path = os.path.join(self.processed_data_dir, "data_distribution.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')

        # 显示图表
        plt.show()

    def print_text_distribution(self):
        """打印文本版数据分布（当图形显示失败时使用）"""
        if not self.stats:
            print("没有统计信息可显示")
            return

        print("\n" + "=" * 60)
        print("📊 数据分布统计")
        print("=" * 60)

        # 总体统计
        print(f"📈 总体信息:")
        print(f"   总样本数: {self.stats['total_samples']}")
        print(f"   序列长度: {self.stats['sequence_length']}")
        print(f"   特征维度: {self.stats['feature_dimension']}")
        print(f"   手势类别数: {self.stats['num_gesture_classes']}")

        # 手势分布
        print(f"\n🤏 手势标签分布:")
        gesture_dist = self.stats["class_distribution"]["gesture"]
        for label, count in sorted(gesture_dist.items()):
            percentage = (count / self.stats['total_samples']) * 100
            bar = "█" * int(count * 20 / max(gesture_dist.values()))
            print(f"   {label:>3}: {count:>3} 样本 ({percentage:>5.1f}%) {bar}")

        # 中文含义分布
        print(f"\n🈲 中文含义分布:")
        chinese_dist = self.stats["class_distribution"]["chinese"]
        for label, count in sorted(chinese_dist.items()):
            percentage = (count / self.stats['total_samples']) * 100
            bar = "█" * int(count * 20 / max(chinese_dist.values()))
            print(f"   {label:>6}: {count:>3} 样本 ({percentage:>5.1f}%) {bar}")

        # 英文含义分布
        print(f"\n🔤 英文含义分布:")
        english_dist = self.stats["class_distribution"]["english"]
        for label, count in sorted(english_dist.items()):
            percentage = (count / self.stats['total_samples']) * 100
            bar = "█" * int(count * 20 / max(english_dist.values()))
            print(f"   {label:>12}: {count:>3} 样本 ({percentage:>5.1f}%) {bar}")

        # 数据平衡性分析
        values = list(gesture_dist.values())
        if values:
            max_count = max(values)
            min_count = min(values)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            print(f"\n⚖️  数据平衡性分析:")
            print(f"   最多样本数: {max_count}")
            print(f"   最少样本数: {min_count}")
            print(f"   不平衡比率: {imbalance_ratio:.2f}")

            if imbalance_ratio > 3:
                print("   ⚠️  数据不平衡较严重，建议收集更多少数类别的样本")
            elif imbalance_ratio > 2:
                print("   ⚠️  数据存在轻微不平衡")
            else:
                print("   ✅ 数据分布较为平衡")

        print("=" * 60)

        # 打印详细统计信息
        print("\n📊 详细数据分布:")
        print("-" * 50)

        print("手势标签分布:")
        for label, count in gesture_dist.items():
            percentage = (count / self.stats['total_samples']) * 100
            print(f"  {label}: {count} 样本 ({percentage:.1f}%)")

        print("\n中文含义分布:")
        for label, count in chinese_dist.items():
            percentage = (count / self.stats['total_samples']) * 100
            print(f"  {label}: {count} 样本 ({percentage:.1f}%)")

        print("\n英文含义分布:")
        for label, count in english_dist.items():
            percentage = (count / self.stats['total_samples']) * 100
            print(f"  {label}: {count} 样本 ({percentage:.1f}%)")

        # 检查数据平衡性
        values = list(gesture_dist.values())
        if values:
            max_count = max(values)
            min_count = min(values)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            print(f"\n📈 数据平衡性分析:")
            print(f"  最多样本数: {max_count}")
            print(f"  最少样本数: {min_count}")
            print(f"  不平衡比率: {imbalance_ratio:.2f}")

            if imbalance_ratio > 3:
                print("  ⚠️  数据不平衡较严重，建议收集更多少数类别的样本")
            elif imbalance_ratio > 2:
                print("  ⚠️  数据存在轻微不平衡")
            else:
                print("  ✅ 数据分布较为平衡")


def main():
    """主函数 - 演示数据预处理流程"""
    print("手语数据预处理器")
    print("=" * 50)

    # 创建预处理器
    preprocessor = HandGesturePreprocessor()

    # 检查是否有原始数据
    raw_data_files = []
    if os.path.exists(preprocessor.raw_data_dir):
        raw_data_files = [f for f in os.listdir(preprocessor.raw_data_dir) if f.endswith('.json')]

    if not raw_data_files:
        print("❌ 没有找到原始数据文件")
        print("\n📋 请先收集数据:")
        print("1. 运行 data_collector.py")
        print("2. 按数字键(0-9)录制至少10个不同的手势")
        print("3. 每个手势建议录制5-10次")
        print("4. 然后再运行此预处理程序")
        return

    print(f"✅ 找到 {len(raw_data_files)} 个原始数据文件")

    try:
        # 处理所有数据
        X, y_gesture, y_chinese, y_english = preprocessor.process_all_data()

        if len(X) == 0:
            print("❌ 没有有效的数据可以处理")
            return

        # 分割数据
        print("\n开始分割数据...")
        data_splits = preprocessor.split_data(X, y_gesture, y_chinese, y_english)

        # 保存处理后的数据
        save_path = preprocessor.save_processed_data(data_splits)

        # 可视化数据分布
        print("\n生成数据分布图...")
        try:
            preprocessor.visualize_data_distribution()
        except Exception as e:
            print(f"图形可视化失败: {e}")
            print("生成文本版数据分布...")
            preprocessor.print_text_distribution()

        print(f"\n✅ 数据预处理完成!")
        print(f"📊 数据统计:")
        print(f"   总样本数: {len(X)}")
        print(f"   序列长度: {X.shape[1]}")
        print(f"   特征维度: {X.shape[2]}")
        print(f"   手势类别: {len(preprocessor.label_encoder['gesture'])}")

        print(f"\n📁 数据集划分:")
        print(f"   训练集: {data_splits['X_train'].shape[0]} 样本")
        if 'X_val' in data_splits:
            print(f"   验证集: {data_splits['X_val'].shape[0]} 样本")
        print(f"   测试集: {data_splits['X_test'].shape[0]} 样本")

        print(f"\n💾 文件保存位置:")
        print(f"   处理后数据: {save_path}")
        print(f"   数据分布图: {os.path.join(preprocessor.processed_data_dir, 'data_distribution.png')}")

        print(f"\n🚀 下一步:")
        print("   现在可以运行 trainer.py 开始训练模型")

        if len(X) < 10:
            print(f"\n⚠️  建议:")
            print("   当前数据量较少，建议:")
            print("   1. 收集更多训练数据 (每个手势至少10-20个样本)")
            print("   2. 增加手势类别的多样性")
            print("   3. 在不同条件下收集数据")

    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        print(f"\n🔧 可能的解决方案:")
        print("1. 检查原始数据文件是否完整")
        print("2. 确保至少有1个有效的手势数据")
        print("3. 检查数据文件格式是否正确")
        print("4. 重新运行 data_collector.py 收集数据")


if __name__ == "__main__":
    main()
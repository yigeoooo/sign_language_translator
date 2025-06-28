import leap
import numpy as np
import cv2
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any


class HandGestureDataCollector:
    """手语数据收集器 - 用于收集Leap Motion手势数据并生成标注文件"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, "raw")
        self.annotations_dir = os.path.join(data_dir, "annotations")

        # 创建数据目录
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)

        # 可视化设置
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.recording_colour = (0, 0, 255)  # 红色表示录制中

        # 数据收集设置
        self.current_gesture = None
        self.is_recording = False
        self.frame_buffer = []
        self.max_frames_per_gesture = 60  # 每个手势最多录制60帧
        self.frame_count = 0

        # 手势标签设置
        self.gesture_labels = {}
        self.load_gesture_labels()

        # 输出图像
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)

    def load_gesture_labels(self):
        """加载手势标签配置"""
        labels_file = os.path.join(self.data_dir, "gesture_labels.json")

        # 默认手势标签（可以根据需要修改）
        default_labels = {
            "1": {"chinese": "1", "english": "one"},
            "2": {"chinese": "2", "english": "two"},
        }

        if os.path.exists(labels_file) and os.path.getsize(labels_file) > 0:
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    self.gesture_labels = json.load(f)
                print(f"成功加载手势标签配置: {len(self.gesture_labels)} 个手势")
            except (json.JSONDecodeError, Exception) as e:
                print(f"加载手势标签配置失败: {e}")
                print("使用默认配置...")
                self.gesture_labels = default_labels
                self.save_gesture_labels()
        else:
            print("手势标签配置文件不存在，创建默认配置...")
            self.gesture_labels = default_labels
            self.save_gesture_labels()

    def save_gesture_labels(self):
        """保存手势标签配置"""
        labels_file = os.path.join(self.data_dir, "gesture_labels.json")
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(self.gesture_labels, f, ensure_ascii=False, indent=2)

    def extract_hand_features(self, hand) -> Dict[str, Any]:
        """提取单只手的特征数据 - 兼容leapc-python-bindings"""
        try:
            hand_data = {
                "hand_type": "left" if hasattr(hand, 'type') and hand.type == leap.HandType.Left else "right",
                "confidence": getattr(hand, 'confidence', 1.0),
                "grab_strength": getattr(hand, 'grab_strength', 0.0),
                "grab_angle": getattr(hand, 'grab_angle', 0.0),
                "pinch_distance": getattr(hand, 'pinch_distance', 0.0),
                "pinch_strength": getattr(hand, 'pinch_strength', 0.0),
                "palm": self._extract_palm_features(hand),
                "arm": self._extract_arm_features(hand),
                "digits": []
            }

            # 提取手指数据
            if hasattr(hand, 'digits') and hand.digits:
                for digit_idx in range(min(5, len(hand.digits))):
                    digit = hand.digits[digit_idx]
                    digit_data = self._extract_digit_features(digit, digit_idx)
                    hand_data["digits"].append(digit_data)

            return hand_data

        except Exception as e:
            print(f"提取手部特征时出错: {e}")
            return self._get_default_hand_data()

    def _extract_palm_features(self, hand) -> Dict[str, Any]:
        """提取手掌特征"""
        try:
            palm = getattr(hand, 'palm', None)
            if palm:
                return {
                    "position": self._get_vector3(getattr(palm, 'position', None)),
                    "direction": self._get_vector3(getattr(palm, 'direction', None)),
                    "normal": self._get_vector3(getattr(palm, 'normal', None)),
                    "velocity": self._get_vector3(getattr(palm, 'velocity', None)),
                    "width": getattr(palm, 'width', 0.0)
                }
        except:
            pass

        return {
            "position": [0.0, 0.0, 0.0],
            "direction": [0.0, -1.0, 0.0],
            "normal": [0.0, 0.0, -1.0],
            "velocity": [0.0, 0.0, 0.0],
            "width": 0.0
        }

    def _extract_arm_features(self, hand) -> Dict[str, Any]:
        """提取手臂特征"""
        try:
            arm = getattr(hand, 'arm', None)
            if arm:
                return {
                    "prev_joint": self._get_vector3(getattr(arm, 'prev_joint', None)),
                    "next_joint": self._get_vector3(getattr(arm, 'next_joint', None)),
                    "direction": self._get_vector3(getattr(arm, 'direction', None)),
                    "length": getattr(arm, 'length', 0.0),
                    "width": getattr(arm, 'width', 0.0)
                }
        except:
            pass

        return {
            "prev_joint": [0.0, 0.0, 0.0],
            "next_joint": [0.0, 0.0, 0.0],
            "direction": [0.0, -1.0, 0.0],
            "length": 0.0,
            "width": 0.0
        }

    def _extract_digit_features(self, digit, digit_idx: int) -> Dict[str, Any]:
        """提取手指特征"""
        digit_data = {
            "digit_type": digit_idx,
            "is_extended": getattr(digit, 'is_extended', True),
            "bones": []
        }

        try:
            if hasattr(digit, 'bones') and digit.bones:
                for bone_idx in range(min(4, len(digit.bones))):
                    bone = digit.bones[bone_idx]
                    bone_data = self._extract_bone_features(bone, bone_idx)
                    digit_data["bones"].append(bone_data)
        except Exception as e:
            print(f"提取手指{digit_idx}特征时出错: {e}")

        # 确保有4个骨骼数据
        while len(digit_data["bones"]) < 4:
            digit_data["bones"].append(self._get_default_bone_data(len(digit_data["bones"])))

        return digit_data

    def _extract_bone_features(self, bone, bone_idx: int) -> Dict[str, Any]:
        """提取骨骼特征"""
        try:
            prev_joint = self._get_vector3(getattr(bone, 'prev_joint', None))
            next_joint = self._get_vector3(getattr(bone, 'next_joint', None))

            # 计算方向向量
            direction = [0.0, 0.0, 0.0]
            if hasattr(bone, 'direction'):
                direction = self._get_vector3(bone.direction)
            else:
                # 从关节位置计算方向
                direction = self._calculate_direction(prev_joint, next_joint)

            # 计算长度
            length = getattr(bone, 'length', 0.0)
            if length == 0.0:
                length = self._calculate_distance(prev_joint, next_joint)

            return {
                "bone_type": bone_idx,
                "prev_joint": prev_joint,
                "next_joint": next_joint,
                "direction": direction,
                "length": length,
                "width": getattr(bone, 'width', 0.0)
            }
        except Exception as e:
            print(f"提取骨骼{bone_idx}特征时出错: {e}")
            return self._get_default_bone_data(bone_idx)

    def _get_vector3(self, vector) -> List[float]:
        """安全获取3D向量"""
        if vector is None:
            return [0.0, 0.0, 0.0]

        try:
            if hasattr(vector, 'x') and hasattr(vector, 'y') and hasattr(vector, 'z'):
                return [float(vector.x), float(vector.y), float(vector.z)]
            elif isinstance(vector, (list, tuple)) and len(vector) >= 3:
                return [float(vector[0]), float(vector[1]), float(vector[2])]
        except:
            pass

        return [0.0, 0.0, 0.0]

    def _calculate_direction(self, start: List[float], end: List[float]) -> List[float]:
        """计算两点间的方向向量"""
        try:
            direction = [end[i] - start[i] for i in range(3)]
            length = sum(x * x for x in direction) ** 0.5
            if length > 0:
                return [x / length for x in direction]
        except:
            pass
        return [0.0, 0.0, 0.0]

    def _calculate_distance(self, start: List[float], end: List[float]) -> float:
        """计算两点间的距离"""
        try:
            return sum((end[i] - start[i]) ** 2 for i in range(3)) ** 0.5
        except:
            return 0.0

    def _get_default_hand_data(self) -> Dict[str, Any]:
        """获取默认手部数据"""
        return {
            "hand_type": "unknown",
            "confidence": 0.0,
            "grab_strength": 0.0,
            "grab_angle": 0.0,
            "pinch_distance": 0.0,
            "pinch_strength": 0.0,
            "palm": {
                "position": [0.0, 0.0, 0.0],
                "direction": [0.0, -1.0, 0.0],
                "normal": [0.0, 0.0, -1.0],
                "velocity": [0.0, 0.0, 0.0],
                "width": 0.0
            },
            "arm": {
                "prev_joint": [0.0, 0.0, 0.0],
                "next_joint": [0.0, 0.0, 0.0],
                "direction": [0.0, -1.0, 0.0],
                "length": 0.0,
                "width": 0.0
            },
            "digits": [self._get_default_digit_data(i) for i in range(5)]
        }

    def _get_default_digit_data(self, digit_idx: int) -> Dict[str, Any]:
        """获取默认手指数据"""
        return {
            "digit_type": digit_idx,
            "is_extended": True,
            "bones": [self._get_default_bone_data(i) for i in range(4)]
        }

    def _get_default_bone_data(self, bone_idx: int) -> Dict[str, Any]:
        """获取默认骨骼数据"""
        return {
            "bone_type": bone_idx,
            "prev_joint": [0.0, 0.0, 0.0],
            "next_joint": [0.0, 0.0, 0.0],
            "direction": [0.0, 0.0, 0.0],
            "length": 0.0,
            "width": 0.0
        }

    def extract_frame_features(self, event) -> Dict[str, Any]:
        """提取单帧的特征数据"""
        try:
            frame_data = {
                "timestamp": time.time(),
                "frame_id": getattr(event, 'tracking_frame_id', 0),
                "hands": []
            }

            for hand in event.hands:
                hand_features = self.extract_hand_features(hand)
                if hand_features:  # 只添加成功提取的手部数据
                    frame_data["hands"].append(hand_features)

            return frame_data

        except Exception as e:
            print(f"提取帧特征时出错: {e}")
            return {
                "timestamp": time.time(),
                "frame_id": 0,
                "hands": []
            }

    def start_recording(self, gesture_key: str):
        """开始录制手势"""
        if gesture_key in self.gesture_labels:
            self.current_gesture = gesture_key
            self.is_recording = True
            self.frame_buffer = []
            self.frame_count = 0
            print(
                f"开始录制手势: {self.gesture_labels[gesture_key]['chinese']} ({self.gesture_labels[gesture_key]['english']})")

    def stop_recording(self):
        """停止录制并保存数据"""
        if self.is_recording and self.frame_buffer:
            self.save_gesture_data()
            self.is_recording = False
            self.current_gesture = None
            self.frame_buffer = []
            self.frame_count = 0
            print("录制完成并保存")

    def save_gesture_data(self):
        """保存手势数据"""
        if not self.frame_buffer:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gesture_info = self.gesture_labels[self.current_gesture]

        # 保存原始数据
        raw_filename = f"{gesture_info['english']}_{gesture_info['chinese']}_{timestamp}.json"
        raw_filepath = os.path.join(self.raw_data_dir, raw_filename)

        gesture_data = {
            "gesture_label": self.current_gesture,
            "chinese_meaning": gesture_info['chinese'],
            "english_meaning": gesture_info['english'],
            "timestamp": timestamp,
            "frame_count": len(self.frame_buffer),
            "frames": self.frame_buffer
        }

        with open(raw_filepath, 'w', encoding='utf-8') as f:
            json.dump(gesture_data, f, ensure_ascii=False, indent=2)

        # 生成标注文件
        self.generate_annotation(gesture_data, raw_filename)

        print(f"数据已保存: {raw_filepath}")

    def generate_annotation(self, gesture_data: Dict, raw_filename: str):
        """生成标注文件"""
        annotation = {
            "data_file": raw_filename,
            "gesture_label": gesture_data["gesture_label"],
            "chinese_meaning": gesture_data["chinese_meaning"],
            "english_meaning": gesture_data["english_meaning"],
            "timestamp": gesture_data["timestamp"],
            "frame_count": gesture_data["frame_count"],
            "data_quality": "good",  # 可以后续手动调整
            "notes": ""
        }

        annotation_filename = raw_filename.replace('.json', '_annotation.json')
        annotation_filepath = os.path.join(self.annotations_dir, annotation_filename)

        with open(annotation_filepath, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

    def get_joint_position(self, joint):
        """获取关节在屏幕上的位置"""
        if joint:
            return int(joint.x + (self.screen_size[1] / 2)), int(joint.z + (self.screen_size[0] / 2))
        else:
            return None

    def render_hands(self, event):
        """渲染手部可视化"""
        # 清除之前的图像
        self.output_image[:, :] = 0

        # 显示录制状态
        if self.is_recording:
            cv2.putText(
                self.output_image,
                f"RECORDING: {self.gesture_labels[self.current_gesture]['chinese']} ({self.frame_count}/{self.max_frames_per_gesture})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.recording_colour,
                2,
            )

        # 显示可用手势
        y_offset = 60
        cv2.putText(
            self.output_image,
            "Available Gestures:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.font_colour,
            1,
        )

        for key, gesture in self.gesture_labels.items():
            y_offset += 20
            text = f"{key}: {gesture['chinese']} ({gesture['english']})"
            cv2.putText(
                self.output_image,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.font_colour,
                1,
            )

        # 显示控制说明
        instructions = [
            "Press number key to start recording",
            "Press SPACE to stop recording",
            "Press 'q' to quit"
        ]

        y_offset += 40
        for instruction in instructions:
            y_offset += 20
            cv2.putText(
                self.output_image,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

        # 绘制手部骨架
        if len(event.hands) > 0:
            for hand in event.hands:
                # 绘制手臂
                wrist = self.get_joint_position(hand.arm.next_joint)
                elbow = self.get_joint_position(hand.arm.prev_joint)
                if wrist:
                    cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)
                if elbow:
                    cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)
                if wrist and elbow:
                    cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

                # 绘制手指
                for digit_idx in range(5):
                    digit = hand.digits[digit_idx]
                    for bone_idx in range(4):
                        bone = digit.bones[bone_idx]
                        bone_start = self.get_joint_position(bone.prev_joint)
                        bone_end = self.get_joint_position(bone.next_joint)

                        if bone_start:
                            cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)
                        if bone_end:
                            cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)
                        if bone_start and bone_end:
                            cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)

                        # 连接手指间的骨骼
                        if ((digit_idx == 0) and (bone_idx == 0)) or (
                                (digit_idx > 0) and (digit_idx < 4) and (bone_idx < 2)
                        ):
                            if digit_idx < 4:
                                next_digit = hand.digits[digit_idx + 1]
                                next_bone = next_digit.bones[bone_idx]
                                next_bone_start = self.get_joint_position(next_bone.prev_joint)
                                if bone_start and next_bone_start:
                                    cv2.line(self.output_image, bone_start, next_bone_start, self.hands_colour, 2)

                        # 连接到手腕
                        if bone_idx == 0 and bone_start and wrist:
                            cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)


class DataCollectionListener(leap.Listener):
    """数据收集监听器"""

    def __init__(self, collector: HandGestureDataCollector):
        self.collector = collector

    def on_connection_event(self, event):
        print("Leap Motion连接成功")

    def on_tracking_mode_event(self, event):
        print(f"跟踪模式: {event.current_tracking_mode}")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        print(f"发现设备: {info.serial}")

    def on_tracking_event(self, event):
        # 渲染可视化
        self.collector.render_hands(event)

        # 如果正在录制，收集数据
        if self.collector.is_recording:
            if self.collector.frame_count < self.collector.max_frames_per_gesture:
                frame_data = self.collector.extract_frame_features(event)
                self.collector.frame_buffer.append(frame_data)
                self.collector.frame_count += 1
            else:
                # 达到最大帧数，自动停止录制
                self.collector.stop_recording()


def main():
    """主函数"""
    print("手语数据收集器启动")
    print("=" * 50)

    collector = HandGestureDataCollector()
    listener = DataCollectionListener(collector)

    connection = leap.Connection()
    connection.add_listener(listener)

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)

        print("数据收集器已准备就绪")
        print("按数字键开始录制对应手势，按空格键停止录制，按q键退出")

        while True:
            cv2.imshow("Hand Gesture Data Collector", collector.output_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键停止录制
                if collector.is_recording:
                    collector.stop_recording()
            elif chr(key) in collector.gesture_labels:  # 数字键开始录制
                gesture_key = chr(key)
                if not collector.is_recording:
                    collector.start_recording(gesture_key)
                else:
                    print("当前正在录制，请先停止当前录制")

    cv2.destroyAllWindows()
    print("数据收集器已关闭")


if __name__ == "__main__":
    main()
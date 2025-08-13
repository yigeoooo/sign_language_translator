# -*- coding: utf-8 -*- 
# @Time    : 2025/8/13 19:23
# @Author  : yigeoooo
# @FileName: ui.py
# @Software: PyCharm
import cv2
import numpy as np
import leap
import time
import os
from typing import Dict, Optional
from inference import GestureRecognitionCore


class GestureRecognitionUI:
    """手语识别UI界面类"""

    def __init__(self, core: GestureRecognitionCore):
        self.core = core
        self.core.set_status_callback(self.on_status_updated)

        # 界面设置
        self.screen_size = [600, 800]
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)

        # UI颜色配置
        self.colors = {
            "SUCCESS": (0, 255, 0),
            "FAILED": (0, 0, 255),
            "COLLECTING": (0, 255, 255),
            "WAITING": (128, 128, 128),
            "TEXT": (255, 255, 255),
            "TITLE": (0, 255, 0),
            "PROGRESS": (0, 255, 255),
            "SKELETON": (255, 255, 255)
        }

        # 字体设置
        self.fonts = {
            "title": cv2.FONT_HERSHEY_SIMPLEX,
            "normal": cv2.FONT_HERSHEY_SIMPLEX,
            "small": cv2.FONT_HERSHEY_SIMPLEX
        }

        # UI布局
        self.layout = {
            "title_y": 40,
            "stats_start_y": 80,
            "stats_line_height": 25,
            "status_area_height": 200,
            "status_area_margin": 20
        }

    def on_status_updated(self, status: str, result: Optional[Dict]):
        """状态更新回调"""
        # 这里可以添加状态更新时的特殊处理
        pass

    def render_interface(self, event):
        """渲染界面"""
        # 清空画布
        self.output_image[:] = 0

        # 渲染各个部分
        self._render_title()
        self._render_stats()
        self._render_status_area()
        self._render_skeleton(event)
        self._render_controls()

    def _render_title(self):
        """渲染标题"""
        cv2.putText(self.output_image, "Sign Language Recognition",
                    (20, self.layout["title_y"]),
                    self.fonts["title"], 0.8, self.colors["TITLE"], 2)

    def _render_stats(self):
        """渲染统计信息"""
        stats = self.core.get_stats()
        y = self.layout["stats_start_y"]

        stats_text = [
            f"Attempts: {stats['total_attempts']}",
            f"Success: {stats['successful_recognitions']}",
            f"Rate: {stats['success_rate']:.1f}%",
            f"Buffer: {stats['buffer_length']}/{stats['max_buffer_length']}"
        ]

        for stat in stats_text:
            cv2.putText(self.output_image, stat, (20, y),
                        self.fonts["small"], 0.5, self.colors["TEXT"], 1)
            y += self.layout["stats_line_height"]

    def _render_status_area(self):
        """渲染状态区域"""
        status_info = self.core.get_current_status()
        status = status_info['status']
        result = status_info['result']

        # 计算状态区域位置
        status_y = self.layout["stats_start_y"] + 4 * self.layout["stats_line_height"] + 20
        status_height = self.layout["status_area_height"]
        margin = self.layout["status_area_margin"]

        # 状态颜色
        color = self.colors.get(status, self.colors["WAITING"])

        # 状态框
        cv2.rectangle(self.output_image,
                      (margin, status_y),
                      (self.screen_size[1] - margin, status_y + status_height),
                      color, 2)

        # 状态内容
        if status == "SUCCESS" and result:
            self._render_success_status(status_y, margin, color, result)
        elif status == "FAILED":
            self._render_failed_status(status_y, margin, color)
        elif status == "COLLECTING":
            self._render_collecting_status(status_y, margin, color)
        else:  # WAITING
            self._render_waiting_status(status_y, margin, color)

    def _render_success_status(self, status_y: int, margin: int, color: tuple, result: Dict):
        """渲染成功状态"""
        cv2.putText(self.output_image, "RECOGNITION SUCCESS!",
                    (margin + 10, status_y + 40),
                    self.fonts["normal"], 0.8, color, 2)

        cv2.putText(self.output_image, f"Gesture: {result['gesture_label']}",
                    (margin + 10, status_y + 70),
                    self.fonts["normal"], 0.6, self.colors["TEXT"], 2)

        cv2.putText(self.output_image, f"Chinese: {result['chinese_meaning']}",
                    (margin + 10, status_y + 100),
                    self.fonts["normal"], 0.6, self.colors["TEXT"], 2)

        cv2.putText(self.output_image, f"English: {result['english_meaning']}",
                    (margin + 10, status_y + 130),
                    self.fonts["normal"], 0.6, self.colors["TEXT"], 2)

        cv2.putText(self.output_image, f"Confidence: {result['confidence']:.2f}",
                    (margin + 10, status_y + 160),
                    self.fonts["small"], 0.5, self.colors["TEXT"], 1)

    def _render_failed_status(self, status_y: int, margin: int, color: tuple):
        """渲染失败状态"""
        cv2.putText(self.output_image, "RECOGNITION FAILED!",
                    (margin + 10, status_y + 40),
                    self.fonts["normal"], 0.8, color, 2)
        cv2.putText(self.output_image, "Try making a clearer gesture",
                    (margin + 10, status_y + 80),
                    self.fonts["small"], 0.5, self.colors["TEXT"], 1)

    def _render_collecting_status(self, status_y: int, margin: int, color: tuple):
        """渲染采集状态"""
        cv2.putText(self.output_image, "COLLECTING GESTURE...",
                    (margin + 10, status_y + 40),
                    self.fonts["normal"], 0.8, color, 2)

        # 进度条
        stats = self.core.get_stats()
        progress = stats['buffer_length'] / stats['max_buffer_length']
        bar_width = int((self.screen_size[1] - 2 * margin - 20) * progress)

        cv2.rectangle(self.output_image,
                      (margin + 10, status_y + 70),
                      (margin + 10 + bar_width, status_y + 90),
                      self.colors["PROGRESS"], -1)
        cv2.rectangle(self.output_image,
                      (margin + 10, status_y + 70),
                      (self.screen_size[1] - margin - 10, status_y + 90),
                      self.colors["TEXT"], 1)

        cv2.putText(self.output_image, f"Progress: {progress * 100:.0f}%",
                    (margin + 10, status_y + 110),
                    self.fonts["small"], 0.5, self.colors["TEXT"], 1)

    def _render_waiting_status(self, status_y: int, margin: int, color: tuple):
        """渲染等待状态"""
        cv2.putText(self.output_image, "WAITING FOR GESTURE...",
                    (margin + 10, status_y + 40),
                    self.fonts["normal"], 0.8, color, 2)
        cv2.putText(self.output_image, "Place hand above sensor",
                    (margin + 10, status_y + 80),
                    self.fonts["small"], 0.5, self.colors["TEXT"], 1)

    def _render_skeleton(self, event):
        """渲染手部骨架"""
        if hasattr(event, 'hands') and event.hands:
            # 更新采集状态
            self.core.update_collecting_status(True)

            for hand in event.hands:
                self._draw_hand_skeleton(hand)
        else:
            # 更新采集状态
            self.core.update_collecting_status(False)

    def _render_controls(self):
        """渲染控制说明"""
        cv2.putText(self.output_image, "Press 'q' to quit",
                    (20, self.screen_size[0] - 20),
                    self.fonts["small"], 0.5, self.colors["PROGRESS"], 1)

    def _draw_hand_skeleton(self, hand):
        """绘制手部骨架"""
        try:
            # 手臂
            wrist = self._get_joint_position(hand.arm.next_joint)
            elbow = self._get_joint_position(hand.arm.prev_joint)

            if wrist and elbow:
                cv2.line(self.output_image, wrist, elbow, self.colors["SKELETON"], 2)
                cv2.circle(self.output_image, wrist, 4, self.colors["SKELETON"], -1)
                cv2.circle(self.output_image, elbow, 4, self.colors["SKELETON"], -1)

            # 手指
            for digit in hand.digits:
                for bone in digit.bones:
                    start = self._get_joint_position(bone.prev_joint)
                    end = self._get_joint_position(bone.next_joint)

                    if start and end:
                        cv2.line(self.output_image, start, end, self.colors["SKELETON"], 2)
                        cv2.circle(self.output_image, start, 3, self.colors["SKELETON"], -1)
                        cv2.circle(self.output_image, end, 3, self.colors["SKELETON"], -1)
        except:
            pass

    def _get_joint_position(self, joint):
        """获取关节屏幕位置"""
        if joint:
            x = int(joint.x + (self.screen_size[1] / 2))
            y = int(joint.z + (self.screen_size[0] / 2))
            return (x, y)
        return None

    def show(self):
        """显示界面"""
        cv2.imshow("Sign Language Recognition", self.output_image)

    def set_colors(self, color_config: Dict):
        """设置颜色配置"""
        self.colors.update(color_config)

    def set_layout(self, layout_config: Dict):
        """设置布局配置"""
        self.layout.update(layout_config)

    def set_fonts(self, font_config: Dict):
        """设置字体配置"""
        self.fonts.update(font_config)


class HandDataExtractor:
    """手部数据提取器"""

    @staticmethod
    def extract_hand_data(hand):
        """提取手部数据（保证返回合法结构，避免 None 导致丢失手）"""
        try:
            hand_data = {
                "hand_type": "left" if hasattr(hand, 'type') and hand.type == leap.HandType.Left else "right",
                "confidence": getattr(hand, 'confidence', 1.0),
                "grab_strength": getattr(hand, 'grab_strength', 0.0),
                "grab_angle": getattr(hand, 'grab_angle', 0.0),
                "pinch_distance": getattr(hand, 'pinch_distance', 0.0),
                "pinch_strength": getattr(hand, 'pinch_strength', 0.0),
                "palm": {
                    "position": HandDataExtractor._get_vector3(getattr(getattr(hand, 'palm', None), 'position', None)),
                    "direction": HandDataExtractor._get_vector3(
                        getattr(getattr(hand, 'palm', None), 'direction', None)),
                    "normal": HandDataExtractor._get_vector3(getattr(getattr(hand, 'palm', None), 'normal', None)),
                    "velocity": HandDataExtractor._get_vector3(getattr(getattr(hand, 'palm', None), 'velocity', None)),
                    "width": getattr(getattr(hand, 'palm', None), 'width', 0.0)
                },
                "arm": {
                    "prev_joint": HandDataExtractor._get_vector3(
                        getattr(getattr(hand, 'arm', None), 'prev_joint', None)),
                    "next_joint": HandDataExtractor._get_vector3(
                        getattr(getattr(hand, 'arm', None), 'next_joint', None)),
                    "direction": HandDataExtractor._get_vector3(getattr(getattr(hand, 'arm', None), 'direction', None)),
                    "length": getattr(getattr(hand, 'arm', None), 'length', 0.0),
                    "width": getattr(getattr(hand, 'arm', None), 'width', 0.0)
                },
                "digits": HandDataExtractor._extract_digits(hand)
            }
            return hand_data

        except Exception as e:
            print(f"提取手部数据出错: {e}")
            # 返回一个空但合法的结构，避免 None 导致手部被忽略
            return {
                "hand_type": "right",
                "confidence": 0.0,
                "grab_strength": 0.0,
                "grab_angle": 0.0,
                "pinch_distance": 0.0,
                "pinch_strength": 0.0,
                "palm": {
                    "position": [0.0, 0.0, 0.0],
                    "direction": [0.0, 0.0, 0.0],
                    "normal": [0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "width": 0.0
                },
                "arm": {
                    "prev_joint": [0.0, 0.0, 0.0],
                    "next_joint": [0.0, 0.0, 0.0],
                    "direction": [0.0, 0.0, 0.0],
                    "length": 0.0,
                    "width": 0.0
                },
                "digits": []
            }

    @staticmethod
    def _extract_digits(hand):
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
                                "prev_joint": HandDataExtractor._get_vector3(getattr(bone, 'prev_joint', None)),
                                "next_joint": HandDataExtractor._get_vector3(getattr(bone, 'next_joint', None)),
                                "direction": HandDataExtractor._get_vector3(getattr(bone, 'direction', None)),
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

    @staticmethod
    def _get_vector3(vector):
        """安全获取3D向量"""
        if vector is None:
            return [0.0, 0.0, 0.0]
        try:
            if hasattr(vector, 'x') and hasattr(vector, 'y') and hasattr(vector, 'z'):
                return [float(vector.x), float(vector.y), float(vector.z)]
        except:
            pass
        return [0.0, 0.0, 0.0]


class GestureRecognitionListener(leap.Listener):
    """Leap Motion监听器"""

    def __init__(self, core: GestureRecognitionCore, ui: GestureRecognitionUI):
        self.core = core
        self.ui = ui

    def on_connection_event(self, event):
        print("Leap Motion连接成功")

    def on_tracking_event(self, event):
        try:
            # 渲染界面
            self.ui.render_interface(event)

            # 构建帧数据
            frame_data = {'timestamp': time.time(), 'hands': []}

            # 调试：打印 event.hands 数量
            print(f"[DEBUG] event.hands 数量: {len(getattr(event, 'hands', []))}")

            # 提取手部数据
            if hasattr(event, 'hands') and event.hands:
                for hand in event.hands:
                    hand_data = HandDataExtractor.extract_hand_data(hand)
                    if hand_data:  # 即使异常，也会返回合法结构
                        frame_data['hands'].append(hand_data)

            # 调试：打印 frame_data['hands'] 数量
            print(f"[DEBUG] frame_data.hands 数量: {len(frame_data['hands'])}")

            # 处理预测
            result = self.core.process_prediction(frame_data)
            if result:
                self.core.update_result(result)

        except Exception as e:
            print(f"处理跟踪事件出错: {e}")


class RealTimeGestureRecognizer:
    """实时手语识别器（主应用类）"""

    def __init__(self, model_path: str, preprocessor_path: str):
        # 创建核心逻辑
        self.core = GestureRecognitionCore(model_path, preprocessor_path)

        # 创建UI
        self.ui = GestureRecognitionUI(self.core)

        # 创建监听器
        self.listener = GestureRecognitionListener(self.core, self.ui)

    def run(self):
        """运行识别系统"""
        connection = leap.Connection()
        connection.add_listener(self.listener)

        print("系统启动成功! 按'q'退出")

        with connection.open():
            connection.set_tracking_mode(leap.TrackingMode.Desktop)

            while True:
                self.ui.show()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

        # 显示统计
        stats = self.core.get_stats()
        print(f"\n统计: 尝试{stats['total_attempts']}次, "
              f"成功{stats['successful_recognitions']}次, "
              f"成功率{stats['success_rate']:.1f}%")

    def customize_ui(self, colors: Dict = None, layout: Dict = None, fonts: Dict = None):
        """自定义UI样式"""
        if colors:
            self.ui.set_colors(colors)
        if layout:
            self.ui.set_layout(layout)
        if fonts:
            self.ui.set_fonts(fonts)


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

        # 可以自定义UI样式（示例）
        # custom_colors = {
        #     "SUCCESS": (0, 255, 0),
        #     "FAILED": (0, 0, 255),
        #     "COLLECTING": (255, 255, 0),  # 黄色
        #     "WAITING": (128, 128, 128)
        # }
        # recognizer.customize_ui(colors=custom_colors)

        # 运行系统
        recognizer.run()

    except Exception as e:
        print(f"启动失败: {e}")


if __name__ == "__main__":
    main()
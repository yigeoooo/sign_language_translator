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
import math
from typing import Dict, Optional
from inference import GestureRecognitionCore


class ModernUIRenderer:
    """现代化UI渲染器 - Element UI风格"""

    @staticmethod
    def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
        """绘制圆角矩形"""
        x1, y1 = pt1
        x2, y2 = pt2

        # 确保坐标正确
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # 限制圆角半径
        radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

        if thickness == -1:  # 填充
            # 绘制主体矩形
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)

            # 绘制四个圆角
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        else:  # 边框
            # 绘制四条边
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

            # 绘制四个圆角边框
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    @staticmethod
    def draw_progress_bar(img, pt1, pt2, progress, bg_color, fill_color, border_color=None):
        """绘制现代化进度条"""
        x1, y1 = pt1
        x2, y2 = pt2
        height = y2 - y1

        # 背景
        ModernUIRenderer.draw_rounded_rect(img, pt1, pt2, bg_color, -1, height // 2)

        # 进度填充
        if progress > 0:
            fill_width = int((x2 - x1) * progress)
            fill_pt2 = (x1 + fill_width, y2)
            ModernUIRenderer.draw_rounded_rect(img, pt1, fill_pt2, fill_color, -1, height // 2)

        # 边框
        if border_color:
            ModernUIRenderer.draw_rounded_rect(img, pt1, pt2, border_color, 1, height // 2)


class GestureRecognitionUI:
    """手语识别UI界面类 - Element UI风格"""

    def __init__(self, core: GestureRecognitionCore):
        self.core = core
        self.core.set_status_callback(self.on_status_updated)

        # 界面设置 - 恢复到原始布局但增加手指区域
        self.screen_size = [800, 1000]
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)

        # 手指显示区域配置
        self.skeleton_area_height = 250
        self.skeleton_area_start = self.screen_size[0] - self.skeleton_area_height - 20

        # Element UI风格配色
        self.colors = {
            # 主题色
            "PRIMARY": (64, 158, 255),  # Element UI蓝色
            "SUCCESS": (103, 194, 58),  # 成功绿色
            "WARNING": (230, 162, 60),  # 警告橙色
            "DANGER": (245, 108, 108),  # 危险红色
            "INFO": (144, 147, 153),  # 信息灰色

            # 文字色
            "TEXT_PRIMARY": (48, 49, 51),  # 主要文字
            "TEXT_REGULAR": (96, 98, 102),  # 常规文字
            "TEXT_SECONDARY": (144, 147, 153),  # 次要文字

            # 背景色
            "BG_WHITE": (255, 255, 255),  # 白色背景
            "BG_BASE": (248, 249, 250),  # 基础背景
            "BG_LIGHT": (245, 247, 250),  # 浅色背景
            "BG_DARK": (30, 30, 30),  # 深色背景（手指区域）

            # 边框色
            "BORDER_BASE": (220, 223, 230),  # 基础边框
            "BORDER_LIGHT": (228, 231, 237),  # 浅色边框

            # 状态映射
            "WAITING": (144, 147, 153),
            "COLLECTING": (230, 162, 60),
            "SUCCESS_STATUS": (103, 194, 58),
            "FAILED": (245, 108, 108),

            # 骨架颜色（恢复原始配色，但在黑色背景上清晰）
            "SKELETON": (0, 255, 255),  # 青色
            "SKELETON_JOINT": (255, 255, 0),  # 黄色关节
            "SKELETON_ARM": (255, 0, 255)  # 紫色手臂
        }

        # 字体设置
        self.fonts = {
            "title": cv2.FONT_HERSHEY_DUPLEX,
            "subtitle": cv2.FONT_HERSHEY_SIMPLEX,
            "normal": cv2.FONT_HERSHEY_SIMPLEX,
            "small": cv2.FONT_HERSHEY_SIMPLEX,
            "mono": cv2.FONT_HERSHEY_TRIPLEX
        }

        # Element UI风格布局
        self.layout = {
            "padding": 20,
            "card_margin": 16,
            "card_padding": 20,
            "card_radius": 8,
            "header_height": 120,
            "stats_height": 100,
            "status_height": 250,  # 减小状态区域为手指腾出空间
            "line_height": 28,
            "section_spacing": 20
        }

        # 防闪烁
        self.last_status = None
        self.last_result = None
        self.last_stats = None
        self.needs_redraw = True

        # 动画
        self.animation = {
            "pulse_phase": 0,
            "last_update": time.time()
        }

    def on_status_updated(self, status: str, result: Optional[Dict]):
        """状态更新回调"""
        if status != self.last_status or result != self.last_result:
            self.needs_redraw = True
            self.last_status = status
            self.last_result = result

    def render_interface(self, event):
        """渲染界面"""
        self._update_animation()

        # 检查是否需要重绘
        current_stats = self.core.get_stats()
        if current_stats != self.last_stats:
            self.needs_redraw = True
            self.last_stats = current_stats.copy()

        # 重绘整个界面（但优化重绘频率）
        if self.needs_redraw:
            self._render_complete_interface(event)
            self.needs_redraw = False
        else:
            # 只更新手指区域
            self._update_skeleton_area(event)

    def _update_animation(self):
        """更新动画"""
        current_time = time.time()
        dt = current_time - self.animation["last_update"]
        self.animation["last_update"] = current_time
        self.animation["pulse_phase"] = (self.animation["pulse_phase"] + dt * 3) % (2 * math.pi)

    def _render_complete_interface(self, event):
        """渲染完整界面"""
        # 清空整个画布
        self.output_image[:] = self.colors["BG_BASE"]

        # 渲染主要组件
        self._render_header()
        self._render_stats_card()
        self._render_status_card()
        self._render_skeleton_area_with_border()

        # 渲染手指
        self._render_skeleton(event)

    def _update_skeleton_area(self, event):
        """只更新手指骨架区域"""
        # 清空手指区域
        cv2.rectangle(self.output_image,
                      (20, self.skeleton_area_start + 30),
                      (self.screen_size[1] - 20, self.screen_size[0] - 20),
                      self.colors["BG_DARK"], -1)

        # 重新绘制手指
        self._render_skeleton(event)

    def _render_header(self):
        """渲染头部"""
        padding = self.layout["padding"]
        height = self.layout["header_height"]

        # 头部卡片
        ModernUIRenderer.draw_rounded_rect(
            self.output_image,
            (padding, padding),
            (self.screen_size[1] - padding, height),
            self.colors["BG_WHITE"], -1, self.layout["card_radius"]
        )

        ModernUIRenderer.draw_rounded_rect(
            self.output_image,
            (padding, padding),
            (self.screen_size[1] - padding, height),
            self.colors["BORDER_LIGHT"], 1, self.layout["card_radius"]
        )

        # 标题
        cv2.putText(self.output_image, "Sign Language Recognition",
                    (padding + 30, padding + 40),
                    self.fonts["title"], 1.2, self.colors["TEXT_PRIMARY"], 2)

        # 副标题
        cv2.putText(self.output_image, "Real-time AI-powered gesture detection system",
                    (padding + 30, padding + 70),
                    self.fonts["normal"], 0.6, self.colors["TEXT_SECONDARY"], 1)

        # 状态指示点
        status_info = self.core.get_current_status()
        status_color = {
            "SUCCESS": self.colors["SUCCESS"],
            "FAILED": self.colors["DANGER"],
            "COLLECTING": self.colors["WARNING"],
            "WAITING": self.colors["INFO"]
        }.get(status_info['status'], self.colors["INFO"])

        cv2.circle(self.output_image, (self.screen_size[1] - 60, 50), 8, status_color, -1)

    def _render_stats_card(self):
        """渲染统计卡片"""
        padding = self.layout["padding"]
        start_y = self.layout["header_height"] + self.layout["section_spacing"]
        height = self.layout["stats_height"]

        # 卡片
        ModernUIRenderer.draw_rounded_rect(
            self.output_image,
            (padding, start_y),
            (self.screen_size[1] - padding, start_y + height),
            self.colors["BG_WHITE"], -1, self.layout["card_radius"]
        )

        ModernUIRenderer.draw_rounded_rect(
            self.output_image,
            (padding, start_y),
            (self.screen_size[1] - padding, start_y + height),
            self.colors["BORDER_LIGHT"], 1, self.layout["card_radius"]
        )

        # 统计数据
        stats = self.core.get_stats()
        card_width = self.screen_size[1] - 2 * padding
        item_width = card_width // 4

        stats_data = [
            ("Attempts", str(stats['total_attempts']), self.colors["PRIMARY"]),
            ("Success", str(stats['successful_recognitions']), self.colors["SUCCESS"]),
            ("Rate", f"{stats['success_rate']:.1f}%", self.colors["WARNING"]),
            ("Buffer", f"{stats['buffer_length']}/{stats['max_buffer_length']}", self.colors["INFO"])
        ]

        for i, (label, value, color) in enumerate(stats_data):
            x = padding + 20 + i * item_width
            y = start_y + 30

            cv2.putText(self.output_image, value, (x, y),
                        self.fonts["title"], 0.8, color, 2)
            cv2.putText(self.output_image, label, (x, y + 25),
                        self.fonts["small"], 0.5, self.colors["TEXT_SECONDARY"], 1)

            if i < len(stats_data) - 1:
                line_x = x + item_width - 10
                cv2.line(self.output_image, (line_x, start_y + 15),
                         (line_x, start_y + height - 15),
                         self.colors["BORDER_LIGHT"], 1)

    def _render_status_card(self):
        """渲染状态卡片"""
        padding = self.layout["padding"]
        start_y = self.layout["header_height"] + self.layout["stats_height"] + 2 * self.layout["section_spacing"]
        height = self.layout["status_height"]

        status_info = self.core.get_current_status()
        status = status_info['status']
        result = status_info['result']

        # 状态颜色
        status_color = {
            "SUCCESS": self.colors["SUCCESS"],
            "FAILED": self.colors["DANGER"],
            "COLLECTING": self.colors["WARNING"],
            "WAITING": self.colors["INFO"]
        }.get(status, self.colors["INFO"])

        # 脉冲效果
        border_width = 1
        if status == "COLLECTING":
            pulse = int(30 * (1 + math.sin(self.animation["pulse_phase"])) / 2)
            status_color = tuple(min(255, max(0, c + pulse)) for c in status_color)
            border_width = 3

        # 卡片
        ModernUIRenderer.draw_rounded_rect(
            self.output_image,
            (padding, start_y),
            (self.screen_size[1] - padding, start_y + height),
            self.colors["BG_WHITE"], -1, self.layout["card_radius"]
        )

        ModernUIRenderer.draw_rounded_rect(
            self.output_image,
            (padding, start_y),
            (self.screen_size[1] - padding, start_y + height),
            status_color, border_width, self.layout["card_radius"]
        )

        # 渲染状态内容
        if status == "SUCCESS" and result:
            self._render_success_content(start_y, padding, status_color, result)
        elif status == "FAILED":
            self._render_failed_content(start_y, padding, status_color)
        elif status == "COLLECTING":
            self._render_collecting_content(start_y, padding, status_color)
        else:
            self._render_waiting_content(start_y, padding, status_color)

    def _render_success_content(self, start_y: int, padding: int, color: tuple, result: Dict):
        """渲染成功状态"""
        content_x = padding + self.layout["card_padding"]
        content_y = start_y + self.layout["card_padding"]

        cv2.putText(self.output_image, "✓ Recognition Success",
                    (content_x, content_y + 30),
                    self.fonts["title"], 0.8, color, 2)

        # 手势信息
        info_y = content_y + 65

        cv2.putText(self.output_image, "Gesture:",
                    (content_x, info_y),
                    self.fonts["normal"], 0.6, self.colors["TEXT_SECONDARY"], 1)
        cv2.putText(self.output_image, result['gesture_label'],
                    (content_x + 100, info_y),
                    self.fonts["normal"], 0.7, self.colors["TEXT_PRIMARY"], 2)

        # 中文含义 - 高亮背景
        chinese_y = info_y + 35
        ModernUIRenderer.draw_rounded_rect(
            self.output_image,
            (content_x - 5, chinese_y - 20),
            (self.screen_size[1] - padding - self.layout["card_padding"], chinese_y + 10),
            color, -1, 6
        )

        cv2.putText(self.output_image, "中文:",
                    (content_x, chinese_y),
                    self.fonts["normal"], 0.6, self.colors["BG_WHITE"], 1)
        cv2.putText(self.output_image, result['chinese_meaning'],
                    (content_x + 70, chinese_y),
                    self.fonts["mono"], 0.9, self.colors["BG_WHITE"], 2)

        # 英文含义
        english_y = chinese_y + 45
        cv2.putText(self.output_image, "English:",
                    (content_x, english_y),
                    self.fonts["normal"], 0.6, self.colors["TEXT_SECONDARY"], 1)
        cv2.putText(self.output_image, result['english_meaning'],
                    (content_x + 100, english_y),
                    self.fonts["normal"], 0.7, self.colors["TEXT_PRIMARY"], 2)

        # 置信度
        conf_y = english_y + 40
        cv2.putText(self.output_image, f"Confidence: {result['confidence']:.1%}",
                    (content_x, conf_y),
                    self.fonts["small"], 0.6, self.colors["TEXT_SECONDARY"], 1)

        # 置信度进度条
        progress_y = conf_y + 20
        ModernUIRenderer.draw_progress_bar(
            self.output_image,
            (content_x, progress_y),
            (self.screen_size[1] - padding - self.layout["card_padding"] - 20, progress_y + 10),
            result['confidence'],
            self.colors["BG_LIGHT"],
            color
        )

    def _render_failed_content(self, start_y: int, padding: int, color: tuple):
        """渲染失败状态"""
        content_x = padding + self.layout["card_padding"]
        content_y = start_y + self.layout["card_padding"]

        cv2.putText(self.output_image, "✗ Recognition Failed",
                    (content_x, content_y + 40),
                    self.fonts["title"], 0.8, color, 2)

        suggestions = [
            "• Ensure hand is clearly visible",
            "• Hold gesture steady",
            "• Check lighting conditions",
            "• Keep hand within sensor range"
        ]

        for i, suggestion in enumerate(suggestions):
            y = content_y + 80 + i * 25
            cv2.putText(self.output_image, suggestion,
                        (content_x + 10, y),
                        self.fonts["small"], 0.5, self.colors["TEXT_REGULAR"], 1)

    def _render_collecting_content(self, start_y: int, padding: int, color: tuple):
        """渲染采集状态"""
        content_x = padding + self.layout["card_padding"]
        content_y = start_y + self.layout["card_padding"]

        cv2.putText(self.output_image, "🤚 Collecting Gesture Data...",
                    (content_x, content_y + 40),
                    self.fonts["title"], 0.7, color, 2)

        stats = self.core.get_stats()
        progress = stats['buffer_length'] / stats['max_buffer_length']

        progress_y = content_y + 80
        cv2.putText(self.output_image,
                    f"Progress: {progress:.0%} ({stats['buffer_length']}/{stats['max_buffer_length']} frames)",
                    (content_x, progress_y),
                    self.fonts["normal"], 0.6, self.colors["TEXT_REGULAR"], 1)

        bar_y = progress_y + 25
        ModernUIRenderer.draw_progress_bar(
            self.output_image,
            (content_x, bar_y),
            (self.screen_size[1] - padding - self.layout["card_padding"] - 20, bar_y + 14),
            progress,
            self.colors["BG_LIGHT"],
            color
        )

    def _render_waiting_content(self, start_y: int, padding: int, color: tuple):
        """渲染等待状态"""
        content_x = padding + self.layout["card_padding"]
        content_y = start_y + self.layout["card_padding"]

        cv2.putText(self.output_image, "👋 Ready to Detect",
                    (content_x, content_y + 40),
                    self.fonts["title"], 0.8, color, 2)

        instructions = [
            "Place your hand above the sensor",
            "Make a clear gesture",
            "Hold position steadily"
        ]

        for i, instruction in enumerate(instructions):
            y = content_y + 80 + i * 30
            cv2.circle(self.output_image, (content_x + 10, y - 8), 10, color, -1)
            cv2.putText(self.output_image, str(i + 1),
                        (content_x + 6, y - 3),
                        self.fonts["small"], 0.4, self.colors["BG_WHITE"], 1)
            cv2.putText(self.output_image, instruction,
                        (content_x + 30, y),
                        self.fonts["normal"], 0.5, self.colors["TEXT_REGULAR"], 1)

    def _render_skeleton_area_with_border(self):
        """渲染手指区域边框和标题"""
        # 黑色背景区域
        cv2.rectangle(self.output_image,
                      (20, self.skeleton_area_start),
                      (self.screen_size[1] - 20, self.screen_size[0] - 20),
                      self.colors["BG_DARK"], -1)

        # 边框
        cv2.rectangle(self.output_image,
                      (20, self.skeleton_area_start),
                      (self.screen_size[1] - 20, self.screen_size[0] - 20),
                      self.colors["SKELETON"], 2)

        # 标题
        cv2.putText(self.output_image, "Hand Tracking - Live View",
                    (30, self.skeleton_area_start + 25),
                    self.fonts["subtitle"], 0.6, self.colors["SKELETON"], 1)

    def _render_skeleton(self, event):
        """渲染手部骨架（使用原始逻辑）"""
        if hasattr(event, 'hands') and event.hands:
            self.core.update_collecting_status(True)

            for i, hand in enumerate(event.hands):
                # 显示手的信息
                hand_type = "Left" if hasattr(hand, 'type') and hand.type == leap.HandType.Left else "Right"
                confidence = getattr(hand, 'confidence', 0)
                text = f"Hand {i + 1}: {hand_type} (Conf: {confidence:.2f})"
                cv2.putText(self.output_image, text,
                            (30, self.skeleton_area_start + 50 + i * 20),
                            self.fonts["small"], 0.5, self.colors["SKELETON"], 1)

                # 绘制手部骨架（使用原始逻辑）
                self._draw_hand_skeleton(hand)
        else:
            self.core.update_collecting_status(False)
            cv2.putText(self.output_image, "No hands detected - Place your hand above the sensor",
                        (50, self.skeleton_area_start + self.skeleton_area_height // 2),
                        self.fonts["normal"], 0.6, self.colors["TEXT_SECONDARY"], 1)

    def _draw_hand_skeleton(self, hand):
        """绘制手部骨架（完全使用原始逻辑）"""
        try:
            # 手臂
            wrist = self._get_joint_position(hand.arm.next_joint)
            elbow = self._get_joint_position(hand.arm.prev_joint)

            if wrist and elbow:
                cv2.line(self.output_image, wrist, elbow, self.colors["SKELETON_ARM"], 2)
                cv2.circle(self.output_image, wrist, 4, self.colors["SKELETON_JOINT"], -1)
                cv2.circle(self.output_image, elbow, 4, self.colors["SKELETON_JOINT"], -1)

            # 手指
            for digit in hand.digits:
                for bone in digit.bones:
                    start = self._get_joint_position(bone.prev_joint)
                    end = self._get_joint_position(bone.next_joint)

                    if start and end:
                        cv2.line(self.output_image, start, end, self.colors["SKELETON"], 2)
                        cv2.circle(self.output_image, start, 3, self.colors["SKELETON_JOINT"], -1)
                        cv2.circle(self.output_image, end, 3, self.colors["SKELETON_JOINT"], -1)
        except:
            pass

    def _get_joint_position(self, joint):
        """获取关节屏幕位置（使用原始逻辑，但映射到手指区域）"""
        if joint:
            # 原始逻辑，但映射到手指显示区域
            x = int(joint.x + (self.screen_size[1] / 2))
            y = int(joint.z + (self.skeleton_area_start + self.skeleton_area_height / 2))

            # 限制在手指区域内
            x = max(25, min(x, self.screen_size[1] - 25))
            y = max(self.skeleton_area_start + 30, min(y, self.screen_size[0] - 25))

            return (x, y)
        return None

    def show(self):
        """显示界面"""
        cv2.imshow("Sign Language Recognition", self.output_image)

    def set_colors(self, color_config: Dict):
        """设置颜色配置"""
        self.colors.update(color_config)
        self.needs_redraw = True

    def set_layout(self, layout_config: Dict):
        """设置布局配置"""
        self.layout.update(layout_config)
        self.needs_redraw = True

    def set_fonts(self, font_config: Dict):
        """设置字体配置"""
        self.fonts.update(font_config)
        self.needs_redraw = True


class HandDataExtractor:
    """手部数据提取器（保持完全不变）"""

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
    """Leap Motion监听器（保持原有逻辑不变）"""

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

            # 提取手部数据
            if hasattr(event, 'hands') and event.hands:
                for hand in event.hands:
                    hand_data = HandDataExtractor.extract_hand_data(hand)
                    if hand_data:  # 即使异常，也会返回合法结构
                        frame_data['hands'].append(hand_data)

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

        # 创建现代化UI
        self.ui = GestureRecognitionUI(self.core)

        # 创建监听器
        self.listener = GestureRecognitionListener(self.core, self.ui)

    def run(self):
        """运行识别系统"""
        connection = leap.Connection()
        connection.add_listener(self.listener)

        print("现代化手语识别系统启动成功! 按'q'退出")

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
    print("启动现代化手语识别系统")
    print("=" * 40)

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

        # Element UI风格自定义（可选）
        custom_colors = {
            "PRIMARY": (64, 158, 255),  # Element UI经典蓝
            "SUCCESS": (103, 194, 58),  # Element UI成功绿
            "WARNING": (230, 162, 60),  # Element UI警告橙
            "DANGER": (245, 108, 108),  # Element UI危险红
            "SKELETON": (0, 255, 255),  # 青色骨架（黑色背景下清晰）
            "SKELETON_JOINT": (255, 255, 0),  # 黄色关节点
            "SKELETON_ARM": (255, 0, 255)  # 紫色手臂
        }

        custom_layout = {
            "padding": 24,  # 更大的内边距
            "card_radius": 12,  # 更大的圆角
            "status_height": 250  # 状态区域高度
        }

        recognizer.customize_ui(colors=custom_colors, layout=custom_layout)

        # 运行系统
        recognizer.run()

    except Exception as e:
        print(f"启动失败: {e}")


if __name__ == "__main__":
    main()
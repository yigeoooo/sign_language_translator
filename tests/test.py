# -*- coding: utf-8 -*- 
# @Time    : 2025/6/27 11:02
# @Author  : yigeoooo
# @FileName: test.py
# @Software: PyCharm
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的 Leap Motion 测试
测试基本连接和手部检测
"""

import time


def test_leap():
    """测试 Leap Motion"""
    print("Leap Motion 简单测试")
    print("=" * 40)

    # 1. 检查 SDK
    try:
        import leap
        print("Leap SDK 导入成功")
    except ImportError as e:
        print(f"无法导入 Leap SDK: {e}")
        print("\n解决方法：")
        print("1. 安装 SDK: pip install leapc-python-bindings")
        print("2. 或使用模拟模式运行主程序")
        return False

    # 2. 创建简单的监听器
    print("\n创建监听器...")

    class SimpleListener(leap.Listener):
        def __init__(self):
            self.events = []

        def on_connection_event(self, event):
            print("收到连接事件")
            self.events.append('connection')

        def on_device_event(self, event):
            print("收到设备事件")
            self.events.append('device')
            try:
                with event.device.open():
                    info = event.device.get_info()
                    print(f"     设备序列号: {info.serial}")
            except:
                pass

        def on_tracking_event(self, event):
            if 'tracking' not in self.events:
                print("收到追踪事件")
                self.events.append('tracking')
            if event.hands:
                print(f"     检测到 {len(event.hands)} 只手")

    # 3. 测试连接
    print("\n测试连接...")
    listener = SimpleListener()
    connection = leap.Connection()
    connection.add_listener(listener)

    try:
        with connection.open():
            print("连接已打开，等待事件...")

            # 等待5秒
            for i in range(5):
                print(f"  等待... {i + 1}/5 秒")
                time.sleep(1)
                if listener.events:
                    print(f"  已收到事件: {listener.events}")

            if 'connection' in listener.events:
                print("\n成功连接到 Leap Service")
            else:
                print("\n未能连接到 Leap Service")
                print("请确保 Ultraleap 服务正在运行")

            if 'device' in listener.events:
                print("检测到 Leap Motion 设备")
            else:
                print("未检测到设备")

            if 'tracking' in listener.events:
                print("正在接收追踪数据")
            else:
                print("⚠未收到追踪数据")

    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        return False

    print("\n测试完成！")
    return True


def test_simulation_mode():
    """测试模拟模式"""
    print("\n" + "=" * 40)
    print("测试模拟模式")
    print("=" * 40)

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    try:
        from utils.config import Config

        config = Config()
        config.set('leap_motion.simulation_mode', True)
        config.save()

        print("已启用模拟模式")
        print("\n现在可以运行主程序：")
        print("  python main.py")
        print("\n在模拟模式下：")
        print("- 不需要 Leap Motion 硬件")
        print("- 系统会生成模拟的手势数据")
        print("- 可以测试 UI 和翻译功能")

        return True

    except Exception as e:
        print(f"启用模拟模式失败: {e}")
        return False


def main():
    """主函数"""
    print("手语翻译系统 - Leap Motion 测试\n")

    # 测试 Leap
    leap_ok = test_leap()

    if not leap_ok:
        # 如果 Leap 测试失败，提供模拟模式选项
        print("\n是否要启用模拟模式？(y/n): ", end='')
        choice = input().strip().lower()

        if choice == 'y':
            test_simulation_mode()
    else:
        print("\nLeap Motion 工作正常！")
        print("可以运行主程序: python main.py")


if __name__ == "__main__":
    main()
    print("\n按任意键退出...")
    input()
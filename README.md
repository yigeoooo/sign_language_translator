# Healer - Sign Language Translation System | 手语翻译系统

##  Project Introduction | 项目简介

**Healer** is an AI-driven sign language translation system designed to bridge the communication gap between deaf and hearing individuals. It captures hand gestures through a camera, recognizes sign language in real-time, and translates it into spoken or written language.

Healer 是一款基于人工智能的手语翻译系统，旨在消除听障人士与健听人群之间的沟通障碍。系统通过摄像头捕捉手势动作，实时识别手语内容，并将其翻译为语音或文字输出。

---

##  环境配置


- **Python版本：  3.8（必须）**

- **leapmotion2环境配置：**  
  * 参考官方Python重构配置环境，由于leapmotion的SDK是基于C语言进行二次开发，所以用Python时需要特殊配置环境。
  * 具体参考Github官方项目。[leapc-python-bindings](https://github.com/ultraleap/leapc-python-bindings)。
  * 当根据流程配置好环境，运行项目中visualiser.py，看到如下效果，则证明环境配置成功。（前提是拥有leapmotion2设备，并且已经连接）</br>
  ![图片](/tests/video.gif "leapmotion2测试")



📊 Gauge Reader - 仪表指针自动读数系统

一个基于 OpenCV + 几何建模 + IoU匹配 的传统视觉仪表读数识别方案，支持：

🎯 指针自动定位
📐 精确角度匹配
📊 标定映射读数
⚙️ 通用表盘适配（非深度学习）
🖼️ 效果展示
模板	实际识别
标准表盘	实际仪表
自动拟合指针角度	输出数值

示例结果：

角度(rad): 1.571
匹配度: 0.92
读数: 0.500
🚀 核心特性
1️⃣ 非深度学习方案（重点优势）
无需数据集
无需训练
可解释性强
工业场景稳定
2️⃣ IoU指针匹配（核心算法）

通过生成“虚拟指针”，与真实指针mask做重叠计算：

IoU = Intersection / Union

选择最优角度：

score = calc_iou(real_mask, temp_mask)
3️⃣ 角度展开（关键优化）

解决仪表跨 0°/360° 导致读数错误问题：

angles = np.unwrap(angles)

✔ 避免读数跳变
✔ 保证插值单调

4️⃣ 标定 + 插值映射

通过人工标定 50 个刻度点：

角度 → 数值（0~1）

再使用：

np.interp()

实现连续读数。

📂 项目结构
Gauge-Reader/
│
├── 990.py                # 标定程序（手动点击）
├── 32.py                 # 自动识别程序
├── calibration_data.txt  # 标定结果
├── template.jpg          # 标准表盘
├── test.jpg              # 测试图像
└── README.md
⚙️ 安装环境
pip install opencv-python numpy
🧪 使用流程
Step 1️⃣ 标定表盘

运行：

python 990.py

操作：

按 c 开始标定
点击：
表盘中心（1次）
指针刻度点（50个）
自动生成：
calibration_data.txt
Step 2️⃣ 自动识别

修改路径：

IMG_PATH = "template.jpg"
TEST_IMG_PATH = "02.jpg"

运行：

python 32.py

输出：

读数: 0.500
🧠 算法流程
输入图像
   ↓
HSV提取红色指针
   ↓
生成候选角度（0~360°）
   ↓
构建虚拟指针
   ↓
IoU匹配最优角度
   ↓
角度归一化（unwrap）
   ↓
插值计算读数
🔧 关键技术点
✔ 指针提取（HSV）
mask = cv2.inRange(hsv, lower, upper)
✔ 圆域约束（去噪）
cv2.circle(mask, center, radius, 255, -1)
✔ 角度归一化（核心）
diff = (ang - base + np.pi) % (2*np.pi) - np.pi
📈 适用场景
电力仪表识别（你项目核心）
压力表 / 温度表
工业巡检机器人
边缘计算设备（Jetson / RDK）
⚠️ 已知限制
仅支持单指针仪表
指针需有明显颜色（默认红色）
强光 / 反光会影响mask提取
非圆形表盘需额外适配
🔥 可扩展方向
✅ YOLO + 本方法融合（检测 + 读数）
✅ 多指针仪表识别
✅ 自动标定（去人工点击）
✅ 支持不同颜色指针
✅ GPU加速匹配
🧩 与深度学习对比
方法	优点	缺点
本项目	无训练、稳定、可解释	需标定
YOLO	自动化强	需要数据集

👉 推荐：工业场景优先用本方法

📜 License

MIT License

🙌 致谢

如果该项目对你有帮助，欢迎 ⭐ Star！

📬 联系

如需合作 / 定制：

工业视觉
电力巡检
ROS + 机器人集成

欢迎交流 👇# Gauge-Reader---

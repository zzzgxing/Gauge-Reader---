import cv2
import numpy as np
import math
import os

# ===============================
# 配置
# ===============================
IMG_PATH = r"template.jpg"
TEST_IMG_PATH = r"02.jpg"


# ===============================
# 工具函数
# ===============================
def extract_pointer_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def generate_pointer_mask(shape, center, angle, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)

    x = int(center[0] + radius * np.cos(angle))
    y = int(center[1] - radius * np.sin(angle))

    cv2.line(mask, center, (x, y), 255, 6)

    return mask


def calc_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0


# ===============================
# 主类
# ===============================
class GaugeApp:
    def __init__(self, img_path):
        self.raw_img = cv2.imread(img_path)
        if self.raw_img is None:
            raise Exception("❌ template读取失败")

        self.h, self.w = self.raw_img.shape[:2]

        self.calib_center = None
        self.calib_angles = []

        self.anchor_indices = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49]
        self.anchor_values = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4,
             0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        self.radius = int(min(self.h, self.w) * 0.45)

        self.load_calibration()

    # ===============================
    # 读取标定
    # ===============================
    def load_calibration(self):
        if not os.path.exists("calibration_data.txt"):
            raise Exception("❌ 没有 calibration_data.txt")

        angles = []

        with open("calibration_data.txt", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if parts[0] == "CENTER":
                    self.calib_center = (int(parts[1]), int(parts[2]))
                else:
                    angles.append(float(parts[1]))

        self.calib_angles = np.unwrap(angles)
        print("✅ 标定加载成功")

    def get_anchor_angles(self):
        return np.array([self.calib_angles[i] for i in self.anchor_indices])

    # ===============================
    # 🔥 核心修复：角度归一化
    # ===============================
    def normalize_angle(self, ang):
        base = self.calib_angles[0]

        diff = ang - base
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        return base + diff

    def angle_to_value(self, ang):
        # 1. 获取锚点角度
        anchor_angles = self.get_anchor_angles()

        # 2. 核心：统一角度周期。将搜索到的角度 ang 映射到锚点的起止范围内
        # 找出锚点的范围
        a_min = min(anchor_angles)
        a_max = max(anchor_angles)

        # 调整 ang 使得它相对于第一个锚点角度处于 [-pi, pi] 之间
        base = anchor_angles[0]
        diff = (ang - base + np.pi) % (2 * np.pi) - np.pi
        target_ang = base + diff

        # 3. 处理 np.interp 要求 xp 递增的问题
        # 如果锚点角度是递减的（顺时针表盘常见情况），我们需要翻转它们进行插值
        if anchor_angles[0] > anchor_angles[-1]:
            # 翻转角度和对应的数值
            value = np.interp(target_ang, anchor_angles[::-1], self.anchor_values[::-1])
        else:
            value = np.interp(target_ang, anchor_angles, self.anchor_values)

        # 4. 打印调试信息（如果你发现还是1，看这里的输出）
        # print(f"DEBUG: target_ang={target_ang:.3f}, range=[{anchor_angles[0]:.3f}, {anchor_angles[-1]:.3f}]")

        return np.clip(value, 0, 1)

    # ===============================
    # 自动识别
    # ===============================
    def detect(self, test_img):

        test_img = cv2.resize(test_img, (self.w, self.h))
        real_mask = extract_pointer_mask(test_img)

        # 限制在圆内
        circle_mask = np.zeros_like(real_mask)
        cv2.circle(circle_mask, self.calib_center,
                   int(self.radius * 0.95), 255, -1)

        real_mask = real_mask & circle_mask

        best_score = -1
        best_angle = 0

        # 🔥 在完整角度范围搜索（修复关键）
        for ang in np.linspace(0, 2*np.pi, 360):

            temp_mask = generate_pointer_mask(
                self.raw_img.shape,
                self.calib_center,
                ang,
                self.radius
            )

            temp_mask = temp_mask & circle_mask

            score = calc_iou(real_mask, temp_mask)

            if score > best_score:
                best_score = score
                best_angle = ang

        value = self.angle_to_value(best_angle)

        print("\n======================")
        print(f"角度(rad): {best_angle:.3f}")
        print(f"匹配度: {best_score:.3f}")
        print(f"读数: {value:.3f}")
        print("======================\n")

        return value, best_angle, real_mask

    # ===============================
    # 运行
    # ===============================
    def run(self):
        test_img = cv2.imread(TEST_IMG_PATH)
        if test_img is None:
            raise Exception("❌ 测试图读取失败")

        value, angle, mask = self.detect(test_img)

        vis = self.raw_img.copy()

        x = int(self.calib_center[0] + self.radius * np.cos(angle))
        y = int(self.calib_center[1] - self.radius * np.sin(angle))

        cv2.line(vis, self.calib_center, (x, y), (0, 255, 0), 3)
        cv2.circle(vis, self.calib_center, 6, (255, 0, 0), -1)

        cv2.putText(vis, f"Value: {value:.3f}",
                    (20, 50), 1, 2, (0, 0, 255), 2)

        cv2.imshow("Result", vis)
        cv2.imshow("Mask", mask)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ===============================
# 入口
# ===============================
if __name__ == "__main__":
    app = GaugeApp(IMG_PATH)
    app.run()
"""生成 SceneSense 的合成示例视频：assets/demo/sample.mp4。

1280x720 / 25fps / 8s，4 个不同颜色的行人（无人像、无版权素材，纯几何绘制）：
- P1 红：从左到右横穿画面
- P2 绿：从右到左，t≈3.2s 与 P1 交叉重叠 → 模拟遮挡
- P3 蓝：在中部低速徘徊（前后来回）→ 模拟 loitering
- P4 黄：t<2s 走出画面右侧，t>5.5s 再走回来 → 模拟离开再出现

用法: python gen_demo_video.py [输出路径]
"""

import sys

import cv2
import numpy as np

W, H, FPS = 1280, 720, 25
SECONDS = 8.0
N = int(FPS * SECONDS)

FLOOR_Y = 380  # 地平线
SKIN = (176, 200, 224)
RNG = np.random.default_rng(7)


def make_background() -> np.ndarray:
    """静态背景：走廊 + 门 + 立柱 + 轻微噪声纹理（噪声帮助检测器，也让画面不像纯色块）。"""
    bg = np.full((H, W, 3), 118, np.uint8)          # 墙面
    cv2.rectangle(bg, (0, FLOOR_Y), (W, H), (96, 99, 103), -1)  # 地面
    cv2.line(bg, (0, FLOOR_Y), (W, FLOOR_Y), (150, 152, 156), 2)

    # 左侧门洞
    cv2.rectangle(bg, (80, 120), (360, FLOOR_Y), (128, 126, 124), -1)
    cv2.rectangle(bg, (110, 190), (250, FLOOR_Y), (78, 84, 94), -1)
    cv2.rectangle(bg, (110, 190), (250, FLOOR_Y), (52, 56, 64), 2)

    # 右侧窗
    cv2.rectangle(bg, (960, 130), (1200, FLOOR_Y), (150, 158, 168), -1)
    cv2.rectangle(bg, (960, 130), (1200, FLOOR_Y), (70, 74, 82), 3)
    cv2.line(bg, (1080, 130), (1080, FLOOR_Y), (70, 74, 82), 3)

    # 立柱（真实遮挡物）
    cv2.rectangle(bg, (620, 150), (700, FLOOR_Y), (158, 158, 162), -1)
    cv2.rectangle(bg, (620, 150), (700, FLOOR_Y), (72, 74, 78), 2)

    # 地面通道虚线 + 远处地砖缝
    for x in range(0, W, 80):
        cv2.line(bg, (x, 560), (x + 40, 560), (180, 180, 180), 3)
    for y in range(FLOOR_Y + 40, H, 70):
        cv2.line(bg, (0, y), (W, y), (108, 111, 115), 1)

    # 顶灯眩光
    cv2.circle(bg, (400, 40), 26, (205, 208, 212), -1)
    cv2.circle(bg, (880, 40), 26, (205, 208, 212), -1)

    # 轻微噪声纹理
    noise = RNG.normal(0, 4.0, (H, W, 3))
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (3, 3), 0)
    return bg


def draw_person(frame, cx, foot_y, shirt, trousers, phase):
    """画一个行走中的人：头 + 躯干 + 摆动的手臂/双腿 + 影子。

    cx, foot_y 为脚底中心；phase 是步态相位（弧度）。
    """
    height = 200.0
    u = height / 8.0
    top = foot_y - height
    head_ry = u * 0.52
    head_cy = top + head_ry
    shoulder_y = int(head_cy + head_ry + u * 0.28)
    hip_y = int(top + height * 0.50)
    thigh_t = int(u * 0.34)
    arm_t = int(u * 0.26)

    # 影子
    cv2.ellipse(frame, (cx, int(foot_y - 2)), (int(u * 1.05), int(u * 0.26)), 0, 0,
                360, (74, 77, 81), -1)

    swing = np.sin(phase)
    swing2 = np.sin(phase + np.pi)

    # 腿（先画，躯干压在上面）
    for sign, sw in ((-1, swing), (1, swing2)):
        hip = (cx + sign * int(u * 0.30), int(hip_y))
        knee = (hip[0] + int(sw * u * 0.45), int(hip_y + (foot_y - hip_y) * 0.5))
        ankle = (hip[0] + int(sw * u * 0.85), int(foot_y - u * 0.18))
        cv2.line(frame, hip, knee, trousers, thigh_t)
        cv2.line(frame, knee, ankle, trousers, thigh_t)
        cv2.ellipse(frame, (ankle[0] + int(sw * u * 0.18), int(foot_y - u * 0.10)),
                    (int(u * 0.34), int(u * 0.14)), 0, 0, 360, (38, 40, 46), -1)

    # 躯干（梯形）
    torso = np.array([
        [cx - int(u * 0.62), shoulder_y],
        [cx + int(u * 0.62), shoulder_y],
        [cx + int(u * 0.50), hip_y],
        [cx - int(u * 0.50), hip_y],
    ], np.int32)
    cv2.fillPoly(frame, [torso], shirt)

    # 手臂（与腿反相摆动）
    for sign, sw in ((-1, swing2), (1, swing)):
        shoulder = (cx + sign * int(u * 0.58), int(shoulder_y + u * 0.06))
        elbow = (shoulder[0] + int(sw * u * 0.42), int(shoulder_y + u * 0.75))
        hand = (shoulder[0] + int(sw * u * 0.75), int(hip_y + u * 0.55))
        cv2.line(frame, shoulder, elbow, shirt, arm_t)
        cv2.line(frame, elbow, hand, shirt, arm_t)
        cv2.circle(frame, hand, int(u * 0.17), SKIN, -1)

    # 颈 + 头
    cv2.line(frame, (cx, int(head_cy + head_ry)), (cx, shoulder_y), SKIN, int(u * 0.30))
    cv2.ellipse(frame, (cx, int(head_cy)), (int(u * 0.42), int(head_ry)), 0, 0, 360, SKIN, -1)


def lerp(a, b, t):
    return a + (b - a) * t


def main(out_path: str) -> None:
    bg = make_background()
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened(), "VideoWriter 打不开，检查 mp4v 编码"

    for i in range(N):
        t = i / FPS
        frame = bg.copy()
        actors = []  # (x, 脚底 y, 上衣, 裤子, 步态相位)

        # P1 红：左 → 右横穿，速度≈1.3 m/s 的视觉等价
        p1x = lerp(110, 1170, t / SECONDS)
        actors.append((int(p1x), 620, (52, 52, 214), (62, 62, 120), t * 7.5))

        # P2 绿：右 → 左，t≈3.2s 时与 P1 在画面中部重叠
        p2x = lerp(1150, 190, t / SECONDS)
        actors.append((int(p2x), 650, (64, 188, 74), (56, 64, 72), t * 7.5 + 1.0))

        # P3 蓝：中部低速徘徊（前后来回，振幅 90px）→ loitering
        p3x = 640 + np.sin(t * 1.9) * 90
        actors.append((int(p3x), 672, (212, 128, 44), (58, 44, 36), t * 3.0))

        # P4 黄：t<2s 在画面内 → 走出右侧 → t>5.5s 走回来
        if t < 2.0:
            p4x = lerp(900, 1300, t / 2.0)
            p4_phase = t * 7.5
        elif t < 5.5:
            p4x = 1330.0  # 画面外
            p4_phase = 0.0
        else:
            p4x = lerp(1300, 1010, (t - 5.5) / 2.5)
            p4_phase = (t - 5.5) * 7.5
        if p4x < W + 25:
            actors.append((int(p4x), 600, (44, 208, 236), (66, 58, 44), p4_phase))

        # 脚底 y 小的（远处）先画，保证近处的人盖住远处的人
        for cx, foot_y, shirt, trousers, phase in sorted(actors, key=lambda a: a[1]):
            if -80 < cx < W + 80:
                draw_person(frame, cx, foot_y, shirt, trousers, phase)

        cv2.putText(frame, "CAM 01 / HALL-B", (24, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (238, 238, 238), 2)
        cv2.putText(frame, f"{t:05.2f}s  frame {i:04d}", (24, H - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        cv2.putText(frame, "SYNTHETIC DEMO", (W - 330, H - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 226, 255), 2)

        vw.write(frame)

    vw.release()
    print(f"written {out_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    main(sys.argv[1] if len(sys.argv) > 1 else "assets/demo/sample.mp4")

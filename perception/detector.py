import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """第一阶段：检测（bbox + class + confidence），不做 pose。"""

    def __init__(
        self,
        model_path: str = "yolov8l.pt",
        device: str = "cpu",
        conf_threshold: float = 0.15,
        classes: list[int] | None = None,
        imgsz: int = 960,
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(
            frame, conf=self.conf_threshold, classes=self.classes,
            imgsz=self.imgsz, verbose=False,
        )[0]

        detections: list[dict] = []
        if results.boxes is None:
            return detections

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class": results.names[int(box.cls[0])],
                "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]),
            })

        return detections


class PoseEstimator:
    """第二阶段：在检测到的 bbox 上做姿态估计。"""

    def __init__(self, model_path: str = "yolov8l-pose.pt", device: str = "cpu"):
        self.model = YOLO(model_path)
        self.model.to(device)

    def estimate(self, frame: np.ndarray, bboxes: list[list[float]]) -> list[list[list[float]]]:
        """对每个 bbox 做 crop + pose 推理，返回关键点列表。

        Args:
            frame: 完整帧
            bboxes: [[x1,y1,x2,y2], ...]

        Returns:
            [[[x,y,conf]*17], ...] 每个 bbox 对应的 17 个 COCO 关键点
        """
        results = []
        for x1, y1, x2, y2 in bboxes:
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            if x2i <= x1i or y2i <= y1i:
                results.append([])
                continue

            crop = frame[y1i:y2i, x1i:x2i]
            r = self.model(crop, verbose=False)[0]
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                # 取置信度最高的那个人
                kpts = r.keypoints.data[0]
                scores = kpts[:, 2] if kpts.shape[1] > 2 else np.ones(17)
                best_idx = int(scores.mean())
                if len(r.keypoints.data) > 1:
                    # 选关键点平均置信度最高的
                    best = 0
                    best_mean = 0.0
                    for i, k in enumerate(r.keypoints.data):
                        m = k[:, 2].mean() if k.shape[1] > 2 else 1.0
                        if m > best_mean:
                            best_mean = m
                            best = i
                    kpts_crop = r.keypoints.data[best].tolist()
                else:
                    kpts_crop = kpts.tolist()
                # 坐标从 crop 转回原图
                kpts_global = []
                for kp in kpts_crop:
                    kpts_global.append([kp[0] + x1i, kp[1] + y1i, kp[2] if len(kp) > 2 else 1.0])
                results.append(kpts_global)
            else:
                results.append([])

        return results

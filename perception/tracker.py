"""
Hybrid tracker: BoT-SORT architecture with GMC + ReID + Kalman Filter.

Replaces supervision ByteTrack. Same update() interface for drop-in compatibility.
Uses torchvision ResNet18 for appearance features, filterpy for Kalman filtering,
and sparse optical flow for camera motion compensation.
"""

import math
from collections import deque

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor


# ═══════════════════════════════════════════════════════════════════════════════
# Camera Motion Compensation (GMC)
# ═══════════════════════════════════════════════════════════════════════════════

class GMC:
    """Estimates camera motion via sparse optical flow, returns 2x3 affine matrix."""

    def __init__(self, method='sparseOptFlow', downscale=2):
        self.method = method
        self.downscale = max(1, int(downscale))
        self._prev_frame = None
        self._prev_keypoints = None
        self._init = False
        if method == 'sparseOptFlow':
            self._feature_params = dict(maxCorners=1000, qualityLevel=0.01,
                                        minDistance=1, blockSize=3,
                                        useHarrisDetector=False, k=0.04)
        elif method == 'orb':
            self._detector = cv2.FastFeatureDetector_create(20)
            self._extractor = cv2.ORB_create()
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def apply(self, frame: np.ndarray, detections=None) -> np.ndarray:
        """Returns 2x3 affine warp matrix (identity if first frame or estimation fails)."""
        H = np.eye(2, 3, dtype=np.float32)
        if self.method == 'none':
            return H

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))

        if self.method == 'sparseOptFlow':
            keypoints = cv2.goodFeaturesToTrack(gray, mask=None, **self._feature_params)
            if not self._init:
                self._prev_frame = gray.copy()
                self._prev_keypoints = keypoints
                self._init = True
                return H
            if keypoints is None or self._prev_keypoints is None or len(self._prev_keypoints) < 4:
                self._prev_frame = gray.copy()
                self._prev_keypoints = keypoints
                return H
            matched, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_frame, gray, self._prev_keypoints, None)
            prev_pts, curr_pts = [], []
            for i, s in enumerate(status):
                if s:
                    prev_pts.append(self._prev_keypoints[i])
                    curr_pts.append(matched[i])
            prev_pts = np.array(prev_pts, dtype=np.float32)
            curr_pts = np.array(curr_pts, dtype=np.float32)
            self._prev_frame = gray.copy()
            self._prev_keypoints = keypoints
            if len(prev_pts) > 4:
                H, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, cv2.RANSAC)
                if H is None:
                    H = np.eye(2, 3, dtype=np.float32)
                elif self.downscale > 1:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale

        elif self.method == 'orb':
            mask = np.zeros_like(gray)
            mask[int(0.02 * h // self.downscale):int(0.98 * h // self.downscale),
                 int(0.02 * w // self.downscale):int(0.98 * w // self.downscale)] = 255
            if detections is not None:
                for det in detections:
                    tlbr = (det[:4] / self.downscale).astype(np.int_)
                    cv2.rectangle(mask, (tlbr[0], tlbr[1]), (tlbr[2], tlbr[3]), 0, -1)
            kp = self._detector.detect(gray, mask)
            kp, desc = self._extractor.compute(gray, kp)
            if not self._init:
                self._prev_frame = gray.copy()
                self._prev_keypoints = kp
                self._prev_desc = desc
                self._init = True
                return H
            if desc is None or self._prev_desc is None or len(kp) < 4:
                self._prev_frame, self._prev_keypoints, self._prev_desc = gray, kp, desc
                return H
            matches = self._matcher.knnMatch(self._prev_desc, desc, 2)
            good_matches = [m for m, n in matches if m.distance < 0.9 * n.distance]
            prev_pts = np.float32([self._prev_keypoints[m.queryIdx].pt for m in good_matches])
            curr_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])
            self._prev_frame, self._prev_keypoints, self._prev_desc = gray, kp, desc
            if len(prev_pts) > 4:
                H, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, cv2.RANSAC)
                if H is None:
                    H = np.eye(2, 3, dtype=np.float32)
                elif self.downscale > 1:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale

        return H


# ═══════════════════════════════════════════════════════════════════════════════
# ReID Feature Extractor (ResNet18 backbone)
# ═══════════════════════════════════════════════════════════════════════════════

class ReIDExtractor:
    """Extracts appearance embeddings from person crops using ResNet18.

    Returns 512-dim L2-normalized features."""

    def __init__(self, device='cpu'):
        self.device = device
        from torchvision.models import ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Use layers up to avgpool (before fc)
        self.model = create_feature_extractor(backbone, {'avgpool': 'feat'})
        self.model.to(device)
        self.model.eval()
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._dummy = torch.zeros(1, 512)  # dummy feature for detection w/o valid crop

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Extract features for each bbox (xyxy format). Returns Nx512 ndarray."""
        if len(bboxes) == 0:
            return np.empty((0, 512), dtype=np.float32)
        crops = []
        valid_idx = []
        h, w = frame.shape[:2]
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 + 4 and y2 > y1 + 4:
                crop = frame[y1:y2, x1:x2]
                crops.append(self._transform(crop))
                valid_idx.append(i)
        if not crops:
            return np.tile(self._dummy.numpy(), (len(bboxes), 1))
        batch = torch.stack(crops).to(self.device)
        feats = self.model(batch)['feat'].squeeze(-1).squeeze(-1)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        # Fill dummy features for invalid bboxes
        out = np.tile(self._dummy.numpy(), (len(bboxes), 1))
        for j, idx in enumerate(valid_idx):
            out[idx] = feats[j].cpu().numpy()
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Track
# ═══════════════════════════════════════════════════════════════════════════════

class Track:
    """Single track with Kalman state, ReID features, and trajectory history."""

    _id_counter = 0

    def __init__(self, bbox_xyxy, score, feature, frame_idx, timestamp):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.state = 'tentative'  # tentative → confirmed → lost → removed
        self.score = score
        self.frame_idx = frame_idx
        self.start_frame = frame_idx
        self.last_seen = frame_idx
        self.hits = 1
        self._age = 1
        self.time_since_update = 0
        self._min_hits_to_confirm = 3

        # Kalman filter (8-state: x, y, w, h, vx, vy, vw, vh)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.mean, self.cov = self._init_kf_state(bbox_xyxy)

        # ReID features
        self.smooth_feat = feature
        self.features = deque([feature], maxlen=50)
        self._feature_alpha = 0.9

        # Trajectory
        cx, cy = (bbox_xyxy[0] + bbox_xyxy[2]) / 2, (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        self.trajectory: list[list[float]] = [[cx, cy, timestamp]]
        self._centroid = [cx, cy]
        self._bbox_xyxy = list(bbox_xyxy)
        self._velocity_ms = 0.0
        self._direction_deg = 0.0
        self._class_name = 'person'
        self._pose_keypoints: list = []

    @staticmethod
    def _init_kf_state(bbox_xyxy):
        """Initialize KF from xyxy bbox, returns (mean, covariance)."""
        x, y, x2, y2 = bbox_xyxy
        w, h = x2 - x, y2 - y
        cx, cy = x + w / 2, y + h / 2
        kf = KalmanFilter(dim_x=8, dim_z=4)
        mean = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        cov = np.eye(8) * 10.0
        cov[4:, 4:] *= 1000.0
        return mean, cov

    def predict(self):
        """Kalman prediction step."""
        self.kf.x = self.mean
        self.kf.P = self.cov
        if self.state != 'confirmed':
            self.kf.x[6] = 0
            self.kf.x[7] = 0
        self.kf.predict()
        self.mean = self.kf.x.copy()
        self.cov = self.kf.P.copy()
        self._age += 1
        self.time_since_update += 1

    def update(self, bbox_xyxy, score, feature, frame_idx, timestamp):
        """Kalman update with new measurement."""
        x, y, x2, y2 = bbox_xyxy
        w, h = max(1, x2 - x), max(1, y2 - y)
        cx, cy = x + w / 2, y + h / 2
        self.kf.x = self.mean
        self.kf.P = self.cov
        self.kf.update(np.array([cx, cy, w, h], dtype=np.float64))
        self.mean = self.kf.x.copy()
        self.cov = self.kf.P.copy()
        self.update_features(feature)
        self.score = score
        self.last_seen = frame_idx
        self.time_since_update = 0
        if self.state == 'tentative' and self.hits >= self._min_hits_to_confirm:
            self.state = 'confirmed'
        self.hits += 1
        # Update trajectory
        self._centroid = [cx, cy]
        self._bbox_xyxy = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        if len(self.trajectory) >= 2:
            prev = self.trajectory[-1]
            dt = timestamp - prev[2]
            if dt > 0:
                self._velocity_ms = math.sqrt((cx - prev[0])**2 + (cy - prev[1])**2) / dt
                self._direction_deg = math.degrees(math.atan2(cy - prev[1], cx - prev[0])) % 360
        self.trajectory.append([cx, cy, timestamp])
        if len(self.trajectory) > 90:
            self.trajectory = self.trajectory[-90:]

    def update_features(self, feat):
        feat = feat / (np.linalg.norm(feat) + 1e-10)
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self._feature_alpha * self.smooth_feat + (1 - self._feature_alpha) * feat
        self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-10)
        self.features.append(feat)

    def apply_gmc(self, H):
        """Apply camera motion compensation to KF state."""
        if H is None:
            return
        R = H[:2, :2]
        t = H[:2, 2]
        # Apply affine transform to bbox corners, then recompute mean xy
        x1, y1, x2, y2 = self.tlbr
        corners = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        corners_t = (R @ corners.T + t.reshape(2, 1)).T
        new_x1, new_y1 = corners_t[0]
        new_x2, new_y2 = corners_t[1]
        new_w = max(1, new_x2 - new_x1)
        new_h = max(1, new_y2 - new_y1)
        self.mean[0] = (new_x1 + new_x2) / 2
        self.mean[1] = (new_y1 + new_y2) / 2
        self.mean[2] = new_w
        self.mean[3] = new_h
        # Update internal bbox
        self._bbox_xyxy = [new_x1, new_y1, new_x2, new_y2]
        self._centroid = [self.mean[0], self.mean[1]]

    @property
    def tlbr(self):
        """xyxy format."""
        mean = self.mean
        w, h = mean[2], mean[3]
        return [mean[0] - w/2, mean[1] - h/2, mean[0] + w/2, mean[1] + h/2]

    @property
    def bbox(self):
        return [float(v) for v in self._bbox_xyxy]

    @property
    def centroid(self):
        return [float(v) for v in self._centroid]

    @property
    def velocity_ms(self):
        return self._velocity_ms

    @property
    def direction_deg(self):
        return self._direction_deg

    def mark_lost(self):
        self.state = 'lost'

    def mark_removed(self):
        self.state = 'removed'

    def is_confirmed(self):
        return self.state == 'confirmed'

    def is_lost(self):
        return self.state == 'lost'


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Object Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class MultiObjectTracker:
    """Hybrid tracker combining BoT-SORT architecture with GMC + ReID + Kalman.

    Interface compatible with the original supervision ByteTrack wrapper:
        update(detections, frame_idx, timestamp, frame=None) → list[dict]
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 120,
        minimum_matching_threshold: float = 0.7,
        frame_rate: int = 30,
        reid_device: str = 'cpu',
        with_reid: bool = True,
        with_gmc: bool = True,
        gmc_method: str = 'sparseOptFlow',
    ):
        self.track_high_thresh = 0.5
        self.track_low_thresh = max(0.08, track_activation_threshold - 0.15)
        self.new_track_thresh = 0.6
        self.match_thresh = minimum_matching_threshold
        self.lost_track_buffer = lost_track_buffer
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25

        self.tracked_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.removed_tracks: list[Track] = []

        self.frame_idx = 0
        self._frame_rate = frame_rate
        self._with_reid = with_reid
        self._with_gmc = with_gmc

        if with_reid:
            self.reid = ReIDExtractor(device=reid_device)
        else:
            self.reid = None
        if with_gmc:
            self.gmc = GMC(method=gmc_method)
        else:
            self.gmc = None

        # EMA bbox smoothing (kept for output consistency)
        self._bbox_ema: dict[int, list[float]] = {}
        self._ema_alpha = 0.3

        # Track ID remapping (for maintaining old IDs across resets)
        self._reset_next_id()

    def _reset_next_id(self):
        Track._id_counter = 0

    def update(
        self, detections: list[dict], frame_idx: int, timestamp: float,
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        """Run tracking on a new frame.

        Args:
            detections: [{"bbox":[x1,y1,x2,y2], "class":str, "class_id":int, "confidence":float}, ...]
            frame_idx: current frame index
            timestamp: current timestamp in seconds
            frame: BGR frame image (for ReID + GMC). If None, ReID/GMC are skipped.

        Returns:
            [{"track_id":int, "bbox":[...], "class":str, "centroid":[...],
              "velocity_ms":float, "direction_deg":float, "trajectory":[...],
              "pose_keypoints":[...]}, ...]
        """
        self.frame_idx += 1

        if not detections:
            for t in self.tracked_tracks:
                t.predict()
            self._prune()
            return []

        # Extract detection data
        bboxes_xyxy = np.array([d['bbox'] for d in detections], dtype=np.float32)
        scores = np.array([d.get('confidence', 0.5) for d in detections], dtype=np.float32)
        class_names = [d.get('class', 'person') for d in detections]

        # ReID features
        if self._with_reid and frame is not None and self.reid is not None:
            features = self.reid.extract(frame, bboxes_xyxy)
        else:
            features = np.random.randn(len(detections), 512).astype(np.float32)
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)

        # GMC
        warp = np.eye(2, 3, dtype=np.float32)
        if self._with_gmc and frame is not None and self.gmc is not None:
            warp = self.gmc.apply(frame, bboxes_xyxy)

        # Split detections by score
        high_mask = scores >= self.track_high_thresh
        low_mask = (scores >= self.track_low_thresh) & ~high_mask

        # Pool of all existing tracks for prediction
        track_pool = self.tracked_tracks + self.lost_tracks
        for t in track_pool:
            t.predict()
            t.apply_gmc(warp)

        confirmed = [t for t in self.tracked_tracks if t.is_confirmed()]
        unconfirmed = [t for t in self.tracked_tracks if not t.is_confirmed()]

        # Keep track of which detections have been assigned
        assigned_dets = set()

        # ── Step 1: (confirmed + lost) × high-score dets ──
        stage1_tracks = confirmed + self.lost_tracks
        high_idx = np.where(high_mask)[0]
        if len(stage1_tracks) > 0 and len(high_idx) > 0:
            high_boxes = bboxes_xyxy[high_idx]
            high_sc = scores[high_idx]
            high_ft = features[high_idx]
            dists = self._combined_distance(stage1_tracks, high_boxes, high_sc, high_ft)
            matches, u_tracks, u_dets = self._match(dists)
            for ti, di in matches:
                track = stage1_tracks[ti]
                det_i = high_idx[di]
                track.update(bboxes_xyxy[det_i], scores[det_i], features[det_i],
                             frame_idx, timestamp)
                track.state = 'confirmed'
                assigned_dets.add(det_i)
                if track in self.lost_tracks:
                    self.lost_tracks.remove(track)
                    self.tracked_tracks.append(track)
            # Unmatched tracks from stage1 that are confirmed → go to stage2
            stage2_tracks = [stage1_tracks[i] for i in u_tracks
                             if stage1_tracks[i].is_confirmed()]
        else:
            stage2_tracks = confirmed
            assigned_dets = set()

        # ── Step 2: remaining stage1 tracks × low-score dets ──
        low_idx = np.where(low_mask)[0]
        if len(stage2_tracks) > 0 and len(low_idx) > 0:
            low_boxes = bboxes_xyxy[low_idx]
            low_sc = scores[low_idx]
            iou_dists = self._iou_distance(stage2_tracks, low_boxes)
            matches, u_tracks, u_dets = self._match(iou_dists, thresh=0.5)
            for ti, di in matches:
                det_i = low_idx[di]
                stage2_tracks[ti].update(bboxes_xyxy[det_i], scores[det_i],
                                         features[det_i], frame_idx, timestamp)
                assigned_dets.add(det_i)
            # Unmatched → lost
            for ti in u_tracks:
                t = stage2_tracks[ti]
                t.mark_lost()
                if t in self.tracked_tracks:
                    self.tracked_tracks.remove(t)
                    self.lost_tracks.append(t)
        elif len(low_idx) == 0:
            # No low-score dets, mark all stage2 unmatched as lost
            for t in stage2_tracks:
                t.mark_lost()
                if t in self.tracked_tracks:
                    self.tracked_tracks.remove(t)
                    self.lost_tracks.append(t)

        # ── Step 3: unconfirmed × unassigned high-score dets ──
        unassigned_high = [i for i in high_idx if i not in assigned_dets]
        if len(unconfirmed) > 0 and len(unassigned_high) > 0:
            uh_boxes = bboxes_xyxy[unassigned_high]
            uh_sc = scores[unassigned_high]
            uh_ft = features[unassigned_high]
            iou_dists = self._iou_distance(unconfirmed, uh_boxes)
            matches, u_unconf, remaining_dets = self._match(iou_dists, thresh=0.7)
            for ti, di in matches:
                det_i = unassigned_high[di]
                unconfirmed[ti].update(bboxes_xyxy[det_i], scores[det_i],
                                       features[det_i], frame_idx, timestamp)
                assigned_dets.add(det_i)
            for ti in u_unconf:
                t = unconfirmed[ti]
                t.mark_removed()
                self.removed_tracks.append(t)
                if t in self.tracked_tracks:
                    self.tracked_tracks.remove(t)
            # Remaining unassigned dets → init new tracks
            final_unassigned = [unassigned_high[i] for i in remaining_dets]
        else:
            final_unassigned = unassigned_high

        # ── Step 4: Init new tracks ──
        for det_i in final_unassigned:
            if scores[det_i] >= self.new_track_thresh:
                track = Track(bboxes_xyxy[det_i], scores[det_i],
                              features[det_i], frame_idx, timestamp)
                self.tracked_tracks.append(track)

        # ── Step 5: Prune ──
        self._prune()

        # ── Build output ──
        return self._build_output(timestamp, class_names, detections, frame_idx)

    def _combined_distance(self, tracks, dets_xyxy, det_scores, det_feats):
        """Compute combined IoU + ReID distance matrix."""
        if not tracks or len(dets_xyxy) == 0:
            return np.empty((len(tracks), len(dets_xyxy)))

        iou_dist = self._iou_distance(tracks, dets_xyxy)
        if self._with_reid and det_feats.shape[1] == 512:
            emb_dist = self._embedding_distance(tracks, det_feats)
            emb_dist[emb_dist > self.appearance_thresh] = 1.0
            iou_mask = iou_dist > self.proximity_thresh
            emb_dist[iou_mask] = 1.0
            return np.minimum(iou_dist, emb_dist)
        return iou_dist

    def _iou_distance(self, tracks, dets_xyxy):
        """1 - IoU distance matrix."""
        if not tracks or len(dets_xyxy) == 0:
            return np.empty((len(tracks), len(dets_xyxy)))
        track_bboxes = np.array([t.tlbr for t in tracks])
        return 1.0 - self._box_iou(track_bboxes, dets_xyxy)

    def _embedding_distance(self, tracks, det_feats):
        """Cosine distance matrix (0=identical, 1=orthogonal)."""
        track_feats = np.array([t.smooth_feat for t in tracks])
        # Cosine similarity
        sim = track_feats @ det_feats.T
        return (1.0 - sim) / 2.0

    @staticmethod
    def _box_iou(boxes1, boxes2):
        """Vectorized IoU. boxes1: Nx4, boxes2: Mx4 (xyxy)."""
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        return inter / (union + 1e-10)

    @staticmethod
    def _match(dists, thresh=0.7):
        """Hungarian matching with threshold."""
        if dists.size == 0:
            return [], list(range(dists.shape[0])), list(range(dists.shape[1]))
        row_ind, col_ind = linear_sum_assignment(dists)
        matches = [(r, c) for r, c in zip(row_ind, col_ind) if dists[r, c] < thresh]
        u_rows = [r for r in range(dists.shape[0]) if r not in [m[0] for m in matches]]
        u_cols = [c for c in range(dists.shape[1]) if c not in [m[1] for m in matches]]
        return matches, u_rows, u_cols

    def _prune(self):
        """Remove stale lost tracks and unconfirmed tracks."""
        for t in self.lost_tracks[:]:
            if self.frame_idx - t.last_seen > self.lost_track_buffer:
                t.mark_removed()
                self.lost_tracks.remove(t)
                self.removed_tracks.append(t)
        # Remove unconfirmed tracks that have gone too long without hits
        for t in self.tracked_tracks[:]:
            if not t.is_confirmed() and t.time_since_update > 1:
                t.mark_removed()
                self.tracked_tracks.remove(t)
                self.removed_tracks.append(t)

    def _build_output(self, timestamp, class_names, detections, frame_idx):
        """Build output list matching the original format. Only confirmed tracks."""
        results = []
        for t in self.tracked_tracks:
            if not t.is_confirmed():
                continue
            bbox_raw = t.bbox
            # EMA smoothing for output bbox
            tid = t.track_id
            if tid in self._bbox_ema:
                prev = self._bbox_ema[tid]
                a = self._ema_alpha
                bbox = [a * r + (1-a) * p for r, p in zip(bbox_raw, prev)]
            else:
                bbox = bbox_raw
            self._bbox_ema[tid] = bbox

            results.append({
                'track_id': tid,
                'bbox': bbox,
                'class': t._class_name,
                'centroid': t.centroid,
                'velocity_ms': round(t.velocity_ms, 2),
                'direction_deg': round(t.direction_deg, 1),
                'trajectory': t.trajectory,
                'pose_keypoints': [],  # injected by pipeline
                'appearance_feature': t.smooth_feat.tolist() if t.smooth_feat is not None else [],
                'track_state': t.state,
                'last_seen_frame': int(t.last_seen),
                'time_since_update': int(t.time_since_update),
            })
        return results

    def get_trajectory(self, track_id: int, max_len: int = 90) -> list[list[float]]:
        for t in self.tracked_tracks:
            if t.track_id == track_id:
                return t.trajectory[-max_len:]
        for t in self.lost_tracks:
            if t.track_id == track_id:
                return t.trajectory[-max_len:]
        return []

    def get_active_tracks(self) -> list[int]:
        return [t.track_id for t in self.tracked_tracks if t.is_confirmed()]

    def reset(self):
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self._bbox_ema.clear()
        self._reset_next_id()

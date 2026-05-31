import numpy as np


class _FakeBoxes:
    def __init__(self):
        self._items = []

    def __iter__(self):
        return iter(self._items)


class _FakeDetectResult:
    boxes = _FakeBoxes()
    names = {}


class _FakeKeypoints:
    def __init__(self, data):
        self.data = data


class _FakePoseResult:
    def __init__(self, offset):
        kpts = np.zeros((1, 17, 3), dtype=float)
        kpts[0, :, 0] = offset
        kpts[0, :, 1] = offset + 1
        kpts[0, :, 2] = 0.9
        self.keypoints = _FakeKeypoints(kpts)


def test_yolo_detector_passes_device_and_half_for_cuda(monkeypatch):
    import perception.detector as detector_module

    calls = {}

    class FakeYOLO:
        def __init__(self, model_path):
            self.model_path = model_path

        def to(self, device):
            calls["to"] = device

        def __call__(self, source, **kwargs):
            calls["kwargs"] = kwargs
            return [_FakeDetectResult()]

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)

    detector = detector_module.YOLODetector("model.pt", device="cuda", conf_threshold=0.2, imgsz=960)
    detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert calls["to"] == "cuda"
    assert calls["kwargs"]["device"] == "cuda"
    assert calls["kwargs"]["half"] is True
    assert calls["kwargs"]["conf"] == 0.2
    assert calls["kwargs"]["imgsz"] == 960


def test_pose_estimator_batches_crops_in_single_model_call(monkeypatch):
    import perception.detector as detector_module

    calls = []

    class FakeYOLO:
        def __init__(self, model_path):
            pass

        def to(self, device):
            pass

        def __call__(self, source, **kwargs):
            calls.append({"count": len(source), "kwargs": kwargs})
            return [_FakePoseResult(3), _FakePoseResult(7)]

    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)

    estimator = detector_module.PoseEstimator("pose.pt", device="cuda")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = estimator.estimate(frame, [[10, 20, 40, 80], [50, 10, 80, 60]])

    assert len(calls) == 1
    assert calls[0]["count"] == 2
    assert calls[0]["kwargs"]["device"] == "cuda"
    assert calls[0]["kwargs"]["half"] is True
    assert result[0][0] == [13.0, 24.0, 0.9]
    assert result[1][0] == [57.0, 18.0, 0.9]

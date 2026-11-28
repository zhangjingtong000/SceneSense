"""ReID 权重来源：models/ 里有 OSNet 权重时，默认 'osnet' 必须优先走本地链路。

否则默认路径会整个交给 BoxMOT，由它去 Google Drive 拉权重——实测超时后静默回退
ResNet18，README 里「把权重放进 models/」的指引就落空了。
"""

import sys
import types

import perception.tracker as tracker_module
from perception.tracker import ReIDExtractor


class FakeOSNet:
    def __init__(self):
        self.loaded_state = None
        self.eval_called = False
        self.device = None

    def load_state_dict(self, state, strict=True):
        self.loaded_state = state

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self

    def half(self):
        return self


def test_local_weights_switch_default_backbone_to_torchreid(tmp_path, monkeypatch):
    weights = tmp_path / "osnet_x0_25_msmt17.pt"
    weights.write_bytes(b"stub-weights")
    monkeypatch.setattr(ReIDExtractor, "_OSNET_WEIGHTS", weights)

    fake_model = FakeOSNet()
    fake_torchreid = types.ModuleType("torchreid")
    fake_torchreid.models = types.SimpleNamespace(
        build_model=lambda name, num_classes, pretrained: fake_model
    )
    monkeypatch.setitem(sys.modules, "torchreid", fake_torchreid)
    monkeypatch.setattr(tracker_module.torch, "load", lambda *args, **kwargs: {"conv": 1})

    # 默认 backbone='osnet'：以前会走 BoxMOT（去 Google Drive 拉权重）
    extractor = ReIDExtractor(device="cpu")

    assert extractor.backbone == "osnet_fast"
    assert extractor._osnet_model is fake_model
    assert fake_model.loaded_state == {"conv": 1}
    assert fake_model.eval_called is True


def test_without_local_weights_default_backbone_stays_boxmot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ReIDExtractor, "_OSNET_WEIGHTS", tmp_path / "not-downloaded.pt"
    )

    created = {}

    class FakeEncoder:
        def encode(self, frames, detections):
            return [None]

    def fake_create_reid_encoder(config):
        created["model"] = config.model
        return FakeEncoder()

    fake_reid = types.ModuleType("boxmot.reid")
    fake_reid.ReIDConfig = lambda model, device: types.SimpleNamespace(
        model=model, device=device
    )
    fake_reid.create_reid_encoder = fake_create_reid_encoder
    fake_boxmot = types.ModuleType("boxmot")
    fake_boxmot.reid = fake_reid
    monkeypatch.setitem(sys.modules, "boxmot", fake_boxmot)
    monkeypatch.setitem(sys.modules, "boxmot.reid", fake_reid)

    extractor = ReIDExtractor(device="cpu")

    assert extractor.backbone == "osnet"
    assert created["model"] == "osnet-x0-25-msmt17"

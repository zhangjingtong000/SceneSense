# 第三方依赖与许可

SceneSense 自身代码以 [MIT](LICENSE) 发布。本文列出运行/使用本项目时会拉进来的第三方组件，
以及它们的许可。**其中 `ultralytics`（YOLOv8）与 `boxmot` 是 AGPL-3.0**，这一点决定了本项目的
分发方式，请先读下面两节。

## 一、关键结论：组合分发受 AGPL-3.0 约束

| 组件 | 版本 | 许可 | 在项目里的角色 | 是否必需 |
|---|---|---|---|---|
| [`ultralytics`](https://github.com/ultralytics/ultralytics) | 8.x | **AGPL-3.0** | YOLOv8 检测与姿态推理（`perception/detector.py`） | 必需 |
| [`boxmot`](https://github.com/mikel-brostrom/boxmot) | 10.x / 26.x | **AGPL-3.0** | 默认 OSNet ReID 后端（`perception/tracker.py`） | 可选（缺失回退 ResNet18） |

本项目**自身**的代码（`perception/`、`agents/`、`kg/`、`web/`、`pipeline.py` 等）是 MIT。
但只要把本项目与 `ultralytics` / `boxmot` 一起分发（打包镜像、发安装包、卖设备），
或者把它做成**网络服务**供他人使用，整体就落入 AGPL-3.0 的约束：

- 必须以 AGPL-3.0 提供整体源码，包括你对本项目做的修改；
- 通过网络提供服务时，用户有权获取对应源码（AGPL 第 13 条）。

**要闭源商用**，有两条路：

1. 替换检测与 ReID 实现，去掉这两个依赖。检测可换 Apache-2.0 / MIT 的模型与推理库
   （例如 ONNX Runtime + 自训练模型），ReID 用 `torchreid`（MIT）或自实现 OSNet 链路；
   `perception/tracker.py` 已经把 ReID 做成可替换后端，`ultralytics` 的调用点集中在
   `perception/detector.py` 与 Web 后端的权重加载处。
2. 向 Ultralytics 与 boxmot 的作者分别取得商业授权。

> 另外注意：`yolov8n-pose.pt` / `yolov8l.pt` 这些**权重**同样由 Ultralytics 以 AGPL-3.0 发布，
> 与代码许可一致。仓库不分发大权重，但首次运行会自动下载。

## 二、其余依赖

| 组件 | 版本（实测） | 许可 | 用途 |
|---|---|---|---|
| Python | 3.10+ | PSF-2.0 | 运行时 |
| [`torch`](https://pytorch.org) | 2.x | BSD-3-Clause | 推理后端 |
| [`torchvision`](https://github.com/pytorch/vision) | 0.15+ | BSD-3-Clause | ResNet18 回退特征 |
| [`torchreid`](https://github.com/KaiyangZhou/deep-person-reid) | 0.2.5 | MIT | OSNet 自实现链路（可选） |
| [`opencv-python`](https://github.com/opencv/opencv-python) | 4.8+ | Apache-2.0 | 视频 IO、光流、绘制 |
| [`numpy`](https://numpy.org) | 1.24+ | BSD-3-Clause | 数值计算 |
| [`scipy`](https://scipy.org) | 1.10+ | BSD-3-Clause | 匈牙利匹配 |
| [`filterpy`](https://github.com/rlabbe/filterpy) | 1.4.5 | MIT | 卡尔曼滤波 |
| [`networkx`](https://networkx.org) | 3.x | BSD-3-Clause | 场景知识图谱 |
| [`openai`](https://github.com/openai/openai-python) | 1.x | Apache-2.0 | DeepSeek 兼容客户端 |
| [`fastapi`](https://fastapi.tiangolo.com) | 0.110+ | MIT | Web 后端（可选） |
| [`uvicorn`](https://www.uvicorn.org) | 0.27+ | BSD-3-Clause | ASGI 服务器（可选） |
| `python-multipart` | 0.0.9+ | Apache-2.0 | 上传表单解析（可选） |
| [`matplotlib`](https://matplotlib.org) | 3.x | PSF-based（BSD 兼容） | 架构图 / KG 可视化（可选） |
| [`motmetrics`](https://github.com/cheind/py-motmetrics) | 1.4 | MIT | IDF1 / MOTA 指标（可选） |
| [`trackeval`](https://github.com/JonathonLuiten/TrackEval) | 1.3 | MIT | HOTA 指标（可选） |
| [`pytest`](https://pytest.org) | 7.4+ | MIT | 测试（开发用） |
| [`ruff`](https://github.com/astral-sh/ruff) | 0.15 | MIT | 静态检查（开发用） |

## 三、不分发的数据与权重

| 资产 | 许可 / 状态 |
|---|---|
| `yolov8n-pose.pt`、`yolov8l.pt` | Ultralytics AGPL-3.0，首次运行自动下载，不入库 |
| `osnet_x0_25_msmt17.pt` | 在 MSMT17 上训练；MSMT17 的图像与权重再分发受限，仓库不入库，需自行获取 |
| MOT17 数据集 | 学术使用条款，仓库不入库 |
| `assets/demo/sample.mp4` | 项目自制的合成素材（几何图形，无真人），随仓库以 MIT 分发 |

## 四、依赖变更时

新增依赖前先看许可：AGPL 依赖会让整体分发受 AGPL 约束，GPL 依赖有类似效果；
Apache-2.0 / MIT / BSD 可以放心组合。更新本文件时请顺手核对 `requirements.txt`
与 `pip show <pkg>` 里的 License 字段。

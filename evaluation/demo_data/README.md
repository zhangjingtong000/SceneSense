# demo_data：合成 MOT 序列（适配器演示用）

**不是真实数据集，别拿这些数字当跟踪性能。** 它们是 `evaluation/make_demo_sequence.py`
生成的合成场景，用途是验证 `evaluation/mot_metrics.py` 能把「好」和「坏」分开。

| 文件 | 内容 |
| --- | --- |
| `gt.txt` | 真值：6 个目标 × 200 帧 = 1200 框，帧号 **1 基**，`frame,id,x,y,w,h,mark,class,visibility` |
| `perfect.txt` | 预测（0 基帧号），与 GT 逐框一致 → IDF1 / MOTA / HOTA 均为 1.0 |
| `idsw.txt` | 预测（0 基帧号），2 / 4 号各换 1 次 ID、6 号换 2 次 → IDF1 降、IDSW=4 |
| `dropout.txt` | 预测（0 基帧号），1 号漏 30 帧、5 号漏 50 帧、3 号每 7 帧漏 1 帧 → FN=109 |
| `expected.json` | 上面三种预测的实测指标（由脚本算出，测试会比对它） |

坐标是 MOTChallenge 的左上角 + 宽高。GT 用 1 基、预测用 0 基，正好对上
`mot_metrics` 的默认偏移（`--gt-frame-offset -1`），所以这样就能跑：

```bash
python -m evaluation.mot_metrics \
  --gt evaluation/demo_data/gt.txt \
  --pred evaluation/demo_data/idsw.txt
```

**不要手改这些 txt。** 它们是生成器的输出，`TestDemoSequence::test_demo_data_is_reproducible`
会检查重新生成的结果与仓库文件逐字节一致。要改场景就改生成器，然后：

```bash
python -m evaluation.make_demo_sequence --report
```

总体积约 210 KB。

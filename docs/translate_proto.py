import re

with open('prototype-timeline.html', 'r', encoding='utf-8') as f:
    html = f.read()

replacements = [
    # Video pane
    ('Person-1 selected · frame 847', '已选中 Person-1 · 第 847 帧'),
    ('Video canvas placeholder', '视频画布（占位）'),
    ('YOLO detection boxes + selected Person highlight + trajectory overlay', 'YOLO 检测框 + 选中人物高亮 + 轨迹叠加'),
    ('⏺ Person-1 path — 312 tracked frames across 2 segments', '⏺ Person-1 轨迹 — 跨越 312 帧 · 2 个片段'),

    # Heatmap
    ('Event heatmap — click to jump', '事件热力 — 点击跳转'),

    # Tabs
    ('<span>Interactions</span>', '<span>交互</span>'),
    ('<span>Alerts (3)</span>', '<span>告警 (3)</span>'),
    ('<span>KG</span>', '<span>图谱</span>'),

    # Subgraph
    ('Evidence subgraph · Person-1', '证据子图 · Person-1'),
    ('>Entity<', '>实体<'),
    ('>Event<', '>事件<'),
    ('>Behavior<', '>行为<'),
    ('>Loitering<', '>徘徊<'),
    ('frames 780–900 · confidence 0.87 · agent: anomaly', '帧 780–900 · 置信度 0.87 · 来源: 异常检测Agent'),
    ('>3 contacts<', '>3 次接触<'),
    ('Events 8, 12, 14 · total duration 47s · cumulative score 0.72', '事件 8, 12, 14 · 累计 47 秒 · 综合得分 0.72'),
    ('>interacts<', '>交互<'),
    ('frames 380–440 · brief contact · agent: interaction', '帧 380–440 · 短暂接触 · 来源: 交互Agent'),

    # Detail panel
    ('Person-1 detail', 'Person-1 详情'),
    ('Entity ID', '实体 ID'),
    ('Track IDs', '跟踪 ID'),
    ('Occlusion', '遮挡'),
    ('frames 421-508 · reid 0.84', '帧 421-508 · 重识别 0.84'),
    ('First seen', '首次出现'),
    ('Last seen', '最后出现'),
    ('Interactions', '交互'),
    ('Behaviors', '行为'),
    ('Loitering (0.87)', '徘徊 (0.87)'),

    # Reasoning
    ('Reasoning path', '推理路径'),
    ('track #3 → occlusion gap (87 frames) → ReID match score 0.84 (IoU 0.91 + appearance 0.78 + motion continuity) → track #17 → merged as Person-1',
     'track #3 → 遮挡缺口 (87 帧) → ReID 匹配分 0.84 (IoU 0.91 + 外观 0.78 + 运动连续性) → track #17 → 合并为 Person-1'),

    # Bottom bar
    ('Person-1 path: 312 frames · 2 segments · 1 occlusion gap',
     'Person-1 轨迹: 312 帧 · 2 个片段 · 1 处遮挡缺口'),
    ('tentative merge candidate: Person-3 (score 0.62 — pending evidence)',
     '待确认合并候选: Person-3 (得分 0.62 — 证据收集中)'),
]

for old, new in replacements:
    html = html.replace(old, new)

with open('prototype-timeline.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('Done — all text replaced')

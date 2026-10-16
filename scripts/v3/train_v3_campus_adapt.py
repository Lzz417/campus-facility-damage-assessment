"""V3-CAMPUS-ADAPT 训练入口（**单一模型，参数与 V2 对齐，不新增增强策略**）。

这是 **continued domain-adaptation / fine-tuning from V2**：
初始化权重是 V2-AUG-RERUN1 的 best.pt，**不是**从官方预训练权重重新训练。

训练数据 = MBDD split_v2 train originals (7,752) + Smart Campus V3 train (1,536) = 9,288。
禁止：split_v2 val/test、Smart Campus V3 val/test、V2 六倍离线增强集、HU。

用法：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe \
    docs/evidence/v3_domain_adaptation/_scripts/train_v3_campus_adapt.py
"""
import json
import pathlib
import sys
import time

from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parents[4]
INIT_W = ROOT / "experiments/runs_v2/V2-AUG-RERUN1/weights/best.pt"
DATA = ROOT / "data/v3_external/v3_campus_adapt.yaml"
PROJ = ROOT / "experiments/runs_v3"
NAME = "V3-CAMPUS-ADAPT"

ARGS = dict(
    data=str(DATA),
    epochs=30,
    imgsz=640,
    batch=32,
    device=0,
    workers=8,
    seed=0,
    deterministic=True,
    optimizer="auto",          # 与 V2 一致
    lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, cos_lr=False,
    # 在线增强：与 V2 完全相同（Ultralytics 默认项），本轮不新增策略
    mosaic=1.0, scale=0.5, fliplr=0.5, flipud=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    translate=0.1, degrees=0.0, shear=0.0, perspective=0.0,
    erasing=0.4, close_mosaic=10, mixup=0.0, copy_paste=0.0,
    save_period=1,             # 每个 epoch 都留 checkpoint，供事后按预登记规则选择
    patience=100,              # 30 epoch 内不会早停
    val=True, plots=True, amp=True, pretrained=True,
    project=str(PROJ), name=NAME, exist_ok=False,
)


def main():
    if not INIT_W.exists():
        print("[FATAL] 初始化权重不存在：%s" % INIT_W)
        return 2
    if (PROJ / NAME).exists():
        print("[FATAL] 运行目录已存在，拒绝覆盖：%s" % (PROJ / NAME))
        return 2
    import hashlib

    def sh(p):
        return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

    print("[info] continued domain-adaptation / fine-tuning from V2")
    print("[info] init weights : %s" % INIT_W)
    print("[info] init sha256  : %s" % sh(INIT_W))
    print("[info] data yaml    : %s (%s)" % (DATA, sh(DATA)[:16]))
    print("[info] started at   : %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    sys.stdout.flush()
    model = YOLO(str(INIT_W))
    model.train(**ARGS)
    print("[DONE] %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

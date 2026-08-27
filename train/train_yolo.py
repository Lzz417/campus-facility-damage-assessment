"""YOLOv11 缺陷检测训练脚本（第一级检测器）。

用法：
  python train/train_yolo.py --data data/splits/data.yaml \
      --weights yolo11n.pt --epochs 100 --batch 16 --imgsz 640

训练完成后，将 runs/detect/train/weights/best.pt 复制到 weights/best.pt，
并在 configs/config.yaml 的 detect.model 中指向它。
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser(description="训练 YOLOv11 检测器")
    ap.add_argument("--data", default="data/splits/data.yaml")
    ap.add_argument("--weights", default="weights/yolo11n.pt",
                    help="预训练权重：weights/yolo11n.pt（已本地下载，避免联网重下）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="train")
    ap.add_argument("--cache", default="false",
                    help="数据缓存：false 不缓存（SSD 用，默认）；disk 合并为单个 .npy "
                         "顺序读（机械盘用）；ram/true 载入内存")
    args = ap.parse_args()
    cache = False if args.cache.lower() in ("false", "none", "0", "off") else args.cache

    from ultralytics import YOLO
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        cache=cache,
        # 训练即测试，保存最优权重
        val=True,
    )
    print("[训练] 完成。最优权重已由 ultralytics 保存到 runs/detect/<name>/weights/best.pt")


if __name__ == "__main__":
    main()

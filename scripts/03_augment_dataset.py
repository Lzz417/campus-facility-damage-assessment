"""数据集二次改造 —— 多工况仿真退化增强（创新点1，纯 opencv/numpy 自研实现）。

对训练集图片批量生成六类真实工况退化变体：
  暗光(dark)、逆光/眩光(backlight)、雾(fog)、阴影(shadow)、
  局部遮挡(occlusion)、镜头模糊(blur)。

区别于普通翻转/缩放，这些退化模拟校园真实巡检中常见的低质量成像条件，
用于缩小公开数据集与校园真实场景的域差距，提升 YOLO 检测器的鲁棒性。
全部用 opencv + numpy 手写（非 albumentations），体现创新点自研属性。

注意：
  * 只对【训练集】增强，避免同一原图与退化变体泄漏进 val/test 而虚高指标；
  * 六类退化均为光度/遮挡类（不改几何），因此 bbox 标签不变，直接复制；
  * 标签为 YOLO 检测格式（cls cx cy w h，归一化），来自 02 清洗结果。

输入结构（划分后）：
  data/splits/images/train/  *.jpg
  data/splits/labels/train/  *.txt
输出：在 train 目录内追加退化变体，如 img_dark.jpg / img_dark.txt。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ------------------------- 自研退化算子（均接收 BGR uint8，返回 BGR uint8） -------------------------

def _dark(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """暗光：降低对比度(alpha<1) + 整体亮度下压(beta<0)。"""
    alpha = rng.uniform(0.35, 0.70)
    beta = rng.uniform(-60, -20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _backlight(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """逆光/眩光：顶部叠加一个径向强光晕（模拟正对光源/太阳）。"""
    h, w = img.shape[:2]
    cx = rng.uniform(0.25 * w, 0.75 * w)
    cy = rng.uniform(-0.25 * h, 0.15 * h)          # 光源在画面上方甚至画外
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    maxd = float(np.sqrt(max(cx, w - cx) ** 2 + (h - cy) ** 2))
    flare = np.clip(1.0 - dist / maxd, 0.0, 1.0) ** 2 * rng.uniform(110, 200)
    out = img.astype(np.float32) + flare[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _fog(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """雾：向浅灰背景线性混合 + 轻微高斯模糊（模拟散射导致对比度下降）。"""
    intensity = rng.uniform(0.15, 0.45)
    fog_color = rng.uniform(180, 225)
    foggy = img.astype(np.float32) * (1.0 - intensity) + fog_color * intensity
    foggy = cv2.GaussianBlur(foggy, (7, 7), 0)
    return np.clip(foggy, 0, 255).astype(np.uint8)


def _shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """阴影：1~2 个软边椭圆阴影块覆盖画面局部（模拟树影/建筑阴影）。"""
    h, w = img.shape[:2]
    shadow_mask = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(1, 3))):
        cx = rng.uniform(0, w)
        cy = rng.uniform(0.3 * h, h)
        rx = rng.uniform(0.15 * w, 0.55 * w)
        ry = rng.uniform(0.05 * h, 0.28 * h)
        e = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        shadow_mask[e <= 1.0] += 1.0
    shadow_mask = np.clip(shadow_mask, 0.0, 1.0)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=max(h, w) * 0.05)
    factor = 1.0 - rng.uniform(0.30, 0.60) * shadow_mask[..., None]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _occlusion(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """局部遮挡：1~4 个随机黑色矩形块（模拟杂物/镜头脏污遮挡）。"""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(int(rng.integers(1, 5))):
        bh = int(rng.uniform(0.03 * h, 0.13 * h))
        bw = int(rng.uniform(0.03 * w, 0.13 * w))
        x = int(rng.uniform(0, max(1, w - bw)))
        y = int(rng.uniform(0, max(1, h - bh)))
        out[y:y + bh, x:x + bw] = 0
    return out


def _motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    """生成一条方向为 angle_deg 的归一化线段卷积核（运动模糊）。"""
    k = np.zeros((size, size), np.float32)
    c = (size - 1) / 2.0
    rad = np.deg2rad(angle_deg)
    # 沿方向取 size 个采样点，落在线段上的像素置 1
    for t in np.linspace(-1, 1, size):
        x = int(round(c + t * np.cos(rad) * c))
        y = int(round(c + t * np.sin(rad) * c))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


def _blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """镜头模糊：高斯模糊 或 随机方向运动模糊（二选一）。"""
    if rng.random() < 0.5:
        k = int(rng.integers(7, 16))
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(img, (k, k), 0)
    k = int(rng.integers(9, 16))
    kernel = _motion_kernel(k, rng.uniform(0, 180))
    return cv2.filter2D(img, -1, kernel)


DEGRADATIONS = {
    "dark": _dark,
    "backlight": _backlight,
    "fog": _fog,
    "shadow": _shadow,
    "occlusion": _occlusion,
    "blur": _blur,
}


def main():
    ap = argparse.ArgumentParser(description="多工况仿真退化增强（仅训练集，纯opencv/numpy自研）")
    ap.add_argument("--images", default="data/splits/images/train")
    ap.add_argument("--labels", default="data/splits/labels/train")
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    rng = np.random.default_rng(args.seed)

    print(f"[增强] 退化类型: {list(DEGRADATIONS.keys())}")
    files = sorted(p for p in img_dir.glob(f"*{args.ext}")
                   if (lbl_dir / f"{p.stem}.txt").exists())
    print(f"[增强] 训练集 {len(files)} 张，每张生成 {len(DEGRADATIONS)} 个变体")

    stats = {k: 0 for k in DEGRADATIONS}
    for img_path in tqdm(files, desc="增强"):
        stem = img_path.stem
        lbl_src = lbl_dir / f"{stem}.txt"
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        for name, func in DEGRADATIONS.items():
            out_img = img_dir / f"{stem}_{name}{args.ext}"
            out_lbl = lbl_dir / f"{stem}_{name}.txt"
            if out_img.exists():
                continue
            aug = func(image, rng)
            cv2.imwrite(str(out_img), aug)
            shutil.copy2(lbl_src, out_lbl)   # 光度/遮挡退化不改几何，标签原样复制
            stats[name] += 1

    total = sum(stats.values())
    print("[增强] 完成。各退化变体数量:", stats)
    print(f"[增强] 训练集现有 {len(files) + total} 张（原 {len(files)} + 变体 {total}）")


if __name__ == "__main__":
    main()

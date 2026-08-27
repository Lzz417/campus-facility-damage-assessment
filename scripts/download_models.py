"""下载运行所需模型权重（SAM2 + YOLO11 基座）。

别人 clone 仓库后，运行本脚本即可获得全部推理所需权重（<1GB）：
  - sam2.1_hiera_large.pt   （SAM2 零样本分割，官方 898MB）
  - yolo11n.pt              （YOLO11 基座，训练/复现用，可选）

用法:
  python scripts/download_models.py
  python scripts/download_models.py --proxy http://127.0.0.1:10809   # 国内网络走代理
  python scripts/download_models.py --skip-yolo                       # 只下 SAM2（推理必需）

说明:
  * 使用 urllib 标准库实现，断点续传（已存在且大小正确则跳过）。
  * 训练好的 YOLO 权重 best.pt / best_orig.pt 已随仓库提交，推理无需训练。
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

# (文件名, 官方地址, 最小字节数, 说明)
TARGETS = [
    (
        "sam2.1_hiera_large.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        100 * 1024 * 1024,  # ~898MB
        "SAM2 零样本分割权重（推理必需）",
    ),
    (
        "yolo11n.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        5 * 1024 * 1024,  # ~5.4MB
        "YOLO11n 基座权重（训练/复现用，推理不需要）",
    ),
]


def _human(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n > 1024 * 1024 else f"{n / 1024:.1f} KB"


def download(url: str, dest: Path, min_bytes: int, proxy: str | None,
             opener: urllib.request.OpenerDirector) -> None:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  [跳过] {dest.name} 已存在（{_human(dest.stat().st_size)}）")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [下载] {dest.name} ({_human(min_bytes)} 起) <- {url}")
    if proxy:
        print(f"         走代理 {proxy}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(req, timeout=120) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = f"{done / total * 100:5.1f}%" if total else f"{_human(done)}"
            print(f"\r         {_human(done)} / {_human(total) if total else '?'}  {pct}", end="", flush=True)
        print()
    actual = dest.stat().st_size
    if actual < min_bytes:
        print(f"  [警告] {dest.name} 文件过小（{_human(actual)}），可能下载不完整，请重试。")
    else:
        print(f"  [完成] {dest.name} ({_human(actual)})")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载模型权重")
    ap.add_argument("--proxy", default=None,
                    help="HTTP/HTTPS 代理，如 http://127.0.0.1:10809")
    ap.add_argument("--skip-yolo", action="store_true",
                    help="只下载 SAM2（推理必需），跳过 YOLO 基座")
    args = ap.parse_args()

    proxy_handler = urllib.request.ProxyHandler(
        {"http": args.proxy, "https": args.proxy}) if args.proxy else urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)

    print("== 模型权重下载 ==")
    for name, url, minb, desc in TARGETS:
        if args.skip_yolo and name.startswith("yolo"):
            continue
        print(f"  [{desc}]")
        download(url, WEIGHTS / name, minb, args.proxy, opener)
    print("\n完成。推理所需：weights/best.pt（已随仓库提交）+ weights/sam2.1_hiera_large.pt（本次下载）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

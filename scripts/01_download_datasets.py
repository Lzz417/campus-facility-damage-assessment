"""公开数据集下载脚本。

来源（已核实）：
  - MBDD2025：Zenodo record 15622584（14,471 张建筑缺陷图，VOC bbox 标注）
      https://zenodo.org/records/15622584
  - CrackForest (CFD)：GitHub cuilimeng/CrackForest-dataset（道路裂缝，像素级掩码）
      https://github.com/cuilimeng/CrackForest-dataset

运行前请确保有正常外网。MBDD 体积较大（数 GB），请预留时间。
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm


def download_mbdd(out_dir: Path, record_id: str = "15622584"):
    """通过 Zenodo API 列出并下载 MBDD 全部文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    api = f"https://zenodo.org/api/records/{record_id}"
    print(f"[下载] 查询 Zenodo: {api}")
    r = requests.get(api, timeout=30)
    r.raise_for_status()
    files = r.json().get("files", [])
    print(f"[下载] MBDD 共 {len(files)} 个文件")
    for f in files:
        url = f["links"]["self"]
        name = f.get("key", url.split("/")[-1])
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size == f.get("size", 0):
            print(f"[跳过] {name}")
            continue
        print(f"[下载] {name} ({f.get('size', 0) / 1e6:.1f} MB)")
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as fp, tqdm(
                    total=total, unit="B", unit_scale=True, desc=name) as bar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fp.write(chunk)
                    bar.update(len(chunk))
    print("[下载] MBDD 完成")


def download_crackforest(out_dir: Path, repo: str = "https://github.com/cuilimeng/CrackForest-dataset.git"):
    """克隆 CrackForest。"""
    if (out_dir / ".git").exists():
        print("[跳过] CrackForest 已存在")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[下载] git clone CrackForest ...")
    subprocess.run(["git", "clone", "--depth", "1", repo, str(out_dir)], check=True)
    print("[下载] CrackForest 完成")


def main():
    ap = argparse.ArgumentParser(description="下载公开数据集")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--skip-mbdd", action="store_true")
    ap.add_argument("--skip-crackforest", action="store_true")
    args = ap.parse_args()
    raw = Path(args.out)
    if not args.skip_mbdd:
        download_mbdd(raw / "MBDD")
    if not args.skip_crackforest:
        download_crackforest(raw / "CrackForest")


if __name__ == "__main__":
    main()

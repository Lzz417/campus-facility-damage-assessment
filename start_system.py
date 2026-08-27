"""一键启动脚本（start_system.bat 调用）。

负责：环境检查、依赖校验、拉起 Streamlit 服务并打开浏览器。
中文提示放在这里，避免 cmd 批处理中文编码问题。
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app" / "app.py"
PORT = os.environ.get("STREAMLIT_PORT", "8501")
URL = f"http://localhost:{PORT}"

REQUIRED = ["cv2", "streamlit", "ultralytics", "sam2", "pandas", "numpy"]


def _bar(text: str, ch: str = "=", width: int = 56) -> str:
    return f"{ch * 4} {text} {ch * (width - len(text) - 6)}"


def main() -> int:
    print()
    print(_bar("校园设施损伤智能分割与量化评估系统"))
    print(_bar("YOLOv11 前置检测 + SAM2 分割 + 量化报表"))
    print(_bar("正在启动 Streamlit 服务...", "-"))
    print()

    # 1) 依赖检查
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[错误] 缺少依赖: {', '.join(missing)}")
        print("       请先安装依赖：")
        print("         pip install -r requirements.txt")
        print("       或参考 README.md（注意 torch 需从 CUDA 源安装）。")
        return 1

    # 2) 应用文件检查
    if not APP.exists():
        print(f"[错误] 未找到应用文件: {APP}")
        return 1

    # 3) 启动 Streamlit
    print(f"[启动] 浏览器将自动打开 {URL}")
    print(f"[提示] 首次加载 SAM2 模型约需 30-60 秒，请耐心等待。")
    print(f"[提示] 关闭本窗口即停止服务。")
    print()
    try:
        code = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(APP),
             "--server.port", PORT, "--server.headless", "false"],
            cwd=str(ROOT),
        ).returncode
    except KeyboardInterrupt:
        print("\n[结束] 用户中断，服务已停止。")
        return 0

    if code != 0:
        print(f"\n[提示] Streamlit 启动异常（退出码 {code}）。")
        print("       若为端口占用，可设置环境变量 STREAMLIT_PORT 换端口后重试；")
        print("       若为依赖问题，请按 README.md 检查环境。")
    return code


if __name__ == "__main__":
    sys.exit(main())

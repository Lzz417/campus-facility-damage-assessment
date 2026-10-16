# Third-Party Notices

## 本项目自有代码 / 文档的版权与许可

```
Copyright (C) 2026 Lzz417
```

本项目**自有代码与文档**按 **[AGPL-3.0](LICENSE)** 发布
（`LICENSE` 为 GNU 官方 AGPL-3.0 正文原样，未在其中插入自定义版权行）。
**第三方组件不因本项目采用 AGPL-3.0 而改变许可**，各自继续受其自身许可证约束。

---

本仓库包含或依赖以下第三方组件。**它们的版权与许可属于各自的作者，不属于本项目**；
本项目**不**对其主张任何所有权、也不把它们重新声明为本项目的许可。

---

## 1. Ultralytics YOLO（含本仓库发布的 YOLO 权重）

| 项 | 说明 |
|---|---|
| 组件 | `ultralytics` 软件包 + YOLO11 模型架构 |
| 上游 | https://github.com/ultralytics/ultralytics |
| **许可** | **AGPL-3.0**（GNU Affero General Public License v3.0） |
| 本仓库的关联 | `src/detect.py` 通过 `ultralytics` 进行推理；`experiments/` 的训练脚本基于它；<br>`weights/best.pt`、`weights/best_orig.pt` 是**基于 Ultralytics YOLO11 训练/微调得到的模型权重** |
| 上游许可说明 | Ultralytics 官方说明其软件**及其训练/微调模型**默认采用 AGPL-3.0；<br>商业闭源使用需另取 Ultralytics Enterprise License |

**重要**：本项目**曾以 MIT 发布**。由于仓库直接使用 Ultralytics YOLO 并发布了基于它训练出的权重，
"整个仓库为 MIT" 的表述与上游 AGPL-3.0 存在**许可冲突**。本项目**自有的代码**（`app/`、`src/`、
`scripts/`、`train/` 等由本项目作者编写部分）**拟调整为 AGPL-3.0 发布**以与之相容——见
`docs/release/RELEASE_PREP.md` 第 1 节。**该调整尚未执行**。

## 2. Meta SAM 2

| 项 | 说明 |
|---|---|
| 组件 | Segment Anything Model 2（`sam2`） |
| 上游 | https://github.com/facebookresearch/sam2 |
| **许可** | **Apache License 2.0** |
| 本仓库的关联 | `src/segment.py` 调用 SAM2 做 box-prompt 分割；<br>`weights/sam2.1_hiera_large.pt` 为其官方检查点（约 856 MB） |
| **是否随仓库分发** | **否**。SAM2 检查点**不**在本仓库中（`.gitignore` 已忽略），需用户自行从上游获取 |

## 3. 数据集

### 3.1 MBDD2025

| 项 | 说明 |
|---|---|
| 名称 | MBDD2025 — A Deep Learning-Oriented Dataset for Diverse Building Defect Detection from UAV Images |
| **许可** | **CC BY 4.0**（Creative Commons Attribution 4.0 International） |
| 本仓库的关联 | 本项目的主要训练/评价数据来源 |
| **是否随仓库分发原始图像** | **否**。原始数据集不在本仓库中；<br>但 `assets/demo_*.jpg` 是**本项目基于 MBDD2025 图像生成的演示对比图**（见 §3.3） |
| 要求 | CC BY 4.0 允许再分发，**须署名**。引用信息见 `_mbdd_readme.md` |

### 3.2 Smart Campus Dataset（V3 使用的第二数据来源）

| 项 | 说明 |
|---|---|
| 名称 | Smart Campus Dataset（Roboflow Universe，workspace `emily-s-workspace`，version 9） |
| **标称许可** | **CC BY 4.0**（由上传者在 `data.yaml` / `README.dataset.txt` 中声明） |
| **是否随仓库分发原始图像** | **否**，且**有意不分发** |
| 不分发的原因 | 本项目的审计**发现该数据集内含第三方水印/图库素材**（如 shutterstock / alamy / iStock / depositphotos 等）<br>以及 AI 生成图、实验室试片、屏幕截图。这些内容的版权**不在**该上传者手中，<br>其 CC BY 4.0 声明**覆盖不了**它们。因此本仓库**只发布来源说明、清洗方法与统计结果**，<br>**不重新分发其任何原始图片** |
| 来源说明限制 | 该数据集页面只标注 "Provided by a Roboflow user"，**没有采集地点/采集方法说明**。<br>因此本项目**不**将其描述为"已证实的真实校园数据" |

### 3.3 HU Infrastructure Cracks Dataset

| 项 | 说明 |
|---|---|
| 名称 | HU Infrastructure Cracks Dataset（The Hashemite University, Jordan） |
| **许可** | **CC BY 4.0** |
| DOI | `10.5281/zenodo.20829347`（concept DOI） |
| **是否随仓库分发** | **否**。本轮**未下载其图像**、也未用于任何训练或评价（仅取得 metadata 用于可行性审计） |

## 4. `assets/` 中的图像

| 文件 | 来源 | 许可与署名 |
|---|---|---|
| `assets/demo_peeling.jpg` | **本项目自制**："原图 ｜ 分割叠加" 双联对比图（`scripts/make_demo_assets.py` 用 `cv2.hconcat` 合成）；**原图部分来自 MBDD2025** | 原图 **CC BY 4.0**，在此署名 MBDD2025；叠加部分为本项目模型输出 |
| `assets/demo_crack.jpg` | 同上 | 同上 |
| `assets/demo_rust.jpg` | 同上 | 同上 |

**MBDD2025 署名与引用**：本仓库使用 MBDD2025 数据集（**CC BY 4.0**）。
引用信息见仓库根 [`_mbdd_readme.md`](_mbdd_readme.md)。
CC BY 4.0 允许再分发，**条件是署名**——上述三张图因含 MBDD2025 图像，特此署名。

**`assets/banner.jpg` 已移除**：该图在仓库内**找不到生成脚本**，来源无法确认，
为不承担版权风险已从当前版本中删除（`README.md` 中相应引用已替换为自制图）。
**本仓库不通过改写 Git 历史来抹除该文件**——它仍存在于历史提交中。
后续如需顶部展示图，只使用：本项目自制的图 / Web UI 截图 / 自绘架构示意图 /
许可证明确且已正确署名的内容。

## 5. 第三方 Python 依赖

完整的依赖清单见 `requirements.txt`（如 `torch`、`opencv-python`、`numpy`、`pandas`、
`streamlit`、`openpyxl`、`scipy` 等）。**各自适用其自身的许可**，本文件不逐一列示；
如需逐项声明，请以各自官方仓库的许可为准。

---

## 声明方式说明（避免误标）

- 本文件**不**把任何第三方组件标为本项目的版权。
- 本项目**自有代码**的许可（拟为 AGPL-3.0）**仅覆盖本项目作者编写的内容**，
  **不覆盖**上述第三方组件及其衍生权重。
- 若同一文件内既有本项目代码又有第三方代码，以文件内的头部声明/来源说明为准。

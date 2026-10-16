# `weights/` 说明

本目录存放模型权重。**本仓库只包含下列两个权重**；其余权重（SAM2 检查点、V2/V3 正式 detector、
全部 epoch checkpoint）**不在仓库中**，需按说明自行获取。

---

## 1. 仓库内已有的权重（历史权重）

### `weights/best.pt`

| 项 | 值 |
|---|---|
| **用途** | 本项目的**默认推理权重**；`configs/config.yaml` 的 `detect.model` 指向它，Web demo 与脚本默认使用它 |
| **对应实验阶段** | **V2 之前的早期训练**（使用 `data/splits/` 旧随机划分 + 增强集） |
| **训练数据配置** | `data/splits`（**旧的随机划分**），增强集 |
| **类别数 / 类别** | `nc=3`，`{0: peeling, 1: crack, 2: rust}` |
| **SHA256** | `f8c9ab585e456845e64a91ae046baa569ffc9c276800c875461e8cdbd37d497f` |
| **字节数** | 5,483,475 |
| **基于** | **Ultralytics YOLO11**（YOLO11n 架构，微调） |
| **License** | **AGPL-3.0**（继承自 Ultralytics；见仓库根 `THIRD_PARTY_NOTICES.md`） |
| **是否为历史权重** | **是（历史权重）** |
| **身份** | **当前 demo 使用的历史 detector 权重**；<br>**不是** V2 正式冻结 detector；**不是** V3 adapted model |

> ⚠️ **关于 `configs/config.yaml` 注释里的 "mAP50 0.858"**：
> 那是该权重在**旧随机划分** val 上的分数。本项目后续通过 split_v2 严格划分证明，
> 旧随机划分**存在同源近重复泄漏**，因此该 0.858 **不可与后续任何冻结结果相比较**。
> 详见 `docs/methodology/` 下的 V2 评价摘要。

### `weights/best_orig.pt`

| 项 | 值 |
|---|---|
| **用途** | **历史对照权重**：在**未增强**（original-only）配置下训练的版本，用于与增强版对照 |
| **对应实验阶段** | **V2 之前的早期训练**（使用 `data/splits/` 旧随机划分，original-only） |
| **训练数据配置** | `data/splits`（旧随机划分），**未增强** |
| **类别数 / 类别** | `nc=3`，`{0: peeling, 1: crack, 2: rust}` |
| **SHA256** | `dbc16595b3b90491f9dad0b242a813c5958f7c489485fa3cf6f8aad3bb4a0e43` |
| **字节数** | 5,483,539 |
| **基于** | **Ultralytics YOLO11**（YOLO11n 架构，微调） |
| **License** | **AGPL-3.0**（继承自 Ultralytics） |
| **是否为历史权重** | **是（历史权重）** |
| **身份** | 历史对照权重；**不是** V2 正式冻结 detector；**不是** V3 adapted model；**不用于** demo |

> 两者**都不是**本项目 V2 正式评价所用的 detector
> （正式 detector 是 `experiments/runs_v2/V2-AUG-RERUN1/weights/best.pt`，**不在本仓库中**）。
> 因此**不要**用这两个权重去复现 V2 的正式评价数字。

## 2. 本目录中**没有**、需要自行获取的权重

| 权重 | 获取方式 | 放置位置 | 许可 |
|---|---|---|---|
| `sam2.1_hiera_large.pt`（≈856 MB） | 从 Meta SAM 2 官方仓库下载 | `weights/sam2.1_hiera_large.pt` | Apache-2.0 |
| `yolo11n.pt`（≈5.4 MB，Ultralytics 官方预训练） | Ultralytics 官方发布 | `weights/yolo11n.pt` | AGPL-3.0 |
| V2 正式 detector | 由本项目训练流程产生 | `experiments/runs_v2/V2-AUG-RERUN1/weights/best.pt` | AGPL-3.0 |
| V3 适配模型（`V3-CAMPUS-ADAPT` epoch 27） | 由本项目训练流程产生 | `experiments/runs_v3/V3-CAMPUS-ADAPT/weights/epoch26.pt` | AGPL-3.0 |

**本仓库不发训练/评价过程中的任何 checkpoint**（`experiments/runs_*/**` 已在 `.gitignore` 中忽略），
也不发 SAM2 检查点。模型体积较大，若日后需要发布，建议作为 **GitHub Release asset** 附加，
**不要**直接写入 Git 历史。

## 3. 许可提示

本目录内两个权重均**基于 Ultralytics YOLO11**（上游 AGPL-3.0）。
**以 MIT 再分发这些权重会与上游许可冲突**——仓库主许可**已由 MIT 调整为 AGPL-3.0**（见根 `LICENSE`），
详见 `docs/release/RELEASE_PREP.md` 与 `THIRD_PARTY_NOTICES.md`。

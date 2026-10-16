# Smart Campus V3 test —— 最终比较协议（**任何 test 推理之前预登记**）

**登记时间**：`2026-10-16`（**在第一次 test 推理之前**）
**状态**：**当前尚未解封 test**。本文件先定死"解封后怎么比"，避免看到结果再定规则。

---

## 1. 对象：两个**事先固定**的模型

| 角色 | 模型 | 权重 |
|---|---|---|
| **Baseline** | `V2-AUG-RERUN1` | `experiments/runs_v2/V2-AUG-RERUN1/weights/best.pt` |
| **Adapted** | `V3-CAMPUS-ADAPT` **selected epoch 27** | `experiments/runs_v3/V3-CAMPUS-ADAPT/weights/epoch26.pt` |

两者均已在 `model_frozen/` 冻结。**一次解封、一次运行、同时评价两者。**

## 2. Primary comparison：标准 mAP（两者参数必须逐项相同）

| 参数 | 值 |
|---|---|
| conf | **0.001** |
| NMS IoU | **0.7** |
| max_det | **300** |
| imgsz | **640** |
| evaluator | `_scripts/eval_v3_checkpoints.py --mode map`（**同一个脚本**） |
| 数据 | `Smart Campus V3 test`，manifest = `v3_split/smart_campus_v3_test_manifest.csv`（**同一份**） |

报告：**standard mAP50**、**standard mAP50-95**、**per-class AP50**。

**只有在上述各项完全一致的前提下，mAP 差值才可以直接解释为
"两个冻结模型在同一 held-out test 上的差异"**——这句是本节存在的唯一理由。

## 3. 工作点结果（另列，不参与 primary 结论）

| 模型 | 工作点阈值 | 来源 |
|---|---|---|
| V2 baseline | **conf 0.050** | 在 **Smart Campus V3 val** 上按 V3 同一套规则校准并冻结（`v2_on_sc_val_baseline.json`） |
| V3 adapted | **conf 0.200** | 已冻结（`model_frozen/v3_model_freeze.json`） |

匹配 IoU 0.5；NMS iou 0.45；max_det 30；imgsz 640。分别报告：

- **P / R / F1**（汇总）
- **per-class P / R / F1**
- **mean boxes/image**
- **no-prediction image rate**（该模型在该图上一框未出的比例）

**必须同时写明的说明**：两个模型的工作点阈值**是在同一个 Smart Campus V3 val 上、
按同一条规则（F1 最大 → 并列带 ≤0.005 → recall 更高 → 阈值更高）分别校准**的，
选阈代码是同一段（`_scripts/select_threshold.py`）。因此两者工作点指标可以并列比较，
但这是**在各自校准过的同一 val 上**的比较，不能与 primary mAP 混为一谈。

## 4. 拆封纪律（**在任何 test image 被模型读取之前**）

先写 `docs/evidence/v3_domain_adaptation/model_frozen/v3_test_unseal.json`，至少含：

| 字段 | 值 |
|---|---|
| `unseal_timestamp` | 运行时写入 |
| `unsealed_by` | **必须由真人提供**（与 V2 的做法一致，agent 不代填） |
| `smart_campus_v3_test_manifest_sha256` | `smart_campus_v3_test_manifest.csv` 的 SHA256 |
| `v2_weights_sha256` | V2 baseline 权重哈希 |
| `v3_weights_sha256` | V3 selected epoch 27 权重哈希 |
| `v2_frozen_threshold` | 0.050 |
| `v3_frozen_threshold` | **0.200** |
| `evaluator_sha256` | `eval_v3_checkpoints.py` 的 SHA256 |
| `comparison_protocol_sha256` | 本文件的 SHA256 |

脚本会**先写这个文件、再构造模型**；若已存在则只读不覆盖。

## 5. 不允许（解封之后）

- 修改模型、改 threshold、改 split、改类别映射、重新训练；
- 删除困难样本；
- 因 test 表现选择 V2/V3（两者都已固定，test 只用于报告差值）；
- 运行 MBDD `split_v2` test；
- 本轮**不加入 SAM2 segmentation**（Smart Campus 没有人工像素 GT，做不了分割评价）；
- 根据 test 结果回头改动任何已冻结物。

## 6. 程序错误导致重跑

沿用 V2 的失败记录纪律：归档失败产物、写明**失败阶段**、说明修改是否影响推理/指标，
中间结果不得进入最终报告；重跑写进 rerun 台账。

## 7. 收尾

**test 评价完成即停止实验，不再优化 V3。**
报告必须带上限制说明：结果受数据来源（Roboflow 用户上传、页面无采集地点说明）、
类别映射、以及 test 规模（329 张、单一数据集单一采集）的限制。

# V3-CAMPUS-ADAPT 模型选择协议（**训练前预登记**）

**登记时间**：`2026-10-16`（**在启动训练之前**写入）
**约束**：本协议一经登记，训练与评价过程中不得修改规则、不得事后补规则。
若事后发现规则有缺陷，**只能如实记录并停止**，不得边看结果边改规则。

---

## 1. 候选与评价范围

| 项 | 值 |
|---|---|
| 候选 | `V3-CAMPUS-ADAPT` 的**每个 epoch checkpoint**（`save_period=1`，共 30 个） |
| 开发集 A | MBDD `split_v2 val`，**n = 2,833** |
| 开发集 B | Smart Campus V3 val，**n = 329**（冻结件 `smart_campus_v3_val.txt`） |
| 评价器 | **同一个 post-hoc evaluator**，对每个 checkpoint 在 A、B 上各跑一次，口径完全一致 |
| 指标 | `mAP50`、`mAP50-95`、逐类 `AP50` |

**禁止读取**：MBDD `split_v2 test`、`Smart Campus V3 test`。
Ultralytics 训练过程中自带的 val（写进 `data yaml` 的那个）**不作为选择依据**，
它只是训练期的内部记账；**正式指标一律来自 post-hoc evaluator**。

## 2. 基线（用于 forgetting guard）与其出处

| 项 | 值 |
|---|---|
| 基线模型 | `V2-AUG-RERUN1`（`experiments/runs_v2/V2-AUG-RERUN1/weights/best.pt`） |
| 基线 **MBDD val mAP50** | **0.1185** |
| 出处 | `docs/evidence/training_v2/training_v2_results.csv`：`split_v2/val`、`n=2833`、`mAP50=0.1185`、`mAP50-95=0.0571`、`P=0.2918`、`R=0.1500`；原始响应见 `docs/evidence/training_v2/raw/V2-AUG-RERUN1_diag_eval.json` |
| 逐类 AP50（基线） | peeling 0.0810、crack 0.0622、rust 0.2125 |
| guard 阈值 | **0.0948 = 0.1185 × 0.80** |

**为什么用 0.1185 而不是训练日志里的 0.1201**：0.1201 是 Ultralytics 在训练过程中的内部 val
（best epoch），而 0.1185 是**用 post-hoc evaluator 在 split_v2 val 上重算的值**。
本协议的候选 checkpoint 也用 post-hoc evaluator 测，
**基线必须与候选同口径**，否则 guard 不可比。

## 3. 选择规则（预登记）

对每个 checkpoint 计算 `A_mAP50`（MBDD val）与 `B_mAP50`（Smart Campus val）。

1. **Forgetting guard**：`A_mAP50 ≥ 0.0948` 才算合格候选。
2. **Primary**：在合格候选中取 **`B_mAP50` 最大**者。
3. **Tie-break（`B_mAP50` 相差 ≤ 0.005 时依次比较）**：
   1. `B_mAP50-95` 更高者；
   2. 仍接近则取 **更早的 epoch**。
4. **失败模式**：若**没有任何** checkpoint 满足 guard →
   报告 **`no checkpoint passed the pre-registered forgetting guard`**，
   **不放宽规则、不解封 test**，停止并交用户决定。

## 4. 工作点阈值校准（选定 checkpoint 之后）

- 只允许在 **Smart Campus V3 val（B）** 上校准；MBDD val 仅承担遗忘监控。
- 预先固定的 conf 扫描点：
  `0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50`
- 匹配口径：**IoU = 0.5**，统计 detection P / R / F1（其余参数取本协议 §5 的冻结值）。
- **Primary：F1 最大**。
- Tie-break：多个阈值的 F1 距最大值 ≤ 0.005 → **优先 recall 更高**；仍相同 → **取较高阈值**。
- 本轮**不设计新的框数 guard**。
- 阈值一旦确定**立即冻结**，之后不得再调。

## 5. 评价参数（冻结，不允许按结果调整）

| 参数 | 值 |
|---|---|
| imgsz | 640 |
| NMS IoU | 0.45（沿用 V2 工作点取值） |
| max_det | 30（沿用 V2 工作点取值） |
| mAP 协议 | conf 0.001 / iou 0.7 / max_det 300（ultralytics 标准口径） |
| 工作点 P/R/F1 匹配 | IoU 0.5，贪心匹配 |
| class 顺序 | `0 peeling / 1 crack / 2 rust`（与 MBDD 一致） |

## 6. 训练配置（与 V2 对齐，不在本轮新增增强策略）

初始化 **`experiments/runs_v2/V2-AUG-RERUN1/weights/best.pt`**，
表述为 **continued domain-adaptation / fine-tuning from V2**——
**不得**写成"从官方预训练权重重新训练"。

| 参数 | 值 | 依据 |
|---|---|---|
| 训练数据 | MBDD `split_v2` train originals 7,752 + Smart Campus V3 train 1,536 = **9,288** | 只用原图，不用 V2 六倍离线增强集 |
| epochs | 30 | 用户指定 |
| imgsz / batch / device / workers | 640 / 32 / 0 / 8 | 对齐 V2 |
| seed / deterministic | 0 / True | 对齐 V2 |
| optimizer | `auto`（V2 亦为 auto） | 对齐 V2 |
| lr0 / lrf / momentum / weight_decay / warmup_epochs / cos_lr | 0.01 / 0.01 / 0.937 / 0.0005 / 3.0 / False | 对齐 V2 |
| 在线增强 | mosaic 1.0、scale 0.5、fliplr 0.5、flipud 0、hsv_h 0.015 / hsv_s 0.7 / hsv_v 0.4、translate 0.1、degrees 0、shear 0、perspective 0、erasing 0.4、close_mosaic 10、mixup 0、copy_paste 0 | 与 V2 完全相同（Ultralytics 默认项），**不新增策略** |
| save_period | **1** | 保留每个 epoch 的 checkpoint，供按本协议选择 |
| patience | 100（不触发早停） | 30 epoch 内不会早停；**不按任何 val/test 结果提前停止** |

**禁止加入**：MBDD split_v2 val、MBDD split_v2 test、Smart Campus V3 val、Smart Campus V3 test、
V2 六倍离线增强集、HU。

## 7. 冻结与解封

选择与阈值校准完成后，建立 `docs/evidence/v3_domain_adaptation/model_frozen/`，
记录选定 epoch、权重 SHA256、初始化权重 SHA256、合并训练清单与哈希、V3 split manifest SHA256、
本协议 SHA256、train args、脚本 SHA256、两域 val 指标、冻结阈值、NMS IoU、imgsz、max_det、
类别映射、冻结时间，并写明：

```
Smart Campus V3 test status: NOT USED
split_v2 test status in V3: NOT USED
frozen_before_first_v3_test_inference: true
```

**做到这里停下，不自动解封 test。**

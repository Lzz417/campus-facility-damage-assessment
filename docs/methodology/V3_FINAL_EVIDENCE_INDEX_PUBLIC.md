# V3 最终证据索引（**公开副本 / PUBLIC COPY**）

> **Public copy redacts the executor's real name for privacy.**
> **The original frozen provenance record is retained privately.**
>
> 本文件是私有冻结记录 `V3_FINAL_EVIDENCE_INDEX.md` 的**脱敏公开副本**，
> **不是**原始冻结文件。两者的关系与差异如下：
>
> | 项 | 原始冻结文件（私有） | 本公开副本 |
> |---|---|---|
> | 执行人姓名 | 原始记录中的实名 | 替换为 `project member` |
> | 科研事实 / 数字 / SHA256 | 完整 | **一字未改** |
> | 本机绝对路径 | 可能有 | 0 处（原文件实测） |
>
> 转换规则（可复核、可重现）：仅执行「执行人姓名 -> `project member`」一处替换（原姓名不在此公开副本中重复出现）；
> 其余内容（含所有数字与 SHA256）**逐字节保留**。

---

# V3 最终证据索引（V3_FINAL_EVIDENCE_INDEX）

**生成时间**：`2026-10-16T16:32:38+0800`
**性质**：**只整理已有证据，不产生任何新指标**；不修改任何 V2/V3 实验产物。
**用途**：作品报告 / 论文的唯一事实来源。报告里任何数字与表述都应能在本文件中找到出处；
**若报告与本文件冲突，以本文件为准**（本文件的数字全部来自冻结件，且附 SHA256）。

> **报告口径基线（先读这条）**：本文件中出现的「**校园**」一律是对 **Smart Campus Dataset 这个数据来源**的简称，**不构成对采集场所的断言**——该数据集只标注「Provided by a Roboflow user」，没有采集地点/方法说明。

## 目录

- [1. V3 启动与 V2 CLOSED 状态](#1-v3-启动与-v2-closed-状态)
- [2. Smart Campus 原始版本与来源限制](#2-smart-campus-原始版本与来源限制)
- [3. 数据完整性与污染审计](#3-数据完整性与污染审计)
- [4. 三类映射与筛除记录](#4-三类映射与筛除记录)
- [5. duplicate / mirror / same-view 审计](#5-duplicate--mirror--same-view-审计)
- [6. V3 split 冻结（test SEALED）](#6-v3-split-冻结test-sealed)
- [7. 模型选择协议（训练前预登记）](#7-模型选择协议训练前预登记)
- [8. V3 训练](#8-v3-训练)
- [9. 两次开发阶段评价失败及修复（纪律记录）](#9-两次开发阶段评价失败及修复纪律记录)
- [10. selected checkpoint 与 forgetting guard](#10-selected-checkpoint-与-forgetting-guard)
- [11. V2-on-Smart-Campus-val baseline（不训练、不改权重）](#11-v2-on-smart-campus-val-baseline不训练、不改权重)
- [12. threshold freeze（工作点）](#12-threshold-freeze工作点)
- [13. test comparison protocol（预登记）](#13-test-comparison-protocol预登记)
- [14. test unseal](#14-test-unseal)
- [15. V2 vs V3 final held-out test（**正式结果**）](#15-v2-vs-v3-final-held-out-test**正式结果**)
- [16. 方法学边界与不可宣称内容](#16-方法学边界与不可宣称内容)
- [附：report_numbers_table](#附report_numbers_table)

---

## 1. V3 启动与 V2 CLOSED 状态

### 关键事实

- V3 起点记录写在任何下载/划分/训练之前，钉住 V2 收口状态与机器环境。
- 仓库：分支 `main`，HEAD `ea15aed9fc023d5b97d245319282cc0990a4675f`（2026-08-27），工作树 dirty（3 个未跟踪条目：`docs/`、`experiments/runs_v2/`、`train/train_yolo_v2.py`）。
- **重要**：`docs/` 与 `experiments/runs_v2/` 从未提交过 ⇒ V3 产物只能靠 SHA256 追溯，不能靠 commit。
- 环境：Python 3.11.15 / torch 2.6.0+cu124 / ultralytics 8.4.129 / cv2 5.0.0；RTX 4080 SUPER 16,376 MiB，驱动 610.88。**本轮未安装/升级任何包。**
- V2 锚点（起点记录时实测）：detector `e810aeb06fff9628…`、split_v2 冻结清单 `297f1dc3335e0f01…`、`configs/config.yaml` `bf58f11a95d3f05f…`（`detect.conf` 保持历史值 0.25）。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `v3_start_record.md` | `afedb6bdc0a916e5842ec0990f7bc4293454809634f9e1c7f357a09861532d27` |
| `v3_start_record.json` | `bbfd358b36eef394f7fce587801045d90714f03e4000af2a5659b4472bf256c8` |

### 可以这样写（准确表述）

> V2 于 2026-10-16 收口；V3 是在 V2 冻结物只读前提下的新域适配实验。

### 不能这样写（禁止夸大）

> 不得称 V3 修改或重跑了 V2 的 split_v2 test；不得称 V2 的结论被 V3 取代。

---

## 2. Smart Campus 原始版本与来源限制

### 关键事实

- 投递 zip：`Smart Campus Dataset.v9-updated_original.yolov11.zip`，274,348,599 字节，SHA256 `866a2258f7a16ae4…`（完整值见 intake_record）。
- `data.yaml` 自述：workspace `emily-s-workspace`、project `smart-campus-dataset`、**version 9**、5 类（corrosion / cracks / paint_degradation / potholes / spalling）、License **CC BY 4.0**。
- Roboflow 导出时间 2026-10-16 06:43 GMT；预处理仅 EXIF 自动转正；**自述未做任何图像增强**。
- ⚠️ **来源描述只有一句「Provided by a Roboflow user」**：没有采集地点、没有采集方法、没有采集时间说明。
- 数据集**没有 bbox 之外的分割/mask 标注**，因此本轮不做分割评价。
- 下载过程：Roboflow Universe 对本机全部 403（Cloudflare 挑战），故由用户手工下载投递；数字均由 zip 本体核验，非页面声称。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `smart_campus/provenance/intake_record.json` | `9c82555b2ebccb74e8fd6616a77fec80f198f6922dcd31bda3a1abdaa2186ffc` |
| `smart_campus/provenance/README.roboflow.txt` | `979ce65c669bb76768f51505c1198b9e4eebdd92b75428fa760a660948c845af` |
| `smart_campus/provenance/README.dataset.txt` | `e707650bbdcfaa51200b2aba2ded8fc35ed7110768d6ec211b64d8878c6ad18e` |
| `smart_campus/provenance/data.yaml` | `efc3d312ae48b2c9ce57c4ddc6782fd0630f4c4de32e696dbc2644fa45116713` |
| `acquisition_log.md` | `2dff259b2b457ca73d173fdb3ea2a4145790432e720c64a73a23043b7f8a43fa` |

### 可以这样写（准确表述）

> Smart Campus Dataset（Roboflow 用户上传，version 9）是一个**公开的第二数据来源**。

### 不能这样写（禁止夸大）

> **不得**称其为「已证实采自真实校园的数据」；不得称其为校园采集数据集（页面无采集地点/方法说明）。报告中出现「校园」一词时，只能是**对该数据来源的简称**。

---

## 3. 数据完整性与污染审计

### 关键事实

- 规模：**4,323 张图 / 4,323 个 label / 6,602 个 bbox**；原始划分 train 3,026 / valid 648 / test 649。
- 完整性：有图无 label **0**、空标签 **0**、orphan label **0**、越界 bbox **0**、非法字段/类别 **0**、读不出的图 **0**、exact duplicate（SHA256）**0 组**。
- 逐类（图片/bbox）：corrosion 635/983、cracks 1083/1213、paint_degradation 871/1395、potholes 844/1478、spalling 1046/1533。
- **污染审计发现（这是本轮最重要的数据发现之一）**：`corrosion` 类含屏幕截图（`Captura-de-Tela-*`、`Screenshot-*`）、涂层**实验室试片**（`I###_steel-high-hard-_abrasive-blasted…MIL-DTL-53022/53039`）、**AI 生成图**（文件名前缀 `craiyon-`，共 5 张）、新闻/厂商文章配图、**带第三方水印的图库照**（shutterstock / alamy / iStock / depositphotos / ©2012 InspectAPedia；一张带 Wikipedia 署名）。
- ⇒ 数据集虽标注 CC BY 4.0，但**其中包含的第三方水印图版权不在上传者手里**，这是许可证风险，如实记录。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `smart_campus/audit/smart_campus_audit.md` | `44473d7fe706c28994c0215698d99f13106f2e159e58d69703d6110096426beb` |
| `smart_campus/audit/smart_campus_audit.json` | `9fd6a4babaae393889942354fd68b152e02ae45f04831e4bb1a906aae3727a37` |
| `smart_campus/audit/smart_campus_images_manifest.csv` | `a7539c52e120e52470cfb57cd6b37e821fa524045680a0da4120879584b10c0f` |
| `smart_campus/audit/smart_campus_labels_manifest.csv` | `5b1c023c7b4a19661277f9b176ee1774894bb46e4b918db01dfe16d7297da069` |
| `smart_campus/audit/smart_campus_class_stats.csv` | `d550dacd8052e7d8b40baf4c07c85461b8eb0d3ce1ec7346c8097e2aa1d23c3e` |

### 可以这样写（准确表述）

> 对第二来源做了完整性审计与污染审计，并据审计结果做了清洗。

### 不能这样写（禁止夸大）

> 不得称该数据集「干净」或「已核实」；不得回避其含 AI 生成图与第三方水印图这一事实。

---

## 4. 三类映射与筛除记录

### 关键事实

- 映射（class id 按 **MBDD 顺序**，取自 `data/splits_v2/data_orig.yaml`）：`paint_degradation → 0 peeling`、`cracks → 1 crack`、`corrosion → 2 rust`。
- `potholes` / `spalling` 本轮**不进训练**。
- 评审方式：**agent 视觉审计（不是人工复核）**；每类抽样 48 张、seed=0，对 corrosion 的**全部剩余 427 张**做了逐屏目视（18 屏）。
- 筛除（文件名规则 + agent 目视）：corrosion 635 → **274**（剔除 361，占 57%）；cracks 1083 → 1075；paint_degradation 871 → 866。
- 计数口径（已对账，见 v3_bookkeeping_checks.md）：**374 class-image exclusions → 365 unique images**（9 张图同时出现在两个类的剔除名单里；无图跨 3 类）。
- 用户裁决：**144 条目视剔除名单全部同意**，依据是**保守数据治理原则**。
- 重映射 label 2,194 个；保留框 peeling 1,389 / crack 1,202 / rust 498；**丢弃 potholes 0 个、spalling 160 个**。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `class_mapping_review.md` | `45decaee666d2909acc62834dc16747809e11f0c1ac6375965d5bac5cb7b45ec` |
| `class_mapping_review_v2.md` | `0d01249e4ca5bb8963453ba0de8da01284de2b99961bf317574f49708289f2d6` |
| `smart_campus/screening/excluded_corrosion.csv` | `90e8d5e3188193eba277e80a693fd7972404960f5a0599d92ab25c426ccab9f8` |
| `smart_campus/screening/excluded_cracks.csv` | `9fff61aedfc4ebd2cabf94a8b56366e5ef549624b744ba12b7ead5d45886c856` |
| `smart_campus/screening/excluded_paint_degradation.csv` | `d41f728b20c62c3d1d0be3951ba4b8191fbccecaf3160bbdf842571a6de430b5` |
| `smart_campus/screening/manual_visual_flags__corrosion.csv` | `9432be635cd4814e0b169d8d974ca947aef1f8f464489402c8f6f3e4431e8fc8` |
| `smart_campus/screening/manual_visual_flags__paint_degradation.csv` | `74af32506d041bd4f26c465b26222420b1b749d9e9ecce47261eb25cecd421cb` |
| `smart_campus/v3_bookkeeping_checks.md` | `e09309e4db06bb16538342893c8631b2f6af25d82e3d8f85bb120656b6dd3e25` |

### 可以这样写（准确表述）

> 类别映射由 AI 辅助视觉审计提出、经项目成员按保守数据治理原则批准；corrosion 类因污染严重剔除了约 57% 的样本。

### 不能这样写（禁止夸大）

> **不得**称该映射经过人工逐张视觉复核（实际是 agent 目视 + 成员按治理原则批准）；不得称筛除后的数据「无污染」（残留约 2% 水印图无法用规则清除）。

---

## 5. duplicate / mirror / same-view 审计

### 关键事实

- exact duplicate（SHA256）：**0 组**。
- 近重复：`min(dHash,pHash) ≤ 2` 先命中 805 对，但 dHash 只有 4,231/4,323 个不同取值 —— 说明是**哈希空间聚集**而非退化。故做了第二步像素级复核（32×32 归一化相关系数 corr）并逐屏看图。
- 看图结论：STRONG 确为同一张图、MODERATE 为同一缺陷不同机位，**weak 里也仍有同一缺陷**（机位/光照差更大）⇒ **不收窄阈值**，沿用 `≤2` 硬约束。
- **镜像副本是独立发现**：原哈希方案查不到水平翻转副本，补做镜像检查后确认存在（STRONG 21 对，corr=1.000，已目视）。
- **Roboflow 自带划分确认存在泄漏**（不是疑似）：至少 86 对目视确认的同一张图 + 7 对镜像同源横跨其 train/valid/test ⇒ **自带划分弃用**。
- 近重复统计口径：805 候选（STRONG 188 / MODERATE 236 / weak 381），跨 split 414 对；镜像候选 345 对（STRONG 21 / MODERATE 16 / weak 308）。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `smart_campus/near_dup/near_dup_pairs_reviewed.csv` | `707b3851b817db04790640283943797eba0af2d6984f3691cb7de0d0c9ba3320` |
| `smart_campus/near_dup/near_dup_groups.csv` | `5d4ec49b86784b2da08319b0ee767beed8d17c3186304653c9fd5bf4a1b4178c` |
| `smart_campus/near_dup/mirror_pairs.csv` | `f4802efcade699b1b179d15a68b4acad76f9ceb4f02c019493cb75b8722c87b8` |
| `smart_campus/near_dup/near_dup_review.md` | `81b25ba5d8249e4cc7f5fbee49bbb362dd0bcfcedac8dd17a38f888ed7a74f08` |
| `proposed_v3_split_plan.md` | `20465de6a06bdfa48fd8d63f4c99757c107e2d09f2515a67ac09c95d298f853a` |

### 可以这样写（准确表述）

> 原数据集的官方划分存在同源重复泄漏，因此弃用其划分、改为自建分组约束。

### 不能这样写（禁止夸大）

> 不得只写「未发现重复」——官方划分**确实**有泄漏；也不得把 weak 桶一笔带过（其中的同缺陷对是**不采用更严阈值**的理由）。

---

## 6. V3 split 冻结（test SEALED）

### 关键事实

- 池：4,323 → 剔除 365（清洗）+ 去掉 1,764 张只含 potholes/spalling 的图 = **2,194**。
- 约束组（union-find）：exact **0** 组 / hash_near(`≤2`) **539** 对 / mirror(`≤2`) **131** 对 / same_view(`2<min≤6` 且同尺寸) **718** 对 ⇒ **1,984 组**；单张组 1,915（96.5%）；**最大组 52 张 = 池的 2.37%**。
- 划分（组级、按目标比例 70/15/15、seed=0）：**train 1,536 / val 329 / test 329**（实际 70.0/15.0/15.0%）。
- 逐类 train/val/test（图/框）：peeling 601·952 / 131·213 / 134·224；crack 753·836 / 161·183 / 161·183；rust 195·348 / 39·68 / 40·82。三类均远超覆盖下限，**未触发修补**。
- 跨 split 检查（**冻结时从图片文件重算哈希**）：hash 近重复命中 465 次、镜像 92 次，**跨 split 均为 0**；exact 跨 split 0 组；constraint group 跨 split 0 个。
- ⚠️ **两套哈希实现**：建组用审计清单的 dHash/pHash（PIL 实现），冻结校验用另一套（cv2 `INTER_AREA`）——两套的重采样不同、命中数不同（539 vs 465、131 vs 92），但**两套下「跨 split = 0」都成立**。
- 命名纪律：这些组是 `duplicate_constraint_group` / `pseudo_scene_group`，**NOT physical building ID**。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `smart_campus/v3_split/v3_split_freeze.json` | `61ce8073491cd25ce42f87be36ece1e91e1d77252ca0bd7e33c51a7f6c3dd81e` |
| `smart_campus/v3_split/v3_split_freeze_manifest.csv` | `482c9de747646a12abef3c807803c319c13476911ce076ea3085009cf1f8d2b8` |
| `smart_campus/v3_split/v3_split_freeze.sha256` | `e16444805b8763759eb8a8ba60bbc92c1560034a8cc255e58142a20fd2d9d700` |
| `smart_campus/v3_split/smart_campus_v3_test_manifest.csv` | `5e792bce3682b5c2ae0fe123ecac658d0a20daa848376a8a5a7abd3adb7544f5` |
| `smart_campus/v3_split/smart_campus_v3_split_manifest.csv` | `a0497748aed05c3753cd7cbddac042f20949f023854b6430d9064e232936a0a4` |
| `smart_campus/v3_split/v3_split_summary.md` | `368e5253472582aea4f48f97a8a65622397379e7a13ace6ef779b458e503e21b` |
| `smart_campus/stage1_summary.md` | `4409b6f3652a3b1810ff8516db8cc7a7d2c4595aec31bc06cfff2343ce0feb34` |

### 可以这样写（准确表述）

> V3 划分由算法与固定 seed 生成、在训练前冻结；test 自冻结起 SEALED，不参与训练、早停、模型选择或阈值选择。

### 不能这样写（禁止夸大）

> 不得称这些组是真实建筑/场景 ID；不得称划分「无任何共源关联」（只能说在设定的判据下未发现跨 split 的近重复）。

---

## 7. 模型选择协议（训练前预登记）

### 关键事实

- 协议在任何训练开始之前写定并算哈希，之后未修改规则。
- guard：`MBDD split_v2 val mAP50 ≥ 0.0948`（= 基线 **0.1185** × 0.80）。
- 基线 0.1185 的出处：`docs/evidence/training_v2/training_v2_results.csv`（V2-AUG-RERUN1 @ split_v2/val，n=2833，post-hoc 口径）。**注意**：训练日志里 in-run 最佳是 0.1201，用 0.1185 是因为候选 checkpoint 也用 post-hoc evaluator 测，基线必须同口径。
- primary：Smart Campus V3 val mAP50 最大；tie band ≤0.005 → 比 mAP50-95 → 再取更早 epoch。
- 失败模式预登记：若无 checkpoint 过 guard，报告 `no checkpoint passed the pre-registered forgetting guard` 并停止、不放宽规则。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `v3_model_selection_protocol.md` | `66d1d3c948ebaea8bb468d35d039d2e6385a78cc8af4fa39af5c13b9b9712a65` |
| `v3_model_selection_protocol.sha256` | `676fe98a81d7ef74a8fb8604ceb753b9eb0b0be66b47938c3836acd7f47bcbde` |

### 可以这样写（准确表述）

> 模型选择规则在训练前预登记并冻结，训练后未修改。

### 不能这样写（禁止夸大）

> 不得称规则是事后为挑某个 checkpoint 而定的；不得把 in-run 0.1201 与 post-hoc 0.1185 混用。

---

## 8. V3 训练

### 关键事实

- 表述：**continued domain-adaptation / fine-tuning from V2**——初始化权重是 `experiments/runs_v2/V2-AUG-RERUN1/weights/best.pt`。
- 训练数据：MBDD `split_v2` train originals **7,752** + Smart Campus V3 train **1,536** = **9,288**（逐文件 sha256 见合并训练清单）。
- 未加入：split_v2 val、split_v2 test、Smart Campus V3 val、Smart Campus V3 test、V2 六倍离线增强集、HU。
- 参数与 V2 逐项对齐：`optimizer=auto`、`lr0/lrf 0.01`、`momentum 0.937`、`weight_decay 0.0005`、`warmup_epochs 3.0`、`imgsz 640`、`batch 32`、`device 0`、`workers 8`、`seed 0`、`deterministic=True`、`save_period=1`、`patience=100`、epochs 30。**未新增任何增强策略。**
- 30 epoch 全部跑满（未触发早停），耗时约 18 分钟；显存 4.5 GB（batch 32 无需下调）。
- **checkpoint 命名**：`save_period=1` 写出的是 `epoch0.pt … epoch29.pt`，对应人类 epoch 1…30。候选严格限定这 30 个。
- alias（按**权重张量指纹**判定，因字节并不相同）：`best.pt ≡ epoch 6`、`last.pt ≡ epoch 30`；两者的两域指标与对应 epoch **逐位相同**。**alias 不作为候选。**

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `experiments/runs_v3/V3-CAMPUS-ADAPT/args.yaml` | `323b9a79fbde34f6627282268625266d3ed9f637cdc9312a2acb2f0391a55bf6` |
| `experiments/runs_v3/V3-CAMPUS-ADAPT/results.csv` | `c5b7712af905990e265b278f039a03ef114f267a8a7c376f12657c831dbff3aa` |
| `data/v3_external/v3_campus_adapt.yaml` | `eb259b2dc45e9afe768c16f465b12bf3dcb788962acf909b67b946601c32279d` |
| `v3_training/combined_train_manifest.csv` | `bcf89b15e85387b996df07c1876b36e427284a0013b2fc398492aca5b7bebc34` |
| `v3_training/combined_train_manifest.sha256` | `b3fb5a0b5d657f13c1279b6736c318cae7143f22ef4718429e83daaa00e6ac5b` |
| `v3_training/weight_fingerprints.json` | `bf0e12b37f300b04eae93a30ceeeb03f89a8725d2275c9ba572129b68146dafb` |

### 可以这样写（准确表述）

> 在 V2 权重基础上，用原 MBDD 训练集加新的第二来源训练数据继续微调；超参与 V2 对齐，仅 epoch 数改为 30。

### 不能这样写（禁止夸大）

> **不得**写成「从官方预训练权重重新训练」；不得称用了 V2 的六倍增强集。

---

## 9. 两次开发阶段评价失败及修复（纪律记录）

### 关键事实

- **失败 #1**：`v3_sc_val.yaml` 缺 `train:` 键（Ultralytics 的 `check_det_dataset` 强制要求 train+val 同时存在）。失败阶段：第 1 个 checkpoint 的 A 域评价完成后、B 域开始前。性质：配置兼容性。
- **失败 #2**：我的补丁只改了参数元组、没改格式串，`%.0f` 收到空字符串。失败阶段：epoch1 两域都评价完成后打印进度行时。性质：工程失误。
- 两次**均在评价阶段**，未触及模型、开发集、指标公式与选择规则；失败运行的中间指标**全部排除**在最终 selection table 之外（只作废、归档）。
- 两次的失败版与修复版指纹、错误日志、失败阶段、影响面均已归档。
- 修复后 `per_epoch_eval/` **整体重跑**，最终表每一行都来自同一次修复版运行。
- 教训已固化成机制：进度格式提为模块常量并在循环前自检，此类错误几秒内暴露。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `v3_training/failed_attempt_1/ATTEMPT_NOTE.md` | `3ebad0a2c9632a9430ad4d75903831924b47f61e7bf890ce5a66ef6ef17bce02` |
| `v3_training/failed_attempt_2/ATTEMPT_NOTE.md` | `115b86e5c7e06d63f94e13b5e6779e07511772a729a46523d625dd4187405dd2` |

### 可以这样写（准确表述）

> 开发阶段发生两次**程序性**失败（非科学问题），均已按纪律记录、修复并整体重跑。

### 不能这样写（禁止夸大）

> 不得称「全程无失败」；这两次失败必须与 test 阶段的「无技术重跑」分开陈述。

---

## 10. selected checkpoint 与 forgetting guard

### 关键事实

- 候选 30 个 epoch checkpoint；**27 个通过 guard**（未过：epoch 5 = 0.0789、epoch 7 = 0.0909、epoch 14 = 0.0941）。
- primary 最大 = **0.7024 @ epoch 25**；与 epoch 27（0.6995）相差 0.0029 ≤ 0.005 → 进入 secondary。
- secondary：epoch 27 的 mAP50-95 **0.4671 > 0.4563** ⇒ **选中 epoch 27**（`weights/epoch26.pt`）。
- 选中模型的开发集指标：
    · A = MBDD split_v2 val（2,833）：mAP50 **0.1064**、mAP50-95 0.0492、P 0.3161、R 0.1351；逐类 AP50 peeling 0.0834 / crack 0.0496 / rust 0.1862。
    · B = Smart Campus V3 val（329）：mAP50 **0.6995**、mAP50-95 0.4671、P 0.6483、R 0.7065；逐类 AP50 peeling 0.7778 / crack 0.8571 / rust 0.4636。
- **forgetting guard：通过。** MBDD val mAP50 由 V2 基线 **0.1185 → 0.1064**，相对 **−10.2%**，未超过 −20% 上限。
- ⚠️ **读法**：B 的 0.70 是**对新来源的拟合**，不是跨域泛化证据——B 的 329 张与 V3 训练用的 1,536 张**同源同次采集**。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `v3_training/selection.json` | `03a7d7e7e0001cba984db0793339cb4da7a3dc0ac77ec2761426cef94966e6d7` |
| `v3_training/selection.md` | `c4e29e2bb528a778122637d4b12daaa127a3088e1a66a142f2837afc917b53d2` |
| `v3_training/selection_table.csv` | `8af6dbb4482972ee982a60d0c74de3421b7a400d09188e75f9387292a5b9ab62` |

### 可以这样写（准确表述）

> 按预登记规则从 30 个候选中选出 epoch 27；旧域 MBDD val 相对下降约 10.2%，仍在预登记 guard 内。

### 不能这样写（禁止夸大）

> 不得把 Smart Campus val 的 0.70 说成「跨域泛化提升」；不得隐藏旧域下降（必须同时给出 −10.2%）。

---

## 11. V2-on-Smart-Campus-val baseline（不训练、不改权重）

### 关键事实

- 用**同一个 evaluator、同一个 val 集（329）**把 V2-AUG-RERUN1 测一遍。
- V2 baseline：mAP50 **0.0255**、mAP50-95 **0.0110**；逐类 AP50 peeling 0.0531 / crack 0.0235 / **rust 0.0000**。
- V2 在该 val 上 **rust 一个都没检出**（AP50 = 0，各阈值下 TP 全为 0）。
- 工作点阈值按**与 V3 同一段代码**（`_scripts/select_threshold.py`）在同一 val 上校准：V2 → **conf 0.050**；V3 → **conf 0.200**。
- 选阈规则一致性已验证：用该函数重推 V3 的阈值，**复现出 0.20**（不是嘴上说同一套规则）。
- 权重 SHA256 `e810aeb06fff9628de1bd8ebafe8d103f29ac25cbf2ed39fb76a7f4595955765`（未改动）。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `model_frozen/v2_on_sc_val_baseline.json` | `60fc23b3b2b637788e532a21539bfa545c13e2985e488f56705d59301a54e29f` |
| `model_frozen/v2_on_sc_val_threshold_scan.csv` | `c661af9fd4b846d620a4234a161b17c50133ac85bccb42ffddf24b7f0f7f009c` |
| `model_frozen/v2_on_sc_val_baseline_summary.md` | `ff5308a5fad3cf95f15f0948465d42b6c9103e38a1a0214bd29c622fa3e89863` |

### 可以这样写（准确表述）

> `V2 在 Smart Campus 数据来源的 held-out val 上对 rust 的 AP50 为 0，表明原 V2 对该新来源的 rust 类泛化极弱。`

### 不能这样写（禁止夸大）

> **不得**写成「V2 没见过校园域」——那等于宣称这是已核实的校园域。

---

## 12. threshold freeze（工作点）

### 关键事实

- 固定 10 点扫描：0.025 / 0.05 / 0.075 / 0.10 / 0.15 / 0.20 / 0.25 / 0.30 / 0.40 / 0.50；匹配 IoU 0.5；NMS iou 0.45；max_det 30；imgsz 640。
- V3：F1 最大 0.7500 @ 0.25；≤0.005 并列 `[0.20, 0.25, 0.30]` → recall 最高者唯一为 0.20 ⇒ **冻结 conf = 0.200**（P 0.7339 / R 0.7608 / F1 0.7471）。
- V2：F1 最大 0.1073 @ 0.075；≤0.005 并列 `[0.05, 0.075, 0.10, 0.15]` → recall 最高者 ⇒ **冻结 conf = 0.050**。
- 选阈代码 `select_threshold.py` 为 V2/V3 共用；规则来自预登记协议第 4 节。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `v3_training/threshold_sweep_sc_val.json` | `6d2adbd6e9612d761d8c2446084184142bf86f1e8752775c9f2cf38ea9c7d77c` |
| `_scripts/select_threshold.py` | `58fb96809c387c94e989070f818a797aba37e2257f5a947454efc074072fbb4d` |
| `model_frozen/v3_model_freeze.json` | `67a58023396aa56a7797b7ad817c3c7fc80cf0495f832019f73a27e324c5dc47` |
| `model_frozen/v3_model_frozen_summary.md` | `70ed451fc1ac35b9708aec5effe2f9759f93a1e63edbadb1ccf8a0930b390186` |

### 可以这样写（准确表述）

> 两个模型的工作点阈值是**在同一个 val 上、按同一条规则、用同一段代码分别校准**的。

### 不能这样写（禁止夸大）

> 不得称两者用的是同一个阈值；不得把工作点指标与 standard mAP 混为一谈。

---

## 13. test comparison protocol（预登记）

### 关键事实

- 在任何 test 推理之前写定：一次解封、同时评价两个事先固定的模型。
- Baseline = V2-AUG-RERUN1；Adapted = V3-CAMPUS-ADAPT epoch 27。
- primary = standard mAP50 / mAP50-95 / per-class AP50，两者逐项相同：conf 0.001 / NMS IoU 0.7 / max_det 300 / imgsz 640 / 同一 evaluator / 同一 test manifest。
- 工作点另报：P/R/F1、per-class、mean boxes/image、no-prediction image rate；V2 用 0.050、V3 用 0.200。
- 禁止：改模型/阈值/split/mapping/evaluator、删困难样本、按 test 表现选模型、跑 MBDD split_v2 test、本轮加 SAM2。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `v3_test_comparison_protocol.md` | `99b49d708d17222b72baa2b4d36af746574c76bf7ef84faf62e5428920a3dd8c` |
| `v3_test_comparison_protocol.sha256` | `a1916ed5a9b664631bd4f70e8c64f71ab91c4a0d7846e8ffb59ac46d6622e2dd` |

### 可以这样写（准确表述）

> 比较协议在 test 推理前冻结，两个模型用完全相同的评价参数。

### 不能这样写（禁止夸大）

> 不得称比较参数是事后为了好看而选的。

---

## 14. test unseal

### 关键事实

- 拆封时间 **2026-10-16T16:23:15+0800**；`unsealed_by` = **project member**。
- 首个 test 产物时间 **16:23:31** —— **晚于拆封记录 16 秒**，机制上保证「先落盘拆封记录、再构造模型」。
- 记录内容：test manifest SHA256、V2/V3 权重 SHA256、V2/V3 冻结阈值、evaluator SHA256、比较协议 SHA256。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `model_frozen/v3_test_unseal.json` | `45c3aba75c1c89f0ed37f19d57bc642365b2848c0f1dffa8ace47b6d260c73b4` |

### 可以这样写（准确表述）

> test 在完成模型与阈值冻结后才解封，且只在本次比较中使用一次。

### 不能这样写（禁止夸大）

> 不得称 test 曾用于任何选择；`unsealed_by` 必须写真人实名，不得填 agent 名称。

---

## 15. V2 vs V3 final held-out test（**正式结果**）

### 关键事实

- test = Smart Campus V3 test，**329 张**；运行退出码 **0**。
- **无任何技术重跑**（`v3_test_reruns.csv` 只有表头）。
- primary（conf 0.001 / IoU 0.7 / max_det 300 / imgsz 640）：
    · mAP50：V2 **0.0353** → V3 **0.6533**（Δ **+0.6180**）
    · mAP50-95：V2 **0.0153** → V3 **0.4283**（Δ **+0.4130**）
    · per-class AP50：peeling 0.0945 → 0.8047（+0.7102）；crack 0.0112 → 0.8113（+0.8001）；rust 0.0001 → 0.3439（+0.3438）
- 工作点（V2@0.050 / V3@0.200）：P 0.0964/0.6984、R 0.1043/0.7055、**F1 0.1002/0.7019**；mean boxes/image 1.6079/1.5015；**no-prediction image rate 0.4286/0.0456**。
- V2 工作点下 rust 的 P/R/F1 全为 0（tp=0 fp=9 fn=82）。

### 对应文件与 SHA256

| 文件 | SHA256 |
|---|---|
| `v3_test_eval/v3_test_results_summary.md` | `bcd45ed0377e64241b660512837e4cc31e491163369a06612e0af090125e7a6c` |
| `v3_test_eval/v3_test_comparison_results.json` | `77e64ef4cac7985ecd04a60803f56c12f3617365511b43988d0a2d1d63ce93b6` |
| `v3_test_eval/v3_test_reruns.csv` | `4ad8b3f1ea5e43507896bff7871b57c31cc1c2230d263a6c361ca5d63f6d71f2` |
| `v3_test_eval/v3_test_eval_freeze_manifest.csv` | `671433ac0b5607ffe6383a368c4b90593af1c7f51a461fc687e80d29b2a9fc86` |

### 可以这样写（准确表述）

> 在预冻结的 held-out test 上，针对新来源做受控域适配后 mAP50 由 0.0353 升至 0.6533，说明**对该来源**的适配显著有效。

### 不能这样写（禁止夸大）

> **不得**称「证明了普遍校园泛化能力」；不得称 test 是独立外部来源验证（它与 V3 训练数据同源）；不得称系统适用于所有校园。

---

## 16. 方法学边界与不可宣称内容

### 关键事实

- ① **来源未核实**：Smart Campus 只标注「Provided by a Roboflow user」，无采集地点/方法说明 ⇒ 不能称其为已证实的真实校园数据。
- ② **同源**：V3 的 train/val/test 与 V3 的适配训练数据来自同一数据集同一次采集 ⇒ test 结果衡量的是「适配到该来源」，不是跨来源泛化。
- ③ **无独立外部来源**：HU Infrastructure Cracks 本轮**搁置**（v4 只含 8 个 record、且无 bbox 标注、只有 presence 级评价能力）⇒ 本轮**没有**独立外部验证。
- ④ **只验证 detector**：Smart Campus **无人工像素 GT**，本轮**未**重新验证 SAM2 分割性能 ⇒ 分割结论仍是 V2 那 60 张参考集的结果，不随 V3 改变。
- ⑤ **类别映射**由 AI 审计提出、按治理原则批准，且 corrosion 剔除了约 57%。
- ⑥ **test 规模小**（329 张、单一来源、单一光照）⇒ 置信区间宽。
- ⑦ **旧域代价真实存在**：MBDD val mAP50 相对下降约 10.2%。
- ⑧ 系统是**研究原型**，不是结构安全评估系统；输出不能用于工程安全性判定。

### 对应文件与 SHA256

（本节为边界声明，无对应文件）

### 可以这样写（准确表述）

> V3 表明「针对特定新来源做受控域适配能显著改善该来源内的检测性能」，同时旧域出现约 10% 的遗忘；该结论严格限定于本研究使用的那一个数据来源。

### 不能这样写（禁止夸大）

> 不得写「证明系统已经适用于所有校园」；不得写「解决了跨场景泛化问题」；不得把 detector 的改善外推到分割性能或结构安全结论。

---

## 附：report_numbers_table

**报告允许使用的全部数字。** 出处列指向本文件的对应小节；
任何未出现在本表中的 V2/V3 数字，写进报告前必须先补进本表。

| 分类 | 指标 | 值 | 单位/备注 | 出处 |
|---|---|---|---|---|
| 数据规模 | Smart Campus 原始图片数 | **4323** | 张 | audit |
| 数据规模 | Smart Campus 原始 bbox 数 | **6602** | 个 | audit |
| 数据规模 | 官方划分 train/valid/test | **3026 / 648 / 649** | 张 | audit（已弃用） |
| 清洗 | 剔除（unique images） | **365** | 张 | v3_bookkeeping_checks |
| 清洗 | 剔除（class-image 计数） | **374** | 个 | v3_bookkeeping_checks |
| 清洗 | 只含 potholes/spalling 而排除 | **1764** | 张 | split build |
| 清洗 | corrosion 类剔除比例 | **57** | % | screening |
| 清洗 | V3 映射池 | **2194** | 张 | split freeze |
| 划分 | train / val / test | **1536 / 329 / 329** | 张 | split freeze |
| 划分 | 实际比例 | **70.0 / 15.0 / 15.0** | % | split freeze |
| 划分 | constraint group 数 | **1984** | 组 | split build |
| 划分 | 最大组 | **52** | 张（2.37%） | split build |
| 划分 | 跨 split 近重复（两套哈希实现） | **0 / 0** | 对 | split freeze 校验 |
| 训练 | 训练图总数（MBDD 7752 + 新来源 1536） | **9288** | 张 | combined train manifest |
| 训练 | epochs | **30** |  | args.yaml |
| 模型选择 | 过 forgetting guard 的候选 | **27 / 30** | 个 | selection.json |
| 模型选择 | selected epoch | **27** |  | selection.json |
| 模型选择 | guard 阈值 | **0.0948** | = 0.1185×0.80 | protocol |
| 开发集 | MBDD val mAP50（V2 基线） | **0.1185** |  | training_v2_results.csv |
| 开发集 | MBDD val mAP50（V3 epoch 27） | **0.1064** |  | per_epoch_eval |
| 开发集 | 旧域相对变化 | **-10.2** | % | 计算 |
| 开发集 | Smart Campus val mAP50（V2） | **0.0255** |  | v2_on_sc_val_baseline |
| 开发集 | Smart Campus val mAP50（V3） | **0.6995** |  | per_epoch_eval |
| 工作点 | V2 冻结阈值 | **0.050** |  | v2_on_sc_val_baseline |
| 工作点 | V3 冻结阈值 | **0.200** |  | v3_model_freeze |
| test | test 集规模 | **329** | 张 | test manifest |
| test | mAP50（V2 / V3） | **0.0353 / 0.6533** | Δ +0.6180 | test results |
| test | mAP50-95（V2 / V3） | **0.0153 / 0.4283** | Δ +0.4130 | test results |
| test | peeling AP50（V2 / V3） | **0.0945 / 0.8047** | Δ +0.7102 | test results |
| test | crack AP50（V2 / V3） | **0.0112 / 0.8113** | Δ +0.8001 | test results |
| test | rust AP50（V2 / V3） | **0.0001 / 0.3439** | Δ +0.3438 | test results |
| test | 工作点 F1（V2 / V3） | **0.1002 / 0.7019** |  | test results |
| test | no-prediction image rate（V2 / V3） | **42.86 / 4.56** | % | test results |
| test | 技术重跑次数 | **0** |  | v3_test_reruns.csv |

### 明确不得出现在报告里的数字

- Smart Campus val 的 0.6995 **不等于**跨域泛化能力（它是同源 val）；
- 任何「其余校园/其它场景」的性能数字（本轮没有测试过其它来源）；
- 任何分割（SAM2）在 V3 上的性能数字（本轮未重测分割）；
- 任何把 test 的 +0.6180 外推为「普遍提升」的说法。

### 本文件的生成方式

由 `_scripts/build_final_evidence_index.py` 生成；所有 SHA256 在生成时**从文件回算**，并经校验与冻结件清单一致，避免手工转录错误。


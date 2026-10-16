# V3 域适配 —— 公开复现脚本

本目录是 V3（Smart Campus 来源域适配）**核心链路**的公开脚本副本。
它们与冻结证据里的原始脚本**逐字节相同**（哈希对照见 `SHA256.txt`），
只是复制到公开位置以便复现；**原件仍保留在私有证据目录中、未被移动或修改**。

> **注意**：这些脚本**不自动执行任何 test 评价**。`eval_v3_checkpoints.py --set sc_test`
> 与 `run_v3_test_comparison.py` 会读取**已封存的 test**，属于一次性正式评价，
> 只有在**明确授权解封**后才可运行（见下）。

---

## 1. 链路与顺序

| 顺序 | 脚本 | 作用 |
|---|---|---|
| 1 | `intake_dataset.py` | 外部数据入库：记 zip 指纹、安全解压（拒绝路径穿越）、留档来源文本 |
| 2 | `audit_smart_campus.py` | 数据完整性与污染审计（缺图/越界框/重复/逐类统计） |
| 3 | `screen_foreign_images.py` | 筛除外来图像（截图 / 实验室试片 / AI 生成图 / 图库素材） |
| 4 | `analyze_near_duplicates.py` | 近重复二次判定（dHash/pHash → 32×32 归一化相关）+ 镜像副本检查 |
| 5 | `build_v3_split.py` | **构建 V3 划分**：类别映射 → union-find 约束组 → 组级 70/15/15 |
| 6 | `freeze_v3_split.py` | **校验并冻结划分**（重算哈希、跨 split 泄漏检查）→ test SEALED |
| 7 | `prepare_v3_training_data.py` | 物化可训练目录 + 生成 data yaml + 合并训练清单 |
| 8 | `train_v3_campus_adapt.py` | **V3 训练**（continued domain-adaptation / fine-tuning from V2） |
| 9 | `eval_v3_checkpoints.py` | post-hoc evaluator：单个 checkpoint 的 mAP / 工作点阈值扫描 |
| 10 | `eval_all_checkpoints.py` | 对全部 epoch checkpoint 在两个开发集上批量评价 |
| 11 | `select_v3_checkpoint.py` | **按预登记规则选 checkpoint**（primary + forgetting guard） |
| 12 | `select_threshold.py` | 工作点阈值选择规则（**V2 与 V3 共用同一段代码**） |
| 13 | `freeze_v2_sc_val_baseline.py` | 冻结 V2 在同一 val 上的 baseline（不训练、不改权重） |
| 14 | `freeze_v3_model.py` | **冻结最终模型**（权重/协议/清单哈希 + 工作点） |
| 15 | `run_v3_test_comparison.py` | **最终 test 比较 runner**（一次解封、两个模型、同一 evaluator） |

## 2. 运行前的路径与配置

这些脚本**默认位于私有证据目录**，用 `Path(__file__).resolve().parents[4]` 推断仓库根。
复制到 `scripts/v3/` 后层级变了**一层**，因此直接运行前需要：

- 传入显式参数（大多数脚本都支持 `--dataset-root` / `--images-manifest` / `--out` 等），**推荐这种方式**；
- 或按需调整脚本内的 `ROOT` 推断。

`train_v3_campus_adapt.py` 通过 `src/` 与 `configs/` 工作，**不需要**私有数据即可阅读；
但训练需要有 MBDD 与 Smart Campus 数据（**均不随仓库分发**，见仓库根 `weights/README.md` 与
`THIRD_PARTY_NOTICES.md`）。

## 3. 关于已封存的 test（**重要**）

- `eval_v3_checkpoints.py` 内含 `sc_test` 数据集条目，**只有显式传 `--set sc_test` 才会读 test**；
- `run_v3_test_comparison.py` 是正式入口，它有三道闸：
  ① 必须给 `--unsealed-by`（真人实名，**我们不代填**）；
  ② 校验比较协议 SHA256 与 sidecar 一致（协议被改过就拒绝）；
  ③ **先写 `v3_test_unseal.json`、再构造模型**（保证拆封记录在第一次推理之前）。
- **本仓库不分发 test 图像**，因此即使运行这些脚本，也需要你自己准备数据；
  这些脚本的作用是**公开方法与流程**，不是提供可一键复跑 test 的完整环境。

## 4. 与 V2 历史脚本的区别

`src/pipeline.py` 的 `frozen_compat` 默认 `False`（保持历史行为，供历史实验复现）；
Web demo 显式传 `True`。**本目录的 V3 脚本不走 `run_pipeline`**——
V3 的评价链自己实现放置逻辑（理由与 V2 冻结评价相同：`run_pipeline` 的默认行为与冻结口径不同）。
